# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   mim_train_pipeline.py
    Time:        2022/09/28 09:18:07
    Editor:      Figo
-----------------------------------
'''

import os
import torch
import argparse
import torchvision as tv

# a vit based encoder MIM model
from vit_pytorch.simmim import SimMIM
from torch.utils.data import DataLoader
from models.mae import MAE
from models.vit import VIT_MODEL
from torch.utils.tensorboard import SummaryWriter
from utils.utils_fun import get_seed, parse_json, parse_train_tools
from utils.utils_train import mim_train, mim_test
from utils.utils_data import load_dataset


def get_argparse():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--seed', default=888, type=int, help='random seed')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epoch', default=30, type=int, help='total epochs to run')
    parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--num_classes', default=10, type=int, help='The number of classes in dataset')
    parser.add_argument('--pretrained', default=1, type=int, help='If pretrain the model, default not')
    parser.add_argument('--dataset', default='imagenet', type=str, help="dataset")
    parser.add_argument('--criterion', default='CE', type=str, help="Loss function")
    parser.add_argument('--scheduler', default='Cosine', type=str, help="The scheduler of optimizer")
    parser.add_argument('--optimizer', default='Adam', type=str, help="The optimizer of training")
    parser.add_argument('--model', default="mae", type=str, help='The type of loaded models, mae, simmim')
    parser.add_argument('--encoder', default='vit_base_patch16_384', type=str, help="The type of encoder, please read README.md")
    
    args = parser.parse_args()
    return args


def pipeline(args, model, loader, optimizer, scheduler, device="cuda:0", writer=None):

    trainloader, testloader = loader

    with open(new_folder+"/Record.txt", 'a') as file:
        params = new_folder.split('/')[-1].split('-')
        for key, value in zip(['encoder', 'model', 'dataset'], params):
            file.write("{}: {}\n".format(key, value))
        file.write('='*20+"\n\n")
    file.close()

    # main training pipeline
    for epoch in range(0, args.epoch):
        model.train()
        loss_train = mim_train(epoch, model, trainloader, optimizer, writer, device, True)
        
        with torch.no_grad():
            model.eval()
            loss_test = mim_test(epoch, model, testloader, writer, device, True)   
        
        def save_checkpoint(name="best"):
            checkpoints = {
                "model": model.state_dict(), 
                "epoch": epoch, 
                "loss": [loss_train, loss_test], 
            }
            torch.save(checkpoints, new_folder+"/cpt-{}.pt".format(name))

            with open(new_folder+"/Record.txt", 'a') as file:
                file.write(
                    "Epoch-{} \t Train Loss:{:.4f} \t Test Loss:{:.4f} \n".format(
                        epoch, loss_train, loss_test
                    )
                )
                file.write("\n\n")
            file.close()

        save_checkpoint("best")
        print("=> Finish one epoch saving...")

        if scheduler is not None: scheduler.step()
    
    print("=> Finish main pipeline training...")

    save_checkpoint("final")
    print("=> Finish last epoch saving...")
    
    with open(new_folder+"/Record.txt", 'a') as file:
        file.write("="*10 + " Config " + "="*9 + "\n")
        for keys, values in vars(args).items():
            file.write("=> {}:{}\n".format(keys, values))
        file.write("="*8 + " Finish record " + "="*8 + "\n")
    file.close()


if __name__ == "__main__":

    #! name of experiment's folder
    global new_folder

    #! Load and update json files
    args = get_argparse()
    json_models = parse_json(json_file="./config/mim_config.json", types=args.model)
    json_tools = parse_json(json_file="./config/mim_config.json", types='tools')
    vars(args).update(dict(json_models, **json_tools))
    
    #! Set seed and device
    get_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_index
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=> GPU {args.cuda_index} will be used...")

    #! Load model
    print(f"=> Load model {args.model}...")
    encoder = VIT_MODEL[args.encoder]
    if args.model == "mae":
        model = MAE(
            encoder=encoder, 
            decoder_dim=args.decoder_dim,
            masking_ratio=args.masking_ratio,               # 0.75
            decoder_depth=args.decoder_depth,               # 1
            decoder_heads=args.decoder_heads,               # 8
            decoder_dim_head=args.decoder_dim_head,         # 64
        )
    elif args.model == "simmim":
        model = SimMIM(
            encoder=encoder, 
            masking_ratio=args.masking_ratio,               # 0.5
        )
    model = torch.nn.DataParallel(model).to(DEVICE)
    print(f"=> Model {args.model} is OK...")

    #! Load dataset
    # an easy data augmentation also achieves a better performance
    print(f"=> Load dataset {args.dataset}...")
    transform_train = tv.transforms.Compose([
        tv.transforms.Resize((224, 224)),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.RandomVerticalFlip(), 
        tv.transforms.ToTensor(),
    ])
    transform_test = tv.transforms.Compose([
        tv.transforms.Resize((224, 224)),
        tv.transforms.ToTensor(),
    ])
    
    if args.dataset == "cifar10":
        train_set, test_set = load_dataset("cifar10", "custom", transform_train, transform_test)
        train_loader, test_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=True), \
                                    DataLoader(test_set, batch_size=64, num_workers=4, shuffle=False, pin_memory=True)
    elif args.dataset == "imagenet":
        dataset = load_dataset("imagenet", "custom", transform_train, transform_test)
        train_loader, test_loader = dataset.make_loaders(workers=8, batch_size=args.batch_size)
    print("=> DataLoader is OK...")

    #! Set criterion and optimizer and scheduler
    _, optimizer, scheduler = parse_train_tools(args, model)
    
    E_M_D = '-'.join([args.encoder, args.model, args.dataset])
    new_folder = "./checkpoint/{}".format(E_M_D)
    if not os.path.exists(new_folder): 
        os.mkdir(new_folder)
        print(f"=> Model will be saved in {new_folder}")
    # else: 
    #     raise ValueError(f"=> Model {E_M_D} is well trained, please change another seed if you want do it again.")
    
    #! Set tensorboard writer
    writer = SummaryWriter(log_dir=new_folder+"/runs", flush_secs=60)
    
    # #! main training pipeline
    pipeline(args, model, (train_loader, test_loader), optimizer, scheduler, DEVICE, writer)