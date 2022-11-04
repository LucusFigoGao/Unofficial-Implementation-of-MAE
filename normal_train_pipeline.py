# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   normal_train_pipeline.py
    Time:        2022/09/27 14:27:02
    Editor:      Figo
-----------------------------------
'''

import os
import torch
import argparse
import torch.nn as nn
import torchvision as tv

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from timm.models import create_model
from utils.utils_fun import get_seed, parse_json, parse_train_tools
from utils.utils_train import normal_train, normal_test
from utils.utils_data import load_dataset


def get_argparse():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--model', default="vit", type=str,help='model type (default: ResNet18)')
    parser.add_argument('--seed', default=888, type=int, help='random seed')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--epoch', default=30, type=int, help='total epochs to run')
    parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--num_classes', default=10, type=int, help='The number of classes in dataset')
    parser.add_argument('--pretrained', default=1, type=int, help='If pretrain the model, default not')
    parser.add_argument('--criterion', default='CE', type=str, help="Loss function")
    parser.add_argument('--scheduler', default='Cosine', type=str, help="The scheduler of optimizer")
    parser.add_argument('--optimizer', default='Adam', type=str, help="The optimizer of training")
    parser.add_argument('--json_type', default='conv', type=str, help="The type of loaded models, transformer, conv")
    args = parser.parse_args()
    return args


def pipeline(args, model, loader, criterion, optimizer, scheduler, device, writer):

    best_acc_score = float('nan')
    trainloader, testloader = loader

    with open(new_folder+"/Record.txt", 'a') as file:
        params = new_folder.split('/')[-1].split('-')
        for key, value in zip(['seed', 'model', 'criterion', 'optimizer', 'scheduler'], params):
            file.write("{}: {}\n".format(key, value))
        file.write('='*20+"\n\n")
    file.close()

    # main training pipeline
    for epoch in range(0, args.epoch):
        model.train()
        top1_train, loss_train = normal_train(epoch, model, trainloader, criterion, optimizer, writer, device, True)
        
        with torch.no_grad():
            model.eval()
            top1_test, loss_test = normal_test(epoch, model, testloader, criterion, writer, device, True)   
        
        def save_checkpoint(name="best"):
            checkpoints = {
                "model": model.state_dict(), 
                "epoch": epoch, 
                "loss": [loss_train, loss_test], 
                "acc": [top1_train, top1_test]
            }
            torch.save(checkpoints, new_folder+"/cpt-{}.pt".format(name))

            with open(new_folder+"/Record.txt", 'a') as file:
                file.write(
                    "Epoch-{} \t Train Loss:{:.4f} \t Test Loss:{:.4f} \t Train Accuracy:{:.2f}% \t Test Accuracy:{:.2f}%\n".format(
                        epoch, loss_train.item(), loss_test.item(), top1_train, top1_test
                    )
                )
            file.write("\n\n")
            file.close()

        if top1_test > best_acc_score:
            best_acc_score = top1_test
            save_checkpoint("best")
        
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
    json_models = parse_json(json_file="./config/train_config.json", types=args.json_type)
    json_tools = parse_json(json_file="./config/train_config.json", types='tools')
    vars(args).update(dict(json_models, **json_tools))
    
    #! Set seed and device
    get_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_index
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # #! Load model
    print(f"=> Load model {args.model}")
    if args.model == "resnet18":
        weights = tv.models.ResNet18_Weights.IMAGENET1K_V1 if args.pretrained else None
        model = tv.models.resnet18(weights)
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)  # change the number of class 
    if args.model == "resnet50":
        weights = tv.models.ResNet50_Weights.DEFAULT if args.pretrained else None
        model = tv.models.resnet50(weights)
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)  # change the number of class
    elif args.model == "vit":
        model = create_model("vit_base_patch16_384", pretrained=args.pretrained)
        model.head = nn.Linear(model.head.in_features, args.num_classes)
    model = torch.nn.DataParallel(model).to(DEVICE)
    print(f"=> Model {args.model} is OK...")

    #! Load dataset
    transform_train = tv.transforms.Compose([
        tv.transforms.RandomCrop(32, padding=4),
        tv.transforms.Resize(384),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = tv.transforms.Compose([
        tv.transforms.Resize(384),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    if args.model == 'vit':
        train_set, test_set = load_dataset("cifar10", "custom", transform_train, transform_test)
    else:
        train_set, test_set = load_dataset("cifar10", "default")
    train_loader, test_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=True), \
                                DataLoader(test_set, batch_size=64, num_workers=4, shuffle=False, pin_memory=True)
    print("=> DataLoader is OK...")

    #! Set criterion and optimizer and scheduler
    criterion, optimizer, scheduler = parse_train_tools(args, model)
    
    S_M_C_O_S = '-'.join([str(args.seed), args.model, args.criterion, args.optimizer, args.scheduler])
    new_folder = "./checkpoint/{}".format(S_M_C_O_S)
    if not os.path.exists(new_folder): 
        os.mkdir(new_folder)
        print(f"=> Model will be saved in {new_folder}")
    else: 
        raise ValueError(f"=> Model {S_M_C_O_S} is well trained, please change another seed if you want do it again.")
    
    #! Set tensorboard writer
    writer = SummaryWriter(log_dir=new_folder+"/runs", flush_secs=60)
    
    #! main training pipeline
    pipeline(args, model, (train_loader, test_loader), criterion, optimizer, scheduler, DEVICE, writer)