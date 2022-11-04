# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   mim_demo.py
    Time:        2022/10/08 09:52:28
    Editor:      Figo
-----------------------------------
'''

import os
import torch
import argparse

# a vit based encoder MIM model
from models.mae import MAE
from models.vit import VIT_MODEL
from vit_pytorch.simmim import SimMIM
from utils.utils_data import LoadImage
from utils.utils_fun import get_seed, parse_json, update_args, Ten2Img


def get_argparse():
    parser = argparse.ArgumentParser(description='PyTorch MIM-IMAGENET Training')
    parser.add_argument('--dataset', default='imagenet', type=str, help="dataset")
    parser.add_argument('--model', default="mae", type=str, help='The type of loaded models, mae, simmim')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    #! Load and update json files
    args = get_argparse()
    json_models = parse_json(json_file="./config/mim_config.json", types=args.model)
    json_tools = parse_json(json_file="./config/mim_config.json", types='tools')
    json_demo = parse_json(json_file="./config/mim_config.json", types='demo')
    args = update_args(args, [json_models, json_tools, json_demo])

    #! check path
    if not os.path.isdir(args.result):
        raise ValueError(f"{args.result} is not a saving folder")
    
    #! Set seed and device
    get_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_index
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=> GPU {args.cuda_index} will be used...")

    #! Load dataset
    # an easy data augmentation also achieves a better performance
    img_size = int(args.encoder.split('_')[-1])
    print(f"=> Load image from {args.target}...")
    dataset = LoadImage(args.target, img_size)
    
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
    if args.resume is not "none":
        print(f"=> Load checkpoint from {args.resume}")
        ckpt = torch.load(args.resume)["model"]
    model.load_state_dict(ckpt)
    print("=> all keys match!")

    #! Inference stage
    for image, path in dataset:
        _, recons_img, _ = model(image.to(DEVICE))
        image = Ten2Img(recons_img.squeeze())
        image.save(os.path.join(args.result, path.split('/')[-1]))
    print("=> Finished saving!")