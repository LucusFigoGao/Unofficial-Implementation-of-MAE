# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   vit.py
    Time:        2022/09/29 20:42:51
    Editor:      Figo
-----------------------------------
'''
from vit_pytorch import ViT
from utils.utils_fun import parse_json

vit_root = "/data/yifei/ViT/config/vision_transformer.json"


def vit_base_patch16_224():
    kwargs = parse_json(vit_root, types="vit_base_patch16_224")
    return ViT(**kwargs)

def vit_base_patch16_384():
    kwargs = parse_json(vit_root, types="vit_base_patch16_384")
    return ViT(**kwargs)

def vit_base_patch32_224():
    kwargs = parse_json(vit_root, types="vit_base_patch32_224")
    return ViT(**kwargs)

def vit_base_patch32_384():
    kwargs = parse_json(vit_root, types="vit_base_patch32_384")
    return ViT(**kwargs)

def vit_large_patch16_224():
    kwargs = parse_json(vit_root, types="vit_large_patch16_224")
    return ViT(**kwargs)

def vit_large_patch16_384():
    kwargs = parse_json(vit_root, types="vit_large_patch16_384")
    return ViT(**kwargs)

def vit_large_patch32_224():
    kwargs = parse_json(vit_root, types="vit_large_patch32_224")
    return ViT(**kwargs)

def vit_large_patch32_384():
    kwargs = parse_json(vit_root, types="vit_large_patch32_384")
    return ViT(**kwargs)



VIT_MODEL = {
    "vit_base_patch16_224": vit_base_patch16_224(), 
    "vit_base_patch16_384": vit_base_patch16_384(), 
    "vit_base_patch32_224": vit_base_patch32_224(), 
    "vit_base_patch32_384": vit_base_patch32_384(), 
    "vit_large_patch16_224": vit_large_patch16_224(), 
    "vit_large_patch16_384": vit_large_patch16_384(), 
    "vit_large_patch32_224": vit_large_patch32_224(), 
    "vit_large_patch32_384": vit_large_patch32_384()
}