# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   utils_data.py
    Time:        2022/09/27 11:25:45
    Editor:      Figo
-----------------------------------
'''

import os
import glob
import torchvision as tv
from PIL import Image
from pathlib import Path
from robustness.datasets import ImageNet
from utils.default import DEFAULT_DATA_ROOT, TRAIN_TRANSFORMS_EASY, TEST_TRANSFORMS_EASY, TRAIN_TRANSFORMS_DEFAULT, \
                          TEST_TRANSFORMS_DEFAULT, TRAIN_TRANSFORMS_IMAGENET, TEST_TRANSFORMS_IMAGENET, IMG_FORMATS



def load_dataset(name, transform_type='easy', trainTransform=None, testTransform=None):
    data_root = DEFAULT_DATA_ROOT[name]
    print(f"=> Load {name} dataset from {data_root}")
    
    if transform_type == 'easy':
        trainTransform, testTransform = TRAIN_TRANSFORMS_EASY, TEST_TRANSFORMS_EASY
    elif transform_type == 'default':
        if name == "imagenet":
            trainTransform = tv.transforms.Compose([
                TRAIN_TRANSFORMS_IMAGENET, 
                tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            testTransform = tv.transforms.Compose([
                TEST_TRANSFORMS_IMAGENET, 
                tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        elif name in ["cifar", 'cifar10', 'cifar100']:
            trainTransform = tv.transforms.Compose([
                TRAIN_TRANSFORMS_DEFAULT(32), 
                tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            testTransform = tv.transforms.Compose([
                TEST_TRANSFORMS_DEFAULT(32), 
                tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    elif transform_type == "custom":
        print("=> Self define transform...")
        trainTransform, testTransform = trainTransform, testTransform
    
    if name == "cifar10":
        train_set = tv.datasets.CIFAR10(data_root, train=True, transform=trainTransform, download=False)
        test_set = tv.datasets.CIFAR10(data_root, train=False, transform=testTransform, download=False)
    
    elif name == "cifar100":
        train_set = tv.datasets.CIFAR100(data_root, train=True, transform=trainTransform, download=False)
        test_set = tv.datasets.CIFAR100(data_root, train=False, transform=testTransform, download=False)
    
    elif name == "imagenet":
        kwargs = {"transform_train": trainTransform, "transform_test": testTransform}
        dataset = ImageNet(data_root, **kwargs)
        return dataset
    
    return train_set, test_set


class LoadImage:
    def __init__(self, path, img_size) -> None:
        p = str(Path(path).resolve())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        self.images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        self.transform = tv.transforms.Compose([
            tv.transforms.Resize((img_size, img_size)), 
            tv.transforms.ToTensor(), 
        ])
        self.nf = len(self.images)
    
    def __iter__(self):
        self.count = 0
        return self
    
    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.images[self.count]
        image = Image.open(path)
        image = self.transform(image).unsqueeze(dim=0)
        self.count += 1
        return image, path

    def __len__(self):
        return self.nf  # number of files
