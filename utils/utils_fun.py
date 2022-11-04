# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   utils_fun.py
    Time:        2022/09/27 13:59:17
    Editor:      Figo
-----------------------------------
'''
import os
import json
import torch
import numpy as np
import torch.nn as nn
import torchvision as tv
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts


class AverageMeter(object):
    """
        Computes and stores the average and current value, an example is:
        >>> losses = AverageMeter()
        >>> losses.update(loss.item(), inp.size(0))
        >>> return losses.avg
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,), exact=False):
    """
        Computes the top-k accuracy for the specified values of k

        Args:
            output (ch.tensor) : model output (N, classes) or (N, attributes) 
                for sigmoid/multitask binary classification
            target (ch.tensor) : correct labels (N,) [multiclass] or (N,
                attributes) [multitask binary]
            topk (tuple) : for each item "k" in this tuple, this method
                will return the top-k accuracy
            exact (bool) : whether to return aggregate statistics (if
                False) or per-example correctness (if True)

        Returns:
            A list of top-k accuracies.
    """
    with torch.no_grad():
        # Binary Classification
        if len(target.shape) > 1:
            assert output.shape == target.shape, "Detected binary classification but output shape != target shape"
            return [torch.round(torch.sigmoid(output)).eq(torch.round(target)).float().mean()], [-1.0] 

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        res_exact = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float()
            ck_sum = correct_k.sum(0, keepdim=True)
            res.append(ck_sum.mul_(100.0 / batch_size))
            res_exact.append(correct_k)

        if not exact: return res
        else: return res_exact


def get_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_json(json_file, types):
    with open(json_file, 'r') as file:
        json_dicts = json.load(file)
    return json_dicts[types]


def parse_train_tools(args, model):
    """
        parse criterion, optimizer, scheduler from args
    """
    if args.criterion == "CE":
        criterion = nn.CrossEntropyLoss()
    elif args.criterion == "BCE":
        criterion = nn.BCELoss()
    elif args.criterion == "MSE":
        criterion = nn.MSELoss()
    
    if args.optimizer == "SGD":
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)
    elif args.optimizer == "Adam":
        optimizer = Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "AdamW":
        optimizer = AdamW(model.parameters(), lr=args.lr)
    
    if args.scheduler == "stepLR":
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    if args.scheduler == "MultiStepLR":
        scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    elif args.scheduler == "Cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=args.eta_min)
    elif args.scheduler == "cosineAnnWarm":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.t_0, T_mult=args.t_mult, eta_min=args.eta_min)
    print("=> Criterion and optimizer and scheduler is set done...")
    return criterion, optimizer, scheduler


Ten2Img, Img2Ten = tv.transforms.ToPILImage(), tv.transforms.ToTensor()


def update_args(args, params):
    for _, p in enumerate(params):
        vars(args).update(p)
    print(args)
    return args
