# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   utils_train.py
    Time:        2022/09/27 11:50:21
    Editor:      Figo
-----------------------------------
'''

import torch
from tqdm import tqdm
from torchvision.utils import make_grid
from utils.utils_fun import AverageMeter, accuracy


def normal_train(epoch, model, loader, criterion, optim, writer=None, device='cuda:0', is_tqdm=True):
    
    top1, losses = AverageMeter(), AverageMeter()
    iterators = tqdm(enumerate(loader), total=len(loader)) if is_tqdm else enumerate(loader)
    
    for idx, (image, label) in iterators:
        inputs, targets, top1_acc = image.to(device), label.to(device), float('nan')
        
        # Calculate outputs, accuracy, loss
        outputs = model(inputs)
        prec1 = accuracy(outputs.cpu().data, label, topk=(1,))[0]
        loss = criterion(outputs, targets)
        
        # Record inportant values
        loss_val = loss.cpu().item()
        losses.update(loss_val, inputs.size(0))
        top1.update(prec1, inputs.size(0))
        top1_acc = top1.avg.item()

        if is_tqdm:
            iterators.desc = "(Train) Epoch [{}-{}] || Loss:{:.3f} || Acc:{:.2f}%".format(
                idx, epoch, loss_val, top1_acc
            )
        
        #! Update model parameters
        optim.zero_grad()
        loss.backward()
        optim.step()

        if writer is not None:
            descs, vals = ['loss', 'top1'], [losses, top1]
            for d, v in zip(descs, vals):
                writer.add_scalar('_'.join(['normal', 'train', d]), v.avg, epoch)
    
    return top1.avg, losses.avg


def normal_test(epoch, model, loader, criterion, writer=None, device='cuda:0', is_tqdm=True):
    
    top1, losses = AverageMeter(), AverageMeter()
    iterators = tqdm(enumerate(loader), total=len(loader)) if is_tqdm else enumerate(loader)
    
    with torch.no_grad():
        for idx, (image, label) in iterators:
            inputs, targets, top1_acc = image.to(device), label.to(device), float('nan')
            
            # Calculate outputs, accuracy, loss
            outputs = model(inputs)
            prec1 = accuracy(outputs.cpu().data, label, topk=(1,))[0]
            loss = criterion(outputs, targets)
            
            # Record inportant values
            loss_val = loss.cpu().item()
            losses.update(loss_val, inputs.size(0))
            top1.update(prec1, inputs.size(0))
            top1_acc = top1.avg.item()

            if is_tqdm:
                iterators.desc = "(Test) Epoch [{}-{}] || Loss:{:.3f} || Acc:{:.2f}%".format(
                    idx, epoch, loss_val, top1_acc
                )

            if writer is not None:
                descs, vals = ['loss', 'top1'], [losses, top1]
                for d, v in zip(descs, vals):
                    writer.add_scalar('_'.join(['normal', 'test', d]), v.avg, epoch)
        
        return top1.avg, losses.avg


def mim_train(epoch, model, loader, optim, writer=None, device='cuda:0', is_tqdm=True):
    
    losses = AverageMeter()
    iterators = tqdm(enumerate(loader), total=len(loader)) if is_tqdm else enumerate(loader)

    for idx, (image, _) in iterators:
        recon_loss, recons_img, patches2img = model(image.to(device))
        recon_loss = recon_loss.mean()
        
        #! Update model parameters
        with torch.autograd.set_detect_anomaly(True):
            optim.zero_grad()
            recon_loss.backward()
            optim.step()

        loss_val = recon_loss.cpu().item()
        losses.update(loss_val, image.size(0))

        if is_tqdm:
            iterators.desc = "(Train) Epoch [{}-{}] || Loss:{:.3f}".format(
                idx, epoch, recon_loss.item()
            )
        if writer is not None:
            writer.add_scalar('_'.join(['MIM', 'train', 'loss']), losses.avg, epoch)
            grid_x = make_grid(tensor=recons_img[:8].detach(), nrow=2, padding=2, normalize=True)
            writer.add_image("Epoch-{} Train recon image".format(epoch), grid_x)
            grid_y = make_grid(tensor=patches2img[:8].detach(), nrow=2, padding=2, normalize=True)
            writer.add_image("Epoch-{} Original mask image".format(epoch), grid_y)
    
    return losses.avg


def mim_test(epoch, model, loader, writer=None, device='cuda:0', is_tqdm=True):
    losses = AverageMeter()
    iterators = tqdm(enumerate(loader), total=len(loader)) if is_tqdm else enumerate(loader)

    for idx, (image, _) in iterators:
        recon_loss, recons_img, patches2img = model(image.to(device))
        loss_val = recon_loss.mean().cpu().item()
        losses.update(loss_val, image.size(0))

        if is_tqdm:
            iterators.desc = "(Test) Epoch [{}-{}] || Loss:{:.3f}".format(
                idx, epoch, loss_val
            )
        if writer is not None:
            writer.add_scalar('_'.join(['MIM', 'test', 'loss']), losses.avg, epoch)
            grid_x = make_grid(tensor=recons_img[:8].detach(), nrow=2, padding=2, normalize=True)
            writer.add_image("Epoch-{} Test recon image".format(epoch), grid_x)
            grid_y = make_grid(tensor=patches2img[:8].detach(), nrow=2, padding=2, normalize=True)
            writer.add_image("Epoch-{} Original mask image".format(epoch), grid_y)
    
    return losses.avg
