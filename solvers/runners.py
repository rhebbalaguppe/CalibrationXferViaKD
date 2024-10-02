import torch
from tqdm import tqdm
from utils import AverageMeter, accuracy, get_lr
import numpy as np
from calibration_library.metrics import ECELoss, SCELoss

from accelerate import Accelerator

def train(trainloader, model, optimizer, criterion, scheduler, accelerator:Accelerator, args):
    # switch to train mode
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()

    bar = tqdm(enumerate(trainloader), total=len(trainloader), disable=(not accelerator.is_main_process))
    for batch_idx, (inputs, targets) in bar:
        # inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()

        if args.scheduler == "warmupcosine":
            scheduler.step()

        # measure accuracy and record loss for rank-0 only
        if accelerator.is_main_process:
            prec1, = accuracy(outputs.data, targets.data, topk=(1, ))
            top1.update(prec1.item(), inputs.size(0))
            losses.update(loss.item(), inputs.size(0))

        # plot progress
        bar.set_postfix_str('({batch}/{size}) Loss: {loss:.8f} | top1: {top1: .4f} | lr {lr: .5f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    loss=losses.avg,
                    top1=top1.avg,
                    lr=get_lr(optimizer)
                    ))

    return (losses.avg, top1.avg)

def train_student(trainloader, student, teacher, optimizer, criterion, scheduler, accelerator:Accelerator, args):
    # switch to train mode on student
    # eval mode on teacher
    student.train()
    teacher.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    bar = tqdm(enumerate(trainloader), total=len(trainloader), disable=(not accelerator.is_main_process))
    for batch_idx, (inputs, targets) in bar:
        
        # inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = student(inputs)

        if accelerator.autocast():
            with torch.no_grad():
                outputs_teacher = teacher(inputs)
            loss = criterion(outputs, outputs_teacher, targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()

        if args.scheduler == "warmupcosine":
            scheduler.step()

        # measure accuracy and record loss
        if accelerator.is_main_process:
            prec1, = accuracy(outputs.data, targets.data, topk=(1, ))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

        # plot progress
        bar.set_postfix_str('({batch}/{size}) Loss: {loss:.8f} | top1: {top1: .4f} | lr {lr: .5f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    loss=losses.avg,
                    top1=top1.avg,
                    lr=get_lr(optimizer)
                    ))

    return (losses.avg, top1.avg)

@torch.no_grad()
def test(testloader, model, criterion, accelerator:Accelerator):

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    top5 = AverageMeter()

    all_targets = None
    all_outputs = None

    # switch to evaluate mode
    model.eval()

    bar = tqdm(enumerate(testloader), total=len(testloader), disable=(not accelerator.is_main_process))
    for batch_idx, (inputs, targets) in bar:

        # inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        prec1, prec3, prec5  = accuracy(outputs.data, targets.data, topk=(1, 3, 5))

        # gather metrics and outputs
        prec1, prec3, prec5, loss = accelerator.gather((prec1, prec3, prec5, loss))
        outputs, targets = accelerator.gather((outputs, targets))

        if accelerator.is_main_process:
            losses.update(loss.mean().item(), inputs.size(0))
            top1.update(prec1.mean().item(), inputs.size(0))
            top3.update(prec3.mean().item(), inputs.size(0))
            top5.update(prec5.mean().item(), inputs.size(0))

            targets = targets.cpu().numpy()
            outputs = outputs.cpu().numpy()

            if all_targets is None:
                all_outputs = outputs
                all_targets = targets
            else:
                all_targets = np.concatenate([all_targets, targets], axis=0)
                all_outputs = np.concatenate([all_outputs, outputs], axis=0)

        # plot progress
        bar.set_postfix_str('({batch}/{size}) Loss: {loss:.8f} | top1: {top1: .4f} | top3: {top3: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    loss=losses.avg,
                    top1=top1.avg,
                    top3=top3.avg,
                    top5=top5.avg,
                    ))

    if accelerator.is_main_process:
        ECE = ECELoss().loss(all_outputs, all_targets, n_bins=15)
        SCE = SCELoss().loss(all_outputs, all_targets, n_bins=15)
    else:
        ECE = None
        SCE = None

    return (losses.avg, top1.avg, top3.avg, top5.avg, SCE, ECE)

@torch.no_grad()
def get_logits_from_model_dataloader(testloader, model):
    """Returns torch tensor of logits and targets on cpu"""
    # switch to evaluate mode
    model.eval()

    all_targets = None
    all_outputs = None

    bar = tqdm(testloader, total=len(testloader), desc="Evaluating logits")
    for inputs, targets in bar:
        inputs = inputs.cuda()
        # compute output
        outputs = model(inputs)
        # to numpy
        targets = targets.cpu().numpy()
        outputs = outputs.cpu().numpy()

        if all_targets is None:
            all_outputs = outputs
            all_targets = targets
        else:
            all_targets = np.concatenate([all_targets, targets], axis=0)
            all_outputs = np.concatenate([all_outputs, outputs], axis=0)

    return torch.from_numpy(all_outputs), torch.from_numpy(all_targets)

    
