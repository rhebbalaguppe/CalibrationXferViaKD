import logging
from .logger import *
from .eval import *
from .argparser import parse_args
from .misc import *
from .earlystopper import EarlyStopping
import torch
import os
import shutil

from accelerate import Accelerator

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth', comment = None):
    if(comment is not None):
        filename = comment + "_" + filename
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        best_filename = "model_best.pth"
        if(comment is not None):
            best_filename = comment + "_" + best_filename
        shutil.copyfile(filepath, os.path.join(checkpoint, best_filename))
        return os.path.join(checkpoint, best_filename)
    

def create_save_path(args, mode = "teacher"):

    if mode == "student":
        ans_str = f"_student_T={args.T}_Lambda={args.Lambda}"
    else:
        ans_str = f"_{args.loss}"

        if args.loss == "FLSD":
            ans_str += f"_gamma={args.gamma}"
            return ans_str
        
        if "focal_loss" in args.loss or "FL" in args.loss:
            ans_str += f"_gamma={args.gamma}"
        
        if "LS" in args.loss:
            ans_str += f"_alpha={args.alpha}"

        if "MDCA" in args.loss:
            ans_str += f"_beta={args.beta}"
            return ans_str

        if "DCA" in args.loss or "MMCE" in args.loss:
            ans_str += f"_beta={args.beta}"

        if "mdca" in args.loss or "FL+MDCA" in args.loss:
            ans_str += f"_beta={args.beta}_gamma={args.gamma}"

    if args.exp_name:
        ans_str += f"_{args.exp_name}"
        
    return ans_str
