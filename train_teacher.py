import os
import pandas as pd
import random

from sys import argv

import torch
import torch.optim as optim

import transformers

import numpy as np

from utils import mkdir_p, parse_args
from utils import get_lr, save_checkpoint, create_save_path

from solvers.runners import train, test
from solvers.loss import loss_dict

from models import get_model
from datasets import dataloader_dict

from datetime import datetime

import logging
from utils import Logger

import accelerate

if __name__ == "__main__":
    
    # input size does not change
    torch.backends.cudnn.benchmark = False # false means deterministic outputs, but less throughput

    # set up accelerator
    accelerator = accelerate.Accelerator()

    # parse arguments
    args = parse_args()

    # set seeds
    accelerate.utils.set_seed(args.seed)
    
    # prepare save path
    if args.current_time:
        current_time = args.current_time
    else:
        current_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
    model_save_pth = f"{args.checkpoint}/{args.dataset}/{args.model}/{current_time}{create_save_path(args, mode='teacher')}"
    checkpoint_dir_name = model_save_pth

    if accelerator.is_main_process:
        if not os.path.isdir(model_save_pth):
            mkdir_p(model_save_pth)

        logging.basicConfig(level=logging.INFO, 
                            format="%(levelname)s:  %(message)s",
                            handlers=[
                                logging.FileHandler(filename=os.path.join(model_save_pth, "train_log.txt")),
                                logging.StreamHandler()
                            ])
        logging.info(f"Setting up logging folder : {model_save_pth}")
        logging.info(args)
        logging.info(argv)
        logging.info(f"GPUs used: {torch.cuda.device_count()}")
        
    accelerator.wait_for_everyone()

    # prepare model
    logging.info(f"Using model : {args.model}")
    model = get_model(args.model, args.dataset, args)

    # set up dataset
    logging.info(f"Using dataset : {args.dataset}")
    trainloader, valloader, testloader = dataloader_dict[args.dataset](args)

    logging.info(f"Setting up optimizer : {args.optimizer}")
    optimizer = optim.SGD(model.parameters(), 
                            lr=args.lr, 
                            momentum=args.momentum, 
                            weight_decay=args.weight_decay,
                            nesterov=True)
    
    criterion = loss_dict[args.loss](loss=args.loss, beta=args.beta, gamma=args.gamma, alpha=args.alpha)
    test_criterion = torch.nn.CrossEntropyLoss()

    # set up logger
    if accelerator.is_main_process:
        logger = Logger(os.path.join(checkpoint_dir_name, "train_metrics.txt"))
        logger.set_names(["lr", "train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc", "SCE", "ECE"])

    start_epoch = args.start_epoch
    best_acc = 0.
    best_acc_stats = {"top1" : 0.0}

    trainloader, valloader, testloader, model, optimizer = accelerator.prepare(
        trainloader, valloader, testloader, model, optimizer
    )

    # defining lr_scheduler here since number of steps can change depending upon GPUs for each process
    if args.scheduler == "multistep":
        logging.info(f"Step sizes : {args.schedule_steps} | lr-decay-factor : {args.lr_decay_factor}")
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule_steps, gamma=args.lr_decay_factor)
    elif args.scheduler == "warmupcosine":
        total_iters = int(len(trainloader) * args.epochs)
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, args.warmup, total_iters)
    elif args.scheduler == "cosine":
        total_iters = int(args.epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, verbose=True)

    accelerator.print("Using the scheduler:", scheduler)

    for epoch in range(start_epoch, args.epochs):

        accelerator.print("Epoch: [%d | %d] LR: %f" % (epoch + 1, args.epochs, get_lr(optimizer)))
        
        train_loss, top1_train = train(trainloader, model, optimizer, criterion, scheduler, accelerator, args)
        val_loss, top1_val, _, _, sce_score_val, ece_score_val = test(valloader, model, test_criterion, accelerator)
        test_loss, top1, top3, top5, sce_score, ece_score = test(testloader, model, test_criterion, accelerator)

        if args.scheduler == "multistep" or args.scheduler == "cosine":
            scheduler.step()

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            print("End of epoch {} stats: train_loss : {:.4f} | val_loss : {:.4f} | top1_train : {:.4f} | top1 : {:.4f} | SCE : {:.5f} | ECE : {:.5f}".format(
                epoch+1,
                train_loss,
                test_loss,
                top1_train,
                top1,
                sce_score,
                ece_score
            ))

            # save best accuracy model
            is_best = top1_val > best_acc
            best_acc = max(best_acc, top1_val)

            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': accelerator.unwrap_model(model).state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict(),
                    'dataset' : args.dataset,
                    'model' : args.model
                }, is_best, checkpoint=model_save_pth)
        
            # Update best stats
            if is_best:
                best_acc_stats = {
                    "top1" : top1,
                    "top3" : top3,
                    "top5" : top5,
                    "SCE" : sce_score,
                    "ECE" : ece_score
                }
            logger.append([get_lr(optimizer), train_loss, top1_train, val_loss, top1_val, test_loss, top1, sce_score, ece_score])

    if accelerator.is_main_process:
        print("training completed...")
        print("The stats for best accuracy model on test set are as below:")
        print(best_acc_stats)
        logger.append(["best_accuracy", 0, 0, 0, 0, 0, best_acc_stats["top1"], best_acc_stats["SCE"], best_acc_stats["ECE"]])

        # log results to a common file
        df = {
            "dataset" : [args.dataset],
            "model" : [args.model],
            "folder_path" : [checkpoint_dir_name],
            "top1" : [best_acc_stats["top1"]],
            "ECE" : [best_acc_stats["ECE"]],
            "SCE" : [best_acc_stats["SCE"]],
            "checkpoint_train_loss" : [train_loss],
            "checkpoint_train_top1" : [top1_train],
            "checkpoint_val_loss" : [val_loss],
            "checkpoint_val_top1" : [top1_val],
            "checkpoint_test_loss" : [test_loss],
            "checkpoint_test_top1" : [top1],
            "checkpoint_test_top3" : [top3],
            "checkpoint_test_top5" : [top5],
            "checkpoint_test_sce" : [sce_score],
            "checkpoint_test_ece" : [ece_score]
        }

        df =  pd.DataFrame(df)
        result_folder = "results_csv"
        os.makedirs(result_folder, exist_ok=True)
        save_path_file = os.path.join(result_folder, "teacher_metrics.csv")
        df.to_csv(save_path_file, mode="a", index=False, header=(not os.path.isfile(save_path_file)))
