# Copyright (c) Xi Weng.
# Licensed under the Apache License, Version 2.0 (the "License");

import os
import time
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from src.transform import MuiltiCropDataset
from args import get_args
from methods import get_method

from src.utils import (
    setup_logging,
    cosine_scheduler,
    build_optimizer,
    restart_from_checkpoint,
    init_distributed_device,
    AverageMeter,
)

import wandb

def random_seed(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model

def main():
    args = get_args()
    random_seed(args)

    # fully initialize distributed device environment
    device, args = init_distributed_device(args)

    print(f"Rank {args.rank} running on device {args.device}")

    if args.rank == 0: wandb.init(project=args.wandb_project, name=args.env_name, config=args)

    if not os.path.exists(args.dump_path):
        # Create the folder if it doesn't exist
        os.makedirs(args.dump_path)
    setup_logging(os.path.join(args.dump_path,'out.log'), logging.INFO)

    # build data
    traindir = os.path.join(args.data_path, 'train')
    train_dataset = MuiltiCropDataset(
        traindir,
        args,
        return_index=False,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    logging.info(f"Building data done with {len(train_dataset)} images loaded.")

    # build model
    model = get_method(args)
    
    # synchronize batch norm layers
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # copy model to GPU
    torch.cuda.set_device(device)
    model.cuda(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device])

    if args.local_rank == 0:
        logging.info(model)
        logging.info("Building model done.")

    # build optimizer
    args.lr = args.lr * args.batch_size * args.world_size / 256

    optimizer = build_optimizer(model.parameters(), args)

    # ============ init schedulers ... ============
    args.lr_schedule = cosine_scheduler(
        args.lr,
        args.lr * 0.001,
        args.epochs, len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )

    # momentum parameter is increased to 1. during training with a cosine schedule
    args.momentum_schedule = cosine_scheduler(
            args.momentum, 1,
            args.epochs, len(train_loader)
    )

    logging.info(f"Building {args.optimizer} optimizer done.")

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
    )
    start_epoch = to_restore["epoch"]

    cudnn.benchmark = True

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logging.info(f"============ Starting epoch {epoch} ... ============")

        # set sampler
        train_loader.sampler.set_epoch(epoch)

        # train the network
        loss = train(train_loader, model, scaler, optimizer, epoch, args)

        # save checkpoints
        if args.local_rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(
                save_dict,
                os.path.join(args.dump_path, "checkpoint.pth.tar"),
            )
            if args.rank == 0:
                wandb.log({"learning_rate": optimizer.param_groups[0]['lr'], 
                        "momentum": unwrap_model(model).momentum, 
                        "loss": loss, "ep": epoch})
        
def train(loader, model, scaler, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.train()

    end = time.time()
    for it, samples in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # update parameters
        iters = len(loader) * epoch + it  # global training iteration
        adjust_parameters(model, optimizer, args, iters)

        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss = model(samples)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # ============ misc ... ============
        losses.update(loss.item(), samples[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if args.local_rank ==0 and it % 50 == 0:
            logging.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )
    return losses.avg

def adjust_parameters(model, optimizer, args, iters):
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr_schedule[iters]

    unwrap_model(model).momentum = args.momentum_schedule[iters]

if __name__ == "__main__":
    main()