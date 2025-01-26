# Copyright (c) Winci.
# Licensed under the Apache License, Version 2.0 (the "License");

import os
import shutil
import time
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import src.resnet as resnet
import src.vision_transformer as vits
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from args import get_args

from src.utils import (
    setup_logging,
    load_pretrained_encoder,
    init_distributed_device,
    restart_from_checkpoint,
    accuracy,
    AverageMeter,
)

best_acc1 = 0

def random_seed(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

def main():
    global best_acc1
    args = get_args()
    random_seed(args)
    # fully initialize distributed device environment
    device, args = init_distributed_device(args)

    if args.train_percent in {1, 10}:
        args.train_files = open('./src/imagenet_subset/{}percent.txt'.format(args.train_percent), 'r').readlines()

    if not os.path.exists(args.dump_path):
        # Create the folder if it doesn't exist
        os.makedirs(args.dump_path)

    setup_logging(os.path.join(args.dump_path,'out.log'), logging.INFO)
    if args.local_rank != 0:
        def log_pass(*args): pass
        logging.info = log_pass

    # Data loading code
    traindir = os.path.join(args.data_path, 'train')
    valdir = os.path.join(args.data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        valdir, 
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.train_percent in {1, 10}:
        train_dataset.samples = []
        for fname in args.train_files:
            fname = fname.strip()
            cls = fname.split('_')[0]
            pth = os.path.join(traindir, cls, fname)
            train_dataset.samples.append(
                (pth, train_dataset.class_to_idx[cls]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.workers, 
        pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, 
        num_workers=args.workers, pin_memory=True)
    logging.info(f"Building data done with {len(train_dataset)} images loaded.")

    # build model
    logging.info(f"creating model '{args.arch}'")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch.startswith('vit'):
        encoder, out_size = vits.__dict__[args.arch](patch_size=args.patch_size)
        out_size = out_size * (args.n_last_blocks + int(args.avgpool_patchtokens))
    else:
        encoder, out_size = resnet.__dict__[args.arch]()
    
    load_pretrained_encoder(encoder, args.pretrained)

    fc = nn.Linear(out_size, args.num_classes)

    # init the fc layer
    fc.weight.data.normal_(mean=0.0, std=0.01)
    fc.bias.data.zero_()

    model = model_forward(encoder, fc, args)

    if args.weights == 'freeze':
        encoder.requires_grad_(False)
        fc.requires_grad_(True)

    # build optimizer
    args.lr_classifier = args.lr_classifier * args.batch_size * args.world_size / 256
    logging.info(f"base classifier learning rate: {args.lr_classifier}")
    args.lr_encoder = args.lr_encoder * args.batch_size * args.world_size / 256
    logging.info(f"base encoder learning rate: {args.lr_encoder}")

    param_groups = [dict(params=fc.parameters(), lr=args.lr_classifier)]
    if args.weights == 'finetune':
        param_groups.append(dict(params=encoder.parameters(), lr=args.lr_encoder))

    optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=0)

    if args.scheduler == 'step':
        scheduler = MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)
    else:
        scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    logging.info(f"Building optimizer and scheduler done.")
    
    # copy model to GPU
    torch.cuda.set_device(device)
    model.cuda(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device])

    if args.weights == 'freeze':
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert len(parameters) == 2  # fc.weight, fc.bias

    if args.local_rank == 0:
        logging.info(model)
        logging.info("Building model done.")

    criterion = nn.CrossEntropyLoss().cuda(device)

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]

    cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logging.info(f"============ Starting epoch {epoch} ... ============")

        # set sampler
        train_loader.sampler.set_epoch(epoch)

        # train the network
        train(train_loader, model, criterion, optimizer, epoch, args)
        scheduler.step()

        # save checkpoints
        if args.local_rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            torch.save(
                save_dict,
                os.path.join(args.dump_path, "checkpoint.pth.tar"),
            )
            
            acc1, acc5 = validate(val_loader, model, criterion, args)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if is_best:
                shutil.copyfile(
                    os.path.join(args.dump_path, "checkpoint.pth.tar"),
                    os.path.join(args.dump_path, "best.pth.tar"),
                )

class model_forward(nn.Module):
    def __init__(self, encoder, fc, args):
        super().__init__()
        self.encoder = encoder
        self.fc = fc
        self.args = args
    
    def forward(self, samples):

        if "vit" in self.args.arch:
            intermediate_output = self.encoder.get_intermediate_layers(samples, self.args.n_last_blocks)
            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
            if self.args.avgpool_patchtokens:
                output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                output = output.reshape(output.shape[0], -1)
        else:
            output = self.encoder(samples)

        return self.fc(output)

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.weights == 'finetune':
        model.train()
    elif args.weights == 'freeze':
        model.eval()
    else:
        assert False, "Invalid weight option. Use 'finetune' or 'freeze'."

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.device is not None:
            images = images.cuda(args.device, non_blocking=True)
            target = target.cuda(args.device, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            logging.info(
                f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                f"Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                f"Acc@5 {top5.val:.3f} ({top5.avg:.3f})"
            )

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.device is not None:
                images = images.cuda(args.device, non_blocking=True)
                target = target.cuda(args.device, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                logging.info(
                    f"Test: [{i}/{len(val_loader)}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    f"Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                    f"Acc@5 {top5.val:.3f} ({top5.avg:.3f})"
                )

        logging.info(f" * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}")

    return top1.avg, top5.avg 

if __name__ == "__main__":
    main()

