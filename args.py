# Copyright (c) Winci.
# Licensed under the Apache License, Version 2.0 (the "License");

import argparse

def get_default_params(arch):
    if "vit" in arch:
        return {"optimizer": 'adamw', "lr": 5e-4, "wd": 0.1, "warmup_epochs": 40}
    else:
        return {"optimizer": 'sgd', "lr": 0.5, "wd": 1e-5, "warmup_epochs": 2}

def get_args():
    parser = argparse.ArgumentParser(description="Implementation of ReSA")

    parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")

    parser.add_argument('--seed', default=None, type=int,
                    help='random seed for initializing training.')

    #####################
    #### data params ####
    #####################
    parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                    help="path to dataset repository")

    parser.add_argument("--crops_nmb", type=int, default=[1], nargs="+",
                    help="list of number of crops (example: [1, 10])")
                    
    parser.add_argument("--crops_size", type=int, default=[224], nargs="+",
                    help="crops resolutions (example: [224, 96])")
    
    parser.add_argument("--crops_min_scale", type=float, default=[0.2], nargs="+",
                    help="minimum scale of the crops (example: [0.25, 0.05])")

    parser.add_argument("--crops_max_scale", type=float, default=[1.], nargs="+",
                    help="maximum scale of the crops (example: [1., 0.25])")

    parser.add_argument("--solarization_prob", type=float, default=[0.2], nargs="+",
                    help="solarization prob (example: [0.2, 0.0])")

    parser.add_argument("--size_dataset", type=int, default=-1, 
                    help="size of dataset, -1 indicates the full dataset")

    parser.add_argument("--workers", default=8, type=int,
                    help="number of data loading workers per gpu")
    
    ############################
    ### resa specific params ###
    ############################
    parser.add_argument("--temperature", default=0.4, type=float,
                    help="temperature parameter in training loss")

    parser.add_argument("--momentum", type=float, default=0.996, 
                    help="Base EMA parameter")

    #####################
    #### optim params ###
    #####################
    parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run")

    parser.add_argument("--batch_size", default=256, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")

    parser.add_argument('--lr', default=None, type=float, 
                    help='initial (base) learning rate for train')

    parser.add_argument('--wd', default=None, type=float, 
                    help='weight decay for train')
    
    parser.add_argument("--optimizer", type=str, choices=["sgd","adamw"], default=None, 
                    help="optimizer")

    parser.add_argument("--warmup_epochs", default=None, type=int, 
                    help="number of warmup epochs")

    ####################
    #### dist params ###
    ####################
    parser.add_argument("--world_size", default=1, type=int, 
                    help="""number of processes: it is set automatically and
                            should not be passed as argument""")

    parser.add_argument("--rank", default=0, type=int, 
                    help="rank of this process: it is set automatically and should not be passed as argument")

    parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
                    
    parser.add_argument("--no-set-device-rank", default=False, action="store_true",
                    help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).")

    parser.add_argument("--dist-url", default="env://", type=str,
                    help="url used to set up distributed training")

    ############################
    #### architecture params ###
    ############################
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    help='model architecture (e.g. resnet18, resnet50, vit_small, vit_base)')

    parser.add_argument('--patch_size', default=16, type=int, 
                    help='Patch resolution of the vision transformer.')

    parser.add_argument("--mlp_layers", type=int, default=3, 
                    help="number of FC layers in projector")

    parser.add_argument("--mlp_dim", type=int, default=2048, 
                    help="size of FC layers in projector/predictor")

    parser.add_argument("--no_pred", dest="pred", action="store_false", 
                    help="do not use an extra predictor")

    parser.add_argument("--emb", type=int, default=512, 
                    help="embedding dimension of the projector")

    parser.add_argument("--drop_path", type=float, default=0., 
                    help="Stochastic Depth")
    
    ########################
    #### evaluate params ###
    ########################
    parser.add_argument('--train_percent', default=100, type=int,
                    choices=(100, 10, 1),
                    help='size of traing set in percent')

    parser.add_argument('--num_classes', default=1000, type=int,   
                    help='number of classes')

    parser.add_argument('--weights', default='freeze', type=str,
                    choices=('finetune', 'freeze'),
                    help='finetune or freeze pretrained encoder weights')

    parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to checkpoint for evaluation(default: none)')

    parser.add_argument('--lr_encoder', default=0.0002, type=float, metavar='LR',
                    help='encoder base learning rate')

    parser.add_argument('--lr_classifier', default=40, type=float, metavar='LR',
                    help='classifier base learning rate')

    parser.add_argument("--scheduler", type=str, default="step", choices=('step', 'cos'),
                    help="learning rate scheduler")

    parser.add_argument('--n_last_blocks', default=4, type=int, 
                    help="""Concatenate [CLS] tokens for the `n` last blocks. 
                            We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")

    parser.add_argument('--avgpool_patchtokens', default=False, action="store_true",
                    help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
                            We typically set this to False for ViT-Small and to True with ViT-Base.""")

    parser.add_argument('--use_cuda', default=True,
                    help="""Should we store the features on GPU in knn evaluation? 
                            We recommend setting this to False if you encounter OOM""")
    
    args = parser.parse_args()

    default_params = get_default_params(args.arch)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args