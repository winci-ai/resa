# Copyright (c) Winci.
# Licensed under the Apache License, Version 2.0 (the "License");

import torch.nn as nn
import src.resnet as resnet
import src.vision_transformer as vits

def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    
def get_projector(out_size, args):
    """ creates projector g() from config """
    x = []
    in_size = out_size
    for _ in range(args.mlp_layers - 1):
        x.append(nn.Linear(in_size, args.mlp_dim))
        if args.is_vit:
            x.append(nn.GELU())
        else:
            x.append(nn.BatchNorm1d(args.mlp_dim))
            x.append(nn.ReLU(inplace=True))
        in_size = args.mlp_dim
    x.append(nn.Linear(in_size, args.emb))

    if not args.is_vit:
        x.append(nn.BatchNorm1d(args.emb, affine=False))

    projector = nn.Sequential(*x)

    if args.is_vit:
        projector.apply(_init_weights)

    return projector

def get_predictor(args):
    """ creates predictor from config """
    if not args.pred:
        return None
    pred_dim = args.mlp_dim
    x = []
    if args.is_vit: x.append(nn.LayerNorm(args.emb, eps=1e-6))
    x.append(nn.Linear(args.emb, pred_dim))
    if args.is_vit:
        x.append(nn.GELU())
    else:
        x.append(nn.BatchNorm1d(args.mlp_dim))
        x.append(nn.ReLU(inplace=True))
    x.append(nn.Linear(pred_dim, args.emb))
    predictor = nn.Sequential(*x)

    if args.is_vit:
        predictor.apply(_init_weights)
    return predictor

def get_encoder(args):
    """ creates encoder E() by name and modifies it for dataset """
    if args.is_vit:
        encoder, out_size = vits.__dict__[args.arch](patch_size=args.patch_size, drop_path_rate=args.drop_path)
    else:
        encoder, out_size = resnet.__dict__[args.arch](
                zero_init_residual=(args.arch != 'resnet18'))
                
    return encoder, out_size
