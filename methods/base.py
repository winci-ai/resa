# Copyright (c) Xi Weng.
# Licensed under the Apache License, Version 2.0 (the "License");

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from src.model import get_encoder, get_projector, get_predictor

class BaseMethod(nn.Module):
    """
        Base class for self-supervised loss implementation.
        It includes encoder and projector for training function.
    """
    def __init__(self, args):
        super().__init__()
        self.is_vit = args.is_vit = ('vit' in args.arch)
        self.encoder, self.out_size = get_encoder(args)
        self.projector = get_projector(self.out_size, args)
        self.predictor = get_predictor(args)
        self.momentum = args.momentum
        self.device = args.device

    def ForwardWrapper(self, samples, encoder, projector):

        if self.is_vit:
            views = len(samples)
            idx_crops = torch.cumsum(torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in samples]),
                return_counts=True,
            )[1], 0)

            start_idx, rep = 0, torch.empty(0).to(self.device)

            for end_idx in idx_crops:
                _out = encoder(torch.cat(samples[start_idx: end_idx]))
                rep = torch.cat((rep, _out))
                start_idx = end_idx
            # Run the projector forward on the concatenated features.
            emb = projector(rep)
            if torch.is_grad_enabled() and self.predictor:
                emb = self.predictor(emb)
            
            emb = emb.chunk(views)
            rep = rep.chunk(views)
        else:
            # do not concate different views if BN is in the model 
            # As it will disrupt the zero-mean, unit-variance distribution
            rep = [encoder(x) for x in samples]
            emb = [projector(x) for x in rep]
            if torch.is_grad_enabled() and self.predictor:
                emb = [self.predictor(x) for x in emb]

        emb = [FullGather.apply(F.normalize(x)) for x in emb]

        with torch.no_grad():
            rep = FullGather.apply(F.normalize(rep[0]))

        return rep, emb

    @torch.no_grad()
    def sinkhorn(self, scores, eps=0.05, niters=3):
        Q = torch.exp(scores / eps).t()
        Q /= Q.sum()
        K, B = Q.shape
        u = torch.zeros(K, device=scores.device)
        r = torch.ones(K, device=scores.device) / K
        c = torch.ones(B, device=scores.device) / B
        for _ in range(niters):
            u = Q.sum(dim=1)
            Q *= (r / u).unsqueeze(1)
            Q *= (c / Q.sum(dim=0)).unsqueeze(0)
        return (Q / Q.sum(dim=0, keepdim=True)).t()

    def cross_entropy(self, emb_sim, assign):

        prob = F.softmax(emb_sim, dim=1)
        prob_t = F.softmax(emb_sim.T, dim=1)
        
        loss = torch.sum(-assign * torch.log(prob), dim=-1).mean() + \
               torch.sum(-assign.T * torch.log(prob_t), dim=-1).mean()

        return loss / 2

    @torch.no_grad()
    def _momentum_params_update(self, m):
        """
        Update of the momentum encoder and projector
        """
        for param_q, param_k in zip(self.encoder.parameters(), 
                                    self.momentum_encoder.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)
        for param_q, param_k in zip(self.projector.parameters(),
                                    self.momentum_projector.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)
        
    def forward(self, samples):
        raise NotImplementedError

class FullGather(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """
    @staticmethod
    def forward(ctx, input):
        ctx.batch_size = input.shape[0]
        gather_list = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(gather_list, input)
        return torch.cat(gather_list, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        batch_size = ctx.batch_size
        rank = dist.get_rank()
        grad_input = grad_output[rank * batch_size : (rank + 1) * batch_size]
        return grad_input
