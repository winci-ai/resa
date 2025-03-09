# Copyright (c) Winci.
# Licensed under the Apache License, Version 2.0 (the "License");

import torch
import copy
from itertools import chain
from method.base import BaseMethod

class ReSA(BaseMethod):
    # ReSA using the momentum network, better performance

    def __init__(self, args):
        super().__init__(args)

        self.momentum_encoder = copy.deepcopy(self.encoder)
        self.momentum_projector = copy.deepcopy(self.projector)
        
        for param in chain(self.momentum_encoder.parameters(), 
                           self.momentum_projector.parameters()):
            param.requires_grad = False

        self.temp = args.temperature

    def forward(self, samples):

        samples = [x.cuda(non_blocking=True) for x in samples]

        h, emb = self.ForwardWrapper(samples, self.encoder, self.projector)
        
        with torch.no_grad():
            self.update_momentum_params(self.momentum)
            h_m, emb_m = self.ForwardWrapper(samples[:2], self.momentum_encoder, self.momentum_projector)
            assign = self.sinkhorn_knopp(h @ h_m.T)
           
        total_loss = 0
        n_loss_terms = 0

        for q in range(len(emb)):
            for v in range(len(emb_m)):
                if v == q:
                    continue
                emb_sim = emb[q] @ emb_m[v].T / self.temp
                total_loss += self.cross_entropy(emb_sim, assign)
                n_loss_terms += 1
        return total_loss / n_loss_terms



