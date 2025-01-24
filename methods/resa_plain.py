import torch
from methods.base import BaseMethod
import torch.nn.functional as F

class ReSA_plain(BaseMethod):
    # ReSA without using the momentum network

    def __init__(self, args):
        super().__init__(args)
        self.temp = args.temperature

    def forward(self, samples):

        samples = [x.cuda(non_blocking=True) for x in samples]
        h, emb = self.ForwardWrapper(samples, self.encoder, self.projector)

        with torch.no_grad():
            assign = self.sinkhorn(h @ h.T)
        
        total_loss = 0
        n_loss_terms = 0
        n_samples = len(emb)

        for q in range(n_samples - 1):
            for v in range(q+1, n_samples):
                emb_sim = emb[q] @ emb_m[k].T / self.temp
                total_loss += self.cross_entropy(emb_sim, assign)
                n_loss_terms += 1
        return total_loss / n_loss_terms
