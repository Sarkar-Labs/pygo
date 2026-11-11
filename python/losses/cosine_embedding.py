"""Differentiable loss: cosine_embedding."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineEmbedding(nn.Module):
    def __init__(self, reduction: str = "mean", **kwargs):
        super().__init__(); self.reduction = reduction; self.kwargs = kwargs
    def forward(self, input, target, *args, **kwargs):
        opts = dict(self.kwargs); opts.update(kwargs); opts.setdefault("reduction", self.reduction)
        return F.cosine_embedding_loss(input, target, *args, **opts)

def cosine_embedding(input, target, reduction="mean", **kwargs):
    return F.cosine_embedding_loss(input, target, reduction=reduction, **kwargs)
