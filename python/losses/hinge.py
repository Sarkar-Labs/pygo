"""Differentiable loss: hinge."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class Hinge(nn.Module):
    def __init__(self, reduction: str = "mean", **kwargs):
        super().__init__(); self.reduction = reduction; self.kwargs = kwargs
    def forward(self, input, target, *args, **kwargs):
        opts = dict(self.kwargs); opts.update(kwargs); opts.setdefault("reduction", self.reduction)
        return F.hinge_embedding_loss(input, target, *args, **opts)

def hinge(input, target, reduction="mean", **kwargs):
    return F.hinge_embedding_loss(input, target, reduction=reduction, **kwargs)
