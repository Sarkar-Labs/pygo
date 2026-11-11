"""Differentiable loss: bce."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class Bce(nn.Module):
    def __init__(self, reduction: str = "mean", **kwargs):
        super().__init__(); self.reduction = reduction; self.kwargs = kwargs
    def forward(self, input, target, *args, **kwargs):
        opts = dict(self.kwargs); opts.update(kwargs); opts.setdefault("reduction", self.reduction)
        return F.binary_cross_entropy(input, target, *args, **opts)

def bce(input, target, reduction="mean", **kwargs):
    return F.binary_cross_entropy(input, target, reduction=reduction, **kwargs)
