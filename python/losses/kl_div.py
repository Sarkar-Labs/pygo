"""Differentiable loss: kl_div."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class KlDiv(nn.Module):
    def __init__(self, reduction: str = "mean", **kwargs):
        super().__init__(); self.reduction = reduction; self.kwargs = kwargs
    def forward(self, input, target, *args, **kwargs):
        opts = dict(self.kwargs); opts.update(kwargs); opts.setdefault("reduction", self.reduction)
        return F.kl_div(input, target, *args, **opts)

def kl_div(input, target, reduction="mean", **kwargs):
    return F.kl_div(input, target, reduction=reduction, **kwargs)
