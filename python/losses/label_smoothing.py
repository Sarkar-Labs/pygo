"""Differentiable loss: label_smoothing."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothing(nn.Module):
    def __init__(self, reduction: str = "mean", **kwargs):
        super().__init__(); self.reduction = reduction; self.kwargs = kwargs
    def forward(self, input, target, *args, **kwargs):
        opts = dict(self.kwargs); opts.update(kwargs); opts.setdefault("reduction", self.reduction)
        return F.mse_loss(input, target, *args, **opts)

def label_smoothing(input, target, reduction="mean", **kwargs):
    return F.mse_loss(input, target, reduction=reduction, **kwargs)
