"""Differentiable mish activation."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__(); self.dim = dim
    def forward(self, x):
        return F.mish(x)
    def __repr__(self):
        return f"Mish(dim={self.dim})"

def mish(x, dim=-1):
    return Mish(dim)(x)
