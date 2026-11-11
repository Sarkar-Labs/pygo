"""Differentiable elu activation."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class Elu(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__(); self.dim = dim
    def forward(self, x):
        return F.elu(x)
    def __repr__(self):
        return f"Elu(dim={self.dim})"

def elu(x, dim=-1):
    return Elu(dim)(x)
