"""Differentiable prelu activation."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class Prelu(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__(); self.dim = dim
    def forward(self, x):
        return F.prelu(x)
    def __repr__(self):
        return f"Prelu(dim={self.dim})"

def prelu(x, dim=-1):
    return Prelu(dim)(x)
