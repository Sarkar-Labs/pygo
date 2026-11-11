"""Differentiable gelu activation."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class Gelu(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__(); self.dim = dim
    def forward(self, x):
        return F.gelu(x)
    def __repr__(self):
        return f"Gelu(dim={self.dim})"

def gelu(x, dim=-1):
    return Gelu(dim)(x)
