"""Differentiable selu activation."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class Selu(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__(); self.dim = dim
    def forward(self, x):
        return F.selu(x)
    def __repr__(self):
        return f"Selu(dim={self.dim})"

def selu(x, dim=-1):
    return Selu(dim)(x)
