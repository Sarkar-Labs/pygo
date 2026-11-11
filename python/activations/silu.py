"""Differentiable silu activation."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class Silu(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__(); self.dim = dim
    def forward(self, x):
        return F.silu(x)
    def __repr__(self):
        return f"Silu(dim={self.dim})"

def silu(x, dim=-1):
    return Silu(dim)(x)
