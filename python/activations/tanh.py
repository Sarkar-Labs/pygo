"""Differentiable tanh activation."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class Tanh(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__(); self.dim = dim
    def forward(self, x):
        return F.tanh(x)
    def __repr__(self):
        return f"Tanh(dim={self.dim})"

def tanh(x, dim=-1):
    return Tanh(dim)(x)
