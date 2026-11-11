"""Differentiable relu activation."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class Relu(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__(); self.dim = dim
    def forward(self, x):
        return F.relu(x)
    def __repr__(self):
        return f"Relu(dim={self.dim})"

def relu(x, dim=-1):
    return Relu(dim)(x)
