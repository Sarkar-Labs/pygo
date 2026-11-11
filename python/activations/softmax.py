"""Differentiable softmax activation."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class Softmax(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__(); self.dim = dim
    def forward(self, x):
        return F.softmax(x, dim=self.dim)
    def __repr__(self):
        return f"Softmax(dim={self.dim})"

def softmax(x, dim=-1):
    return Softmax(dim)(x)
