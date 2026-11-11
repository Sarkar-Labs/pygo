"""Differentiable sparsemax activation."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class Sparsemax(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__(); self.dim = dim
    def forward(self, x):
        return F.sparsemax(x)
    def __repr__(self):
        return f"Sparsemax(dim={self.dim})"

def sparsemax(x, dim=-1):
    return Sparsemax(dim)(x)
