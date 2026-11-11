"""Differentiable sigmoid activation."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class Sigmoid(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__(); self.dim = dim
    def forward(self, x):
        return F.sigmoid(x)
    def __repr__(self):
        return f"Sigmoid(dim={self.dim})"

def sigmoid(x, dim=-1):
    return Sigmoid(dim)(x)
