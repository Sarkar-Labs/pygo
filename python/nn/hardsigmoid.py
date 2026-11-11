"""Differentiable hardsigmoid activation."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class Hardsigmoid(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__(); self.dim = dim
    def forward(self, x):
        return F.hardsigmoid(x)
    def __repr__(self):
        return f"Hardsigmoid(dim={self.dim})"

def hardsigmoid(x, dim=-1):
    return Hardsigmoid(dim)(x)
