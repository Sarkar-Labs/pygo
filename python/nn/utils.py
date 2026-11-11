"""Core neural-network helper: utils."""
from __future__ import annotations
import torch
import torch.nn as nn

class Utils(nn.Module):
    def __init__(self, module=None):
        super().__init__()
        self.module = module if module is not None else nn.Identity()
    def forward(self, x, *args, **kwargs):
        return self.module(x, *args, **kwargs) if callable(self.module) else x
    def state_summary(self):
        return {name: tuple(p.shape) for name,p in self.named_parameters()}

def build(*args, **kwargs):
    return Utils(*args, **kwargs)
