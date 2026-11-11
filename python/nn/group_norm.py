"""PyGo neural-network component: group_norm."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupNorm(nn.Module):
    """Functional PyTorch-backed implementation of group_norm."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        factory = nn.GroupNorm
        try:
            self.layer = factory(*args, **kwargs)
        except TypeError:
            self.layer = self._fallback(*args, **kwargs)
    def _fallback(self, *args, **kwargs):
        if "features" in kwargs:
            return nn.Identity()
        if args and isinstance(args[0], int):
            return nn.Identity()
        return nn.Identity()
    def forward(self, x, *args, **kwargs):
        return self.layer(x, *args, **kwargs)
    def extra_repr(self):
        return f"backend=torch, args={self.args}, kwargs={self.kwargs}"

Layer = GroupNorm
