"""Data transform: tokenize."""
from __future__ import annotations
import torch
import torch.nn.functional as F

class Tokenize:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
    def __call__(self, x):
        if not isinstance(x, torch.Tensor): x = torch.as_tensor(x)
        if "normalize" in "tokenize":
            mean = torch.as_tensor(self.kwargs.get("mean", 0.0), device=x.device, dtype=x.dtype)
            std = torch.as_tensor(self.kwargs.get("std", 1.0), device=x.device, dtype=x.dtype).clamp_min(1e-12)
            return (x-mean)/std
        if "resize" in "tokenize" and self.args:
            size = self.args[0] if isinstance(self.args[0], (tuple,list)) else (self.args[0], self.args[0])
            return F.interpolate(x.unsqueeze(0) if x.ndim==3 else x, size=size, mode="bilinear", align_corners=False).squeeze(0) if x.ndim==3 else F.interpolate(x,size=size,mode="bilinear",align_corners=False)
        return x.clone()
    def __repr__(self): return f"Tokenize(args={self.args}, kwargs={self.kwargs})"

def apply(x, *args, **kwargs): return Tokenize(*args, **kwargs)(x)
