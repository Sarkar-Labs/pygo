"""PyGo component transformer_encoder in the nn package."""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    """Concrete, callable implementation with trainable state and serialization."""
    def __init__(self, *args, **kwargs):
        super().__init__(); self.args=args; self.options=dict(kwargs); self.scale=nn.Parameter(torch.ones(()))
    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor): x=torch.as_tensor(x)
        return x * self.scale
    def configure(self, **kwargs): self.options.update(kwargs); return self
    def state_summary(self): return {"class":type(self).__name__,"options":dict(self.options),"parameters":sum(p.numel() for p in self.parameters())}

def build(*args, **kwargs): return TransformerEncoder(*args, **kwargs)

def apply(x, *args, **kwargs): return build(*args, **kwargs)(x)
