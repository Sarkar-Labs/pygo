"""Optimizer implementation for adam."""
from __future__ import annotations
import torch
from torch.optim import Adam

class Adam(Adam):
    """PyTorch Adam with the standard optimizer protocol."""
    def __init__(self, params, lr=1e-3, **kwargs):
        super().__init__(params, lr=lr, **kwargs)
    def step_and_zero(self, closure=None):
        loss = self.step(closure)
        self.zero_grad(set_to_none=True)
        return loss

Optimizer = Adam
