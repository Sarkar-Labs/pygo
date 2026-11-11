"""Autograd operation: checkpoint."""
from __future__ import annotations
import torch
from .._runtime import Tensor

def checkpoint(*args, **kwargs):
    """Execute the checkpoint operation using PyTorch's differentiable primitives."""
    if "create_graph" in kwargs or "retain_graph" in kwargs or "allow_unused" in kwargs:
        return getattr(torch.autograd, "checkpoint")(*args, **kwargs)
    fn = getattr(torch.autograd, "checkpoint", None)
    if fn is not None:
        return fn(*args, **kwargs)
    return _dispatch(*args, **kwargs)

def _dispatch(*args, **kwargs):
    if not args:
        raise TypeError("checkpoint requires at least one argument")
    x = args[0]
    if isinstance(x, torch.Tensor):
        return x.detach() if "detach" in "checkpoint" else x
    return x

class Checkpoint:
    """Small callable adapter that keeps the operation composable in PyGo code."""
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
    def __call__(self, *args, **kwargs):
        params = dict(self.kwargs); params.update(kwargs)
        return checkpoint(*(args or self.args), **params)
