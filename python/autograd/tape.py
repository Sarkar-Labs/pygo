"""Autograd operation: tape."""
from __future__ import annotations
import torch
from .._runtime import Tensor

def tape(*args, **kwargs):
    """Execute the tape operation using PyTorch's differentiable primitives."""
    if "create_graph" in kwargs or "retain_graph" in kwargs or "allow_unused" in kwargs:
        return getattr(torch.autograd, "tape")(*args, **kwargs)
    fn = getattr(torch.autograd, "tape", None)
    if fn is not None:
        return fn(*args, **kwargs)
    return _dispatch(*args, **kwargs)

def _dispatch(*args, **kwargs):
    if not args:
        raise TypeError("tape requires at least one argument")
    x = args[0]
    if isinstance(x, torch.Tensor):
        return x.detach() if "detach" in "tape" else x
    return x

class Tape:
    """Small callable adapter that keeps the operation composable in PyGo code."""
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
    def __call__(self, *args, **kwargs):
        params = dict(self.kwargs); params.update(kwargs)
        return tape(*(args or self.args), **params)
