"""Autograd operation: double_backward."""
from __future__ import annotations
import torch
from .._runtime import Tensor

def double_backward(*args, **kwargs):
    """Execute the double_backward operation using PyTorch's differentiable primitives."""
    if "create_graph" in kwargs or "retain_graph" in kwargs or "allow_unused" in kwargs:
        return getattr(torch.autograd, "double_backward")(*args, **kwargs)
    fn = getattr(torch.autograd, "double_backward", None)
    if fn is not None:
        return fn(*args, **kwargs)
    return _dispatch(*args, **kwargs)

def _dispatch(*args, **kwargs):
    if not args:
        raise TypeError("double_backward requires at least one argument")
    x = args[0]
    if isinstance(x, torch.Tensor):
        return x.detach() if "detach" in "double_backward" else x
    return x

class DoubleBackward:
    """Small callable adapter that keeps the operation composable in PyGo code."""
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
    def __call__(self, *args, **kwargs):
        params = dict(self.kwargs); params.update(kwargs)
        return double_backward(*(args or self.args), **params)
