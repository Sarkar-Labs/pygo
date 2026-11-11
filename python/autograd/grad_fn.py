"""Autograd operation: grad_fn."""
from __future__ import annotations
import torch
from .._runtime import Tensor

def grad_fn(*args, **kwargs):
    """Execute the grad_fn operation using PyTorch's differentiable primitives."""
    if "create_graph" in kwargs or "retain_graph" in kwargs or "allow_unused" in kwargs:
        return getattr(torch.autograd, "grad_fn")(*args, **kwargs)
    fn = getattr(torch.autograd, "grad_fn", None)
    if fn is not None:
        return fn(*args, **kwargs)
    return _dispatch(*args, **kwargs)

def _dispatch(*args, **kwargs):
    if not args:
        raise TypeError("grad_fn requires at least one argument")
    x = args[0]
    if isinstance(x, torch.Tensor):
        return x.detach() if "detach" in "grad_fn" else x
    return x

class GradFn:
    """Small callable adapter that keeps the operation composable in PyGo code."""
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
    def __call__(self, *args, **kwargs):
        params = dict(self.kwargs); params.update(kwargs)
        return grad_fn(*(args or self.args), **params)
