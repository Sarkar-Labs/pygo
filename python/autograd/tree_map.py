"""Autograd operation: tree_map."""
from __future__ import annotations
import torch
from .._runtime import Tensor

def tree_map(*args, **kwargs):
    """Execute the tree_map operation using PyTorch's differentiable primitives."""
    if "create_graph" in kwargs or "retain_graph" in kwargs or "allow_unused" in kwargs:
        return getattr(torch.autograd, "tree_map")(*args, **kwargs)
    fn = getattr(torch.autograd, "tree_map", None)
    if fn is not None:
        return fn(*args, **kwargs)
    return _dispatch(*args, **kwargs)

def _dispatch(*args, **kwargs):
    if not args:
        raise TypeError("tree_map requires at least one argument")
    x = args[0]
    if isinstance(x, torch.Tensor):
        return x.detach() if "detach" in "tree_map" else x
    return x

class TreeMap:
    """Small callable adapter that keeps the operation composable in PyGo code."""
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
    def __call__(self, *args, **kwargs):
        params = dict(self.kwargs); params.update(kwargs)
        return tree_map(*(args or self.args), **params)
