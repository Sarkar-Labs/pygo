"""Gradient checkpointing for memory-efficient training."""
import torch
import torch.nn as nn
from typing import Callable, List

def checkpoint_sequential(fns: List[Callable], segments: int, x):
    chunk = max(len(fns)//segments, 1)
    for i in range(0, len(fns), chunk):
        end=min(i+chunk,len(fns))
        x = torch.utils.checkpoint.checkpoint(
            lambda inp,s=i,e=end: _run_fns(fns[s:e], inp), x, use_reentrant=False)
    return x

def _run_fns(fns, x):
    for f in fns: x = f(x)
    return x

def enable_gradient_checkpointing(model: nn.Module, module_class=None) -> nn.Module:
    for m in model.modules():
        if module_class is None or isinstance(m, module_class):
            _wrap(m)
    return model

def _wrap(module: nn.Module):
    orig = module.forward
    def checkpointed_forward(*args, **kwargs):
        def fn(*a): return orig(*a, **kwargs)
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
    module.forward = checkpointed_forward

class ActivationCheckpointingMixin:
    _gradient_checkpointing: bool = False
    def gradient_checkpointing_enable(self): self._gradient_checkpointing = True
    def gradient_checkpointing_disable(self): self._gradient_checkpointing = False
    def _cp(self, fn, *args):
        if self._gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)
