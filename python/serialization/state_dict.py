"""Serialization utility: state_dict."""
from __future__ import annotations
from pathlib import Path
import torch

def save(obj, path, **kwargs):
    """Persist tensors/modules with torch's portable state serialization."""
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    return torch.save(obj, target, **kwargs)

def load(path, map_location="cpu", **kwargs):
    return torch.load(Path(path), map_location=map_location, **kwargs)

class StateDict:
    def save(self, obj, path, **kwargs): return save(obj, path, **kwargs)
    def load(self, path, **kwargs): return load(path, **kwargs)
    def round_trip(self, obj, path):
        self.save(obj, path); return self.load(path)
