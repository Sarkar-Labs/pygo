"""Device management utility: device_properties."""
from __future__ import annotations
import torch

def get_device(prefer_cuda=True):
    if prefer_cuda and torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

def move(obj, device=None):
    device = get_device() if device is None else torch.device(device)
    return obj.to(device) if hasattr(obj,"to") else obj

class DeviceProperties:
    def __init__(self, device=None): self.device=torch.device(device) if device else get_device()
    def __call__(self,obj): return move(obj,self.device)
    def synchronize(self):
        if self.device.type=="cuda": torch.cuda.synchronize(self.device)
