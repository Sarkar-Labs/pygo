"""Model implementation for unet_tensorrt."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class UnetTensorrt(nn.Module):
    """A trainable baseline architecture for the unet_tensorrt model entry point."""
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=128, dropout=0.0):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.GELU(),nn.Dropout(dropout))
        self.proj=nn.Linear(hidden_dim,output_dim)
    def forward(self,x):
        if x.ndim>2: x=x.flatten(1)
        if x.shape[-1] != self.encoder[0].in_features:
            if x.shape[-1] < self.encoder[0].in_features: x=F.pad(x,(0,self.encoder[0].in_features-x.shape[-1]))
            else: x=x[...,:self.encoder[0].in_features]
        return self.proj(self.encoder(x))
    def num_parameters(self): return sum(p.numel() for p in self.parameters())

def create(**kwargs): return UnetTensorrt(**kwargs)
