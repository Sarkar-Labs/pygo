"""SwiGLU, GEGLU, ReGLU activations."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class SwiGLU(nn.Module):
    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d, bias=False)
    def forward(self, x: Tensor) -> Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class GEGLU(nn.Module):
    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.proj = nn.Linear(d, d_ff * 2, bias=False)
        self.out  = nn.Linear(d_ff, d, bias=False)
    def forward(self, x: Tensor) -> Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return self.out(x * F.gelu(gate))

class ReGLU(nn.Module):
    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.proj = nn.Linear(d, d_ff * 2, bias=False)
        self.out  = nn.Linear(d_ff, d, bias=False)
    def forward(self, x: Tensor) -> Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return self.out(x * F.relu(gate))

class GatedMLP(nn.Module):
    def __init__(self, d: int, d_ff: int, activation: str = "silu"):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up   = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)
        self.act  = {"silu": F.silu, "gelu": F.gelu, "relu": F.relu}[activation]
    def forward(self, x: Tensor) -> Tensor:
        return self.down(self.act(self.gate(x)) * self.up(x))