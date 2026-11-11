"""Positional encodings: Sinusoidal, Learned, ALiBi, RoPE."""
import math
import torch
import torch.nn as nn
from torch import Tensor

class SinusoidalPE(nn.Module):
    def __init__(self, d: int, max_len: int = 5000, dropout: float = 0.):
        super().__init__(); self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d); pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x: Tensor) -> Tensor: return self.drop(x + self.pe[:, :x.size(1)])

class LearnedPE(nn.Module):
    def __init__(self, d: int, max_len: int = 2048):
        super().__init__(); self.emb = nn.Embedding(max_len, d)
        self.register_buffer("pos", torch.arange(max_len).unsqueeze(0))
    def forward(self, x: Tensor) -> Tensor: return x + self.emb(self.pos[:, :x.size(1)])

class ALiBi(nn.Module):
    def __init__(self, n_heads: int):
        super().__init__(); self.n = n_heads
        slopes = torch.pow(2, -torch.arange(1, n_heads+1) * 8 / n_heads)
        self.register_buffer("slopes", slopes)
    def forward(self, q_len: int, k_len: int) -> Tensor:
        pos = torch.arange(k_len).unsqueeze(0) - torch.arange(q_len).unsqueeze(1)
        bias = -pos.abs().unsqueeze(0) * self.slopes.view(-1, 1, 1)
        return bias.unsqueeze(0)