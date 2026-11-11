"""Attention block implemented with torch.nn functional attention."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativeAttention(nn.Module):
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
        self.num_heads = num_heads
    def forward(self, x, context=None, mask=None):
        context = x if context is None else context
        q,k,v = self.q(x), self.k(context), self.v(context)
        score = q @ k.transpose(-2,-1) / math.sqrt(q.shape[-1])
        if mask is not None: score = score.masked_fill(~mask.bool(), float('-inf'))
        return self.out(F.dropout(score.softmax(-1) @ v, self.dropout, self.training))
