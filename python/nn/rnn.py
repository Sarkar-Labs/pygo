"""RNN, LSTM, GRU modules."""
import torch
import torch.nn as nn
from typing import Optional, Tuple

class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__(); self.hs = hidden_size
        self.linear = nn.Linear(input_size + hidden_size, 4 * hidden_size)
    def forward(self, x, state=None):
        B = x.shape[0]
        h, c = state if state is not None else (torch.zeros(B,self.hs,device=x.device), torch.zeros(B,self.hs,device=x.device))
        gates = self.linear(torch.cat([x, h], -1))
        i, f, g, o = gates.chunk(4, -1)
        c2 = torch.sigmoid(f)*c + torch.sigmoid(i)*torch.tanh(g)
        h2 = torch.sigmoid(o)*torch.tanh(c2)
        return h2, (h2, c2)

class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__(); self.hs = hidden_size
        self.rz = nn.Linear(input_size + hidden_size, 2 * hidden_size)
        self.n  = nn.Linear(input_size + hidden_size, hidden_size)
    def forward(self, x, h=None):
        B = x.shape[0]
        if h is None: h = torch.zeros(B, self.hs, device=x.device)
        r, z = torch.sigmoid(self.rz(torch.cat([x, h], -1))).chunk(2, -1)
        n = torch.tanh(self.n(torch.cat([x, r*h], -1)))
        return (1-z)*h + z*n