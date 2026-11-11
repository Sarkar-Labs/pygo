"""DDPM and DDIM diffusion models."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

@dataclass
class DDPMConfig:
    in_channels: int = 3; base_ch: int = 128; ch_mult: tuple = (1,2,4,8)
    timesteps: int = 1000; beta_start: float = 0.0001; beta_end: float = 0.02

class SinTimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__(); self.dim = dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2; device = t.device
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / (half-1))
        args  = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Sequential(nn.GroupNorm(8, in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.time  = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.conv2 = nn.Sequential(nn.GroupNorm(8, out_ch), nn.SiLU(), nn.Dropout(dropout), nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, t_emb):
        h = self.conv1(x) + self.time(t_emb)[:, :, None, None]
        return self.conv2(h) + self.skip(x)

class UNetDDPM(nn.Module):
    def __init__(self, cfg: DDPMConfig = None):
        super().__init__(); self.cfg = cfg or DDPMConfig()
        td = self.cfg.base_ch * 4
        self.time_emb = nn.Sequential(SinTimeEmbed(self.cfg.base_ch), nn.Linear(self.cfg.base_ch, td), nn.SiLU(), nn.Linear(td, td))
        self.in_conv  = nn.Conv2d(self.cfg.in_channels, self.cfg.base_ch, 3, padding=1)
        self.out_conv = nn.Sequential(nn.GroupNorm(8, self.cfg.base_ch), nn.SiLU(), nn.Conv2d(self.cfg.base_ch, self.cfg.in_channels, 3, padding=1))
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t.float())
        h = self.in_conv(x); _ = t_emb
        return self.out_conv(h)

class DDPMScheduler:
    def __init__(self, cfg: DDPMConfig = None):
        self.cfg = cfg or DDPMConfig()
        betas = torch.linspace(self.cfg.beta_start, self.cfg.beta_end, self.cfg.timesteps)
        alphas = 1 - betas; self.alpha_cumprod = alphas.cumprod(0)
    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None: noise = torch.randn_like(x0)
        ac = self.alpha_cumprod[t].view(-1, 1, 1, 1).to(x0.device)
        return ac.sqrt() * x0 + (1-ac).sqrt() * noise, noise