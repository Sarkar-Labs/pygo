"""Vision Transformer (ViT) variants."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ViTConfig:
    image_size: int = 224; patch_size: int = 16; in_channels: int = 3
    embed_dim: int = 768; depth: int = 12; num_heads: int = 12
    mlp_ratio: float = 4.; num_classes: int = 1000; drop_rate: float = 0.

class ViTBlock(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__(); d = cfg.embed_dim; ff = int(d * cfg.mlp_ratio)
        self.norm1 = nn.LayerNorm(d); self.attn = nn.MultiheadAttention(d, cfg.num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d); self.drop = nn.Dropout(cfg.drop_rate)
        self.mlp = nn.Sequential(nn.Linear(d,ff), nn.GELU(), nn.Dropout(cfg.drop_rate), nn.Linear(ff,d), nn.Dropout(cfg.drop_rate))
    def forward(self, x):
        a,_ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.drop(a); return x + self.drop(self.mlp(self.norm2(x)))

class ViT(nn.Module):
    def __init__(self, cfg: ViTConfig = None):
        super().__init__(); self.cfg = cfg or ViTConfig()
        d = self.cfg.embed_dim; n = (self.cfg.image_size // self.cfg.patch_size) ** 2
        self.patch = nn.Conv2d(self.cfg.in_channels, d, self.cfg.patch_size, stride=self.cfg.patch_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, n+1, d))
        self.blocks = nn.ModuleList([ViTBlock(self.cfg) for _ in range(self.cfg.depth)])
        self.norm = nn.LayerNorm(d); self.head = nn.Linear(d, self.cfg.num_classes)
    def forward(self, x):
        B = x.shape[0]; x = self.patch(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(B,-1,-1), x], dim=1) + self.pos
        for blk in self.blocks: x = blk(x)
        return self.head(self.norm(x[:, 0]))

def vit_tiny():  return ViT(ViTConfig(embed_dim=192, depth=12, num_heads=3))
def vit_small(): return ViT(ViTConfig(embed_dim=384, depth=12, num_heads=6))
def vit_base():  return ViT(ViTConfig(embed_dim=768, depth=12, num_heads=12))
def vit_large(): return ViT(ViTConfig(embed_dim=1024, depth=24, num_heads=16))