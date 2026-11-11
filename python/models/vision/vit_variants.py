"""ViT model variants and DeiT, DINO wrappers."""
import torch
import torch.nn as nn
from python.nn.vit import VisionTransformer, ViTConfig


class DeiT(VisionTransformer):
    """Data-efficient Image Transformer — Touvron et al. 2021.
    Adds a distillation token alongside the CLS token.
    """
    def __init__(self, cfg: ViTConfig = None):
        super().__init__(cfg)
        d = self.cfg.embed_dim
        self.dist_token = nn.Parameter(torch.zeros(1, 1, d))
        n = (self.cfg.image_size // self.cfg.patch_size) ** 2
        # Extend position embeddings for dist token
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 2, d))
        nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.dist_head = nn.Linear(d, self.cfg.num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_proj(x).flatten(2).transpose(1, 2)
        cls  = self.cls_token.expand(B, -1, -1)
        dist = self.dist_token.expand(B, -1, -1)
        x    = torch.cat([cls, dist, x], dim=1)
        x    = self.drop(x + self.pos_embed)
        x    = self.norm(self.blocks(x))
        cls_out  = self.head(x[:, 0])
        dist_out = self.dist_head(x[:, 1])
        if self.training:
            return cls_out, dist_out
        return (cls_out + dist_out) / 2


class DINO(VisionTransformer):
    """DINO self-supervised ViT backbone."""
    def __init__(self, cfg: ViTConfig = None, out_dim: int = 65536):
        super().__init__(cfg)
        d = self.cfg.embed_dim
        self.head = nn.Sequential(
            nn.Linear(d, 2048), nn.GELU(), nn.Linear(2048, 2048),
            nn.GELU(), nn.Linear(2048, out_dim),
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_proj(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B,-1,-1), x], dim=1)
        x = self.norm(self.blocks(self.drop(x + self.pos_embed)))
        return self.head(x[:, 0])


def deit_tiny(nc=1000):  return DeiT(ViTConfig(embed_dim=192, depth=12, num_heads=3, num_classes=nc))
def deit_small(nc=1000): return DeiT(ViTConfig(embed_dim=384, depth=12, num_heads=6, num_classes=nc))
def deit_base(nc=1000):  return DeiT(ViTConfig(embed_dim=768, depth=12, num_heads=12, num_classes=nc))
def dino_small(): return DINO(ViTConfig(embed_dim=384, depth=12, num_heads=6))
def dino_base():  return DINO(ViTConfig(embed_dim=768, depth=12, num_heads=12))
