"""Loss functions for PyGo."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2., alpha: float = 0.25):
        super().__init__(); self.gamma = gamma; self.alpha = alpha
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        probs = logits.sigmoid(); pt = torch.where(targets==1, probs, 1-probs)
        alpha_t = torch.where(targets==1, torch.tensor(self.alpha), torch.tensor(1-self.alpha))
        return (-(alpha_t * (1-pt)**self.gamma * (pt+1e-8).log())).mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.):
        super().__init__(); self.smooth = smooth
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        inter = (pred * target).sum(dim=(2,3))
        denom = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
        return 1 - ((2*inter + self.smooth) / (denom + self.smooth)).mean()

class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__(); self.smoothing = smoothing
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        log_probs = F.log_softmax(pred, -1)
        nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        smooth_loss = -log_probs.mean(-1)
        return ((1-self.smoothing)*nll + self.smoothing*smooth_loss).mean()

class VARLoss(nn.Module):
    def forward(self, recon, x, mu, log_var):
        bce = F.mse_loss(recon, x, reduction="sum") / x.shape[0]
        kl  = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1).mean()
        return bce + kl