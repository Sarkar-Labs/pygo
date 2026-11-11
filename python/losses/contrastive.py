"""Contrastive & self-supervised losses: InfoNCE, SupCon, CLIP, DINO, BYOL, VICReg."""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from torch import Tensor


def nt_xent_loss(z1: Tensor, z2: Tensor, temperature: float = 0.5) -> Tensor:
    """Normalized Temperature-scaled Cross Entropy (SimCLR)."""
    B = z1.shape[0]
    z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
    z = torch.cat([z1, z2], dim=0)               # [2B, D]
    sim = z @ z.T / temperature                  # [2B, 2B]
    sim.fill_diagonal_(-1e9)

    targets = torch.cat([
        torch.arange(B, 2 * B), torch.arange(0, B)
    ]).to(z.device)
    return F.cross_entropy(sim, targets)


def supervised_contrastive_loss(features: Tensor, labels: Tensor,
                                temperature: float = 0.07) -> Tensor:
    """SupCon loss (Khosla et al. 2020) — extends InfoNCE to labeled batches."""
    device = features.device
    B = features.shape[0]
    features = F.normalize(features, dim=-1)

    sim = features @ features.T / temperature
    sim_max, _ = sim.max(dim=1, keepdim=True)
    logits = sim - sim_max.detach()

    mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float().to(device)
    self_mask = torch.eye(B, device=device)
    mask = mask * (1 - self_mask)

    exp_logits = torch.exp(logits) * (1 - self_mask)
    log_prob   = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
    return -mean_log_prob_pos.mean()


def clip_loss(image_emb: Tensor, text_emb: Tensor,
              logit_scale: Tensor) -> Tensor:
    """CLIP symmetric contrastive loss between image and text embeddings."""
    image_emb = F.normalize(image_emb, dim=-1)
    text_emb  = F.normalize(text_emb,  dim=-1)

    logits_per_image = logit_scale.exp() * image_emb @ text_emb.T
    logits_per_text  = logits_per_image.T

    B = image_emb.shape[0]
    labels = torch.arange(B, device=image_emb.device)

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text,  labels)
    return (loss_i + loss_t) / 2


def byol_loss(p1: Tensor, z2: Tensor) -> Tensor:
    """BYOL prediction-target negative cosine similarity (Grill et al. 2020)."""
    p1 = F.normalize(p1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    return 2 - 2 * (p1 * z2).sum(dim=-1).mean()


def simsiam_loss(p1: Tensor, z1: Tensor, p2: Tensor, z2: Tensor) -> Tensor:
    """SimSiam symmetric stop-gradient loss."""
    def neg_cos_sim(p, z):
        z = z.detach()
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return -(p * z).sum(dim=-1).mean()
    return 0.5 * neg_cos_sim(p1, z2) + 0.5 * neg_cos_sim(p2, z1)


def barlow_twins_loss(z1: Tensor, z2: Tensor, lambda_param: float = 5e-3) -> Tensor:
    """Barlow Twins redundancy-reduction loss (Zbontar et al. 2021)."""
    B, D = z1.shape
    z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
    z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)

    cross_corr = (z1.T @ z2) / B           # [D, D]
    on_diag  = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
    off_diag = _off_diagonal(cross_corr).pow_(2).sum()
    return on_diag + lambda_param * off_diag


def _off_diagonal(x: Tensor) -> Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(z1: Tensor, z2: Tensor,
                sim_coeff: float = 25.0, std_coeff: float = 25.0,
                cov_coeff: float = 1.0) -> Tensor:
    """VICReg: Variance-Invariance-Covariance Regularization (Bardes et al. 2022)."""
    B, D = z1.shape

    # Invariance term
    inv_loss = F.mse_loss(z1, z2)

    # Variance term — encourage each dim to have std ≥ 1
    z1 = z1 - z1.mean(0)
    z2 = z2 - z2.mean(0)
    std1 = torch.sqrt(z1.var(0) + 1e-4)
    std2 = torch.sqrt(z2.var(0) + 1e-4)
    var_loss = (F.relu(1 - std1).mean() + F.relu(1 - std2).mean()) / 2

    # Covariance term — decorrelate features
    cov1 = (z1.T @ z1) / (B - 1)
    cov2 = (z2.T @ z2) / (B - 1)
    cov_loss = (_off_diagonal(cov1).pow_(2).sum() / D +
                _off_diagonal(cov2).pow_(2).sum() / D)

    return sim_coeff * inv_loss + std_coeff * var_loss + cov_coeff * cov_loss


def dino_loss(student_logits: Tensor, teacher_logits: Tensor,
             student_temp: float = 0.1, teacher_temp: float = 0.04,
             center: Tensor = None) -> Tensor:
    """DINO self-distillation loss (Caron et al. 2021)."""
    student_out = student_logits / student_temp
    teacher_out = F.softmax((teacher_logits - center) / teacher_temp, dim=-1)
    teacher_out = teacher_out.detach()

    loss = -(teacher_out * F.log_softmax(student_out, dim=-1)).sum(dim=-1)
    return loss.mean()


class DINOLossModule(torch.nn.Module):
    """Stateful DINO loss with center EMA updating."""
    def __init__(self, out_dim: int, teacher_temp: float = 0.04,
                student_temp: float = 0.1, center_momentum: float = 0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp  = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_out: Tensor, teacher_out: Tensor) -> Tensor:
        loss = dino_loss(student_out, teacher_out,
                         self.student_temp, self.teacher_temp, self.center)
        self._update_center(teacher_out)
        return loss

    @torch.no_grad()
    def _update_center(self, teacher_out: Tensor):
        batch_center = teacher_out.mean(dim=0, keepdim=True)
        self.center = (self.center * self.center_momentum +
                      batch_center * (1 - self.center_momentum))


def triplet_margin_loss(anchor: Tensor, positive: Tensor, negative: Tensor,
                        margin: float = 1.0, p: float = 2.0) -> Tensor:
    """Standard triplet loss: max(0, d(a,p) - d(a,n) + margin)."""
    d_pos = F.pairwise_distance(anchor, positive, p=p)
    d_neg = F.pairwise_distance(anchor, negative, p=p)
    return F.relu(d_pos - d_neg + margin).mean()


def angular_margin_loss(embeddings: Tensor, labels: Tensor, weight: Tensor,
                        margin: float = 0.5, scale: float = 64.0) -> Tensor:
    """ArcFace additive angular margin loss for face/metric learning."""
    embeddings = F.normalize(embeddings, dim=-1)
    weight     = F.normalize(weight, dim=-1)
    cosine = embeddings @ weight.T
    cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)

    theta = torch.acos(cosine)
    target_logits = torch.cos(theta + margin)

    one_hot = F.one_hot(labels, num_classes=weight.shape[0]).float()
    output  = one_hot * target_logits + (1 - one_hot) * cosine
    return F.cross_entropy(output * scale, labels)


def lifted_structured_loss(embeddings: Tensor, labels: Tensor,
                           margin: float = 1.0) -> Tensor:
    """Lifted Structured Feature Embedding (Song et al. 2016)."""
    B = embeddings.shape[0]
    dist = torch.cdist(embeddings, embeddings, p=2)
    pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    neg_mask = ~pos_mask

    loss_sum = 0.0
    count = 0
    for i in range(B):
        for j in range(B):
            if i != j and pos_mask[i, j]:
                neg_i = torch.logsumexp(margin - dist[i][neg_mask[i]], dim=0)
                neg_j = torch.logsumexp(margin - dist[j][neg_mask[j]], dim=0)
                J = F.relu(torch.logsumexp(torch.stack([neg_i, neg_j]), dim=0) + dist[i, j])
                loss_sum += J ** 2
                count += 1
    return loss_sum / max(count, 1)
