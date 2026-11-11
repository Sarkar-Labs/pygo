"""Vision metrics: mAP, IoU, FID, SSIM, PSNR, LPIPS-style perceptual distance."""
from __future__ import annotations
import math
import numpy as np
from typing import List, Tuple, Dict


def iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def batch_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Vectorized IoU matrix [N, M]."""
    area1 = (boxes1[:, 2]-boxes1[:, 0]) * (boxes1[:, 3]-boxes1[:, 1])
    area2 = (boxes2[:, 2]-boxes2[:, 0]) * (boxes2[:, 3]-boxes2[:, 1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """11-point or all-point interpolated AP (Pascal VOC / COCO style)."""
    mrec  = np.concatenate(([0.], recalls, [1.]))
    mprec = np.concatenate(([0.], precisions, [0.]))
    for i in range(len(mprec) - 1, 0, -1):
        mprec[i - 1] = max(mprec[i - 1], mprec[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mprec[idx + 1]))


def mean_average_precision(
    pred_boxes:  List[np.ndarray],  # per-image [N,4]
    pred_scores: List[np.ndarray],  # per-image [N]
    pred_labels: List[np.ndarray],  # per-image [N]
    gt_boxes:    List[np.ndarray],
    gt_labels:   List[np.ndarray],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Computes mAP@iou_threshold across all classes (COCO-style)."""
    aps = []
    for cls in range(num_classes):
        all_scores, all_matches = [], []
        n_gt_total = 0

        for img_idx in range(len(pred_boxes)):
            p_mask = pred_labels[img_idx] == cls
            g_mask = gt_labels[img_idx]   == cls
            p_box, p_score = pred_boxes[img_idx][p_mask], pred_scores[img_idx][p_mask]
            g_box = gt_boxes[img_idx][g_mask]
            n_gt_total += len(g_box)

            matched = np.zeros(len(g_box), dtype=bool)
            order = np.argsort(-p_score)

            for idx in order:
                box, score = p_box[idx], p_score[idx]
                if len(g_box) == 0:
                    all_scores.append(score); all_matches.append(0)
                    continue
                ious = batch_iou(box[None], g_box)[0]
                best = np.argmax(ious)
                if ious[best] >= iou_threshold and not matched[best]:
                    matched[best] = True
                    all_scores.append(score); all_matches.append(1)
                else:
                    all_scores.append(score); all_matches.append(0)

        if n_gt_total == 0:
            continue

        order = np.argsort(-np.array(all_scores))
        matches = np.array(all_matches)[order]
        tp = np.cumsum(matches)
        fp = np.cumsum(1 - matches)
        recalls    = tp / n_gt_total
        precisions = tp / np.clip(tp + fp, 1e-9, None)
        aps.append(average_precision(recalls, precisions))

    return {
        "mAP": float(np.mean(aps)) if aps else 0.0,
        "AP_per_class": aps,
    }


def ssim(img1: np.ndarray, img2: np.ndarray, window_size: int = 11,
        C1: float = 0.01**2, C2: float = 0.03**2) -> float:
    """Structural Similarity Index (Wang et al. 2004)."""
    from scipy.ndimage import uniform_filter

    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    mu1 = uniform_filter(img1, window_size)
    mu2 = uniform_filter(img2, window_size)

    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = uniform_filter(img1**2, window_size) - mu1_sq
    sigma2_sq = uniform_filter(img2**2, window_size) - mu2_sq
    sigma12   = uniform_filter(img1*img2, window_size) - mu1_mu2

    num = (2*mu1_mu2 + C1) * (2*sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return float(np.mean(num / den))


def psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """Peak Signal-to-Noise Ratio."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(max_val / math.sqrt(mse))


def frechet_inception_distance(real_features: np.ndarray,
                                fake_features: np.ndarray) -> float:
    """FID — Wasserstein-2 distance between Gaussians fit to Inception features."""
    from scipy import linalg

    mu1, sigma1 = real_features.mean(0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(0), np.cov(fake_features, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def inception_score(probs: np.ndarray, splits: int = 10) -> Tuple[float, float]:
    """Inception Score: exp(E[KL(p(y|x) || p(y))])."""
    N = probs.shape[0]
    scores = []
    split_size = N // splits

    for i in range(splits):
        part = probs[i * split_size: (i + 1) * split_size]
        py = part.mean(axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-12) - np.log(py + 1e-12))
        kl = kl.sum(axis=1)
        scores.append(np.exp(kl.mean()))

    return float(np.mean(scores)), float(np.std(scores))


def dice_coefficient(pred_mask: np.ndarray, gt_mask: np.ndarray,
                     smooth: float = 1e-6) -> float:
    """Dice coefficient (F1 for segmentation)."""
    pred = pred_mask.astype(bool).flatten()
    gt   = gt_mask.astype(bool).flatten()
    intersection = np.logical_and(pred, gt).sum()
    return float((2.0 * intersection + smooth) / (pred.sum() + gt.sum() + smooth))


def confusion_matrix(preds: np.ndarray, labels: np.ndarray,
                     num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, l in zip(preds, labels):
        cm[l, p] += 1
    return cm


def pixel_accuracy(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    return float((pred_mask == gt_mask).mean())


def mean_iou_segmentation(pred_mask: np.ndarray, gt_mask: np.ndarray,
                          num_classes: int) -> float:
    """Mean IoU across all classes for semantic segmentation."""
    ious = []
    for cls in range(num_classes):
        p = (pred_mask == cls)
        g = (gt_mask   == cls)
        intersection = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        if union > 0:
            ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0
