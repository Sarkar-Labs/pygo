"""Image and text data augmentation transforms for PyGo."""
import random
import numpy as np
from typing import List, Tuple, Optional, Callable, Any


class Compose:
    """Apply a list of transforms sequentially."""
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    def __call__(self, x: Any) -> Any:
        for t in self.transforms:
            x = t(x)
        return x


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5): self.p = p
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img[:, ::-1].copy() if random.random() < self.p else img


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5): self.p = p
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img[::-1].copy() if random.random() < self.p else img


class RandomCrop:
    def __init__(self, size: int, padding: int = 0):
        self.size    = size
        self.padding = padding
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.padding > 0:
            img = np.pad(img, ((self.padding,)*2,(self.padding,)*2,(0,0)), mode="reflect")
        H, W = img.shape[:2]
        top  = random.randint(0, H - self.size)
        left = random.randint(0, W - self.size)
        return img[top:top+self.size, left:left+self.size]


class CenterCrop:
    def __init__(self, size: int): self.size = size
    def __call__(self, img: np.ndarray) -> np.ndarray:
        H, W = img.shape[:2]
        top  = (H - self.size) // 2
        left = (W - self.size) // 2
        return img[top:top+self.size, left:left+self.size]


class Resize:
    def __init__(self, size: Tuple[int,int]): self.size = size
    def __call__(self, img: np.ndarray) -> np.ndarray:
        try:
            from PIL import Image
            pil = Image.fromarray(img)
            return np.array(pil.resize(self.size[::-1], Image.BILINEAR))
        except ImportError:
            return img


class Normalize:
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = np.array(mean, dtype=np.float32)
        self.std  = np.array(std,  dtype=np.float32)
    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32) / 255.0
        return (img - self.mean) / self.std


class ColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0):
        self.brightness = brightness
        self.contrast   = contrast
        self.saturation = saturation
        self.hue        = hue

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        if self.brightness > 0:
            f = 1 + random.uniform(-self.brightness, self.brightness)
            img = img * f
        if self.contrast > 0:
            mean = img.mean()
            f = 1 + random.uniform(-self.contrast, self.contrast)
            img = (img - mean) * f + mean
        return np.clip(img, 0, 255).astype(np.uint8)


class RandomErasing:
    """Randomly erase a rectangular region (Zhong et al. 2020)."""
    def __init__(self, p=0.5, scale=(0.02,0.33), ratio=(0.3,3.3), value=0):
        self.p = p; self.scale = scale; self.ratio = ratio; self.value = value
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.p: return img
        H, W = img.shape[:2]
        for _ in range(100):
            area = random.uniform(*self.scale) * H * W
            ar   = random.uniform(*self.ratio)
            eh   = int((area * ar) ** 0.5)
            ew   = int((area / ar) ** 0.5)
            if eh < H and ew < W:
                top  = random.randint(0, H - eh)
                left = random.randint(0, W - ew)
                img[top:top+eh, left:left+ew] = self.value
                break
        return img


class Cutout:
    """Cutout augmentation (DeVries & Taylor 2017)."""
    def __init__(self, n_holes: int = 1, size: int = 16):
        self.n_holes = n_holes; self.size = size
    def __call__(self, img: np.ndarray) -> np.ndarray:
        H, W = img.shape[:2]
        for _ in range(self.n_holes):
            cy, cx = random.randint(0, H), random.randint(0, W)
            y1, y2 = max(0, cy-self.size//2), min(H, cy+self.size//2)
            x1, x2 = max(0, cx-self.size//2), min(W, cx+self.size//2)
            img[y1:y2, x1:x2] = 0
        return img


def mixup(x1: np.ndarray, y1: int, x2: np.ndarray, y2: int,
          alpha: float = 0.2) -> Tuple[np.ndarray, float, int, int]:
    """Mixup data augmentation (Zhang et al. 2018)."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    return lam*x1 + (1-lam)*x2, lam, y1, y2


def cutmix(x1: np.ndarray, y1: int, x2: np.ndarray, y2: int,
           alpha: float = 1.0) -> Tuple[np.ndarray, float, int, int]:
    """CutMix augmentation (Yun et al. 2019)."""
    H, W = x1.shape[:2]
    lam = np.random.beta(alpha, alpha)
    cut_ratio = (1 - lam) ** 0.5
    ch, cw = int(H * cut_ratio), int(W * cut_ratio)
    cx, cy = random.randint(0, W), random.randint(0, H)
    x1c, y1c = max(cx-cw//2, 0), max(cy-ch//2, 0)
    x2c, y2c = min(cx+cw//2, W), min(cy+ch//2, H)
    out = x1.copy()
    out[y1c:y2c, x1c:x2c] = x2[y1c:y2c, x1c:x2c]
    lam_real = 1 - (y2c-y1c)*(x2c-x1c)/(H*W)
    return out, lam_real, y1, y2
