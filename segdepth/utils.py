"""
utils.py
--------
Small helpers shared across the package.

Functions:
  * get_device   Choose CUDA, Apple MPS or CPU.
  * free_memory  Release cached GPU memory.
  * load_image   Open an image file as RGB.
  * erode_mask / dilate_mask
                 Binary morphology implemented with max-pooling, avoiding an
                 OpenCV or SciPy dependency.
  * normalise    Min-max scale an array to [0, 1].
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def free_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_image(path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _pool(mask: np.ndarray, pixels: int, erode: bool) -> np.ndarray:
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    k = 2 * pixels + 1
    if erode:
        out = -F.max_pool2d(-t, kernel_size=k, stride=1, padding=pixels)
    else:
        out = F.max_pool2d(t, kernel_size=k, stride=1, padding=pixels)
    return out[0, 0].numpy() > 0.5


def erode_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Shrink a binary mask by `pixels`. Removes boundary pixels where depth
    tends to bleed between neighbouring objects."""
    return mask if pixels <= 0 else _pool(mask, pixels, erode=True)


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Grow a binary mask by `pixels`."""
    return mask if pixels <= 0 else _pool(mask, pixels, erode=False)


def normalise(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo + 1e-8)
