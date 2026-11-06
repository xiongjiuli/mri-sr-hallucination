"""Degradation operators shared by data generation, training and probes.

The in-distribution operator is a Gaussian point-spread function followed
by 4x4 average pooling (`blur_pool`).  It is fully differentiable and
runs on MPS, which lets the same operator be used for the degradation
self-consistency loss.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def gaussian_kernel1d(sigma: float, truncate: float = 3.0, device=None, dtype=torch.float32):
    radius = max(1, int(truncate * sigma + 0.5))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    k = gaussian_kernel1d(sigma, device=x.device, dtype=x.dtype)
    r = (k.numel() - 1) // 2
    c = x.shape[1]
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, k.numel())
    ky = k.view(1, 1, -1, 1).expand(c, 1, k.numel(), 1)
    x = F.conv2d(x, kx, padding=(0, r), groups=c)
    x = F.conv2d(x, ky, padding=(r, 0), groups=c)
    return x


def degrade(hr: torch.Tensor, scale: int = 4, sigma: float = 0.8,
            noise_sigma: float = 0.0, generator: torch.Generator | None = None) -> torch.Tensor:
    """Blur + average-pool + optional Gaussian noise. hr in [0, 1]."""
    x = gaussian_blur(hr, sigma)
    lr = F.avg_pool2d(x, scale, scale)
    if noise_sigma > 0:
        e = torch.randn(lr.shape, device=lr.device, dtype=lr.dtype, generator=generator)
        lr = lr + noise_sigma * e
    return lr.clamp(0, 1)


def degrade_bicubic(hr: torch.Tensor, scale: int = 4) -> torch.Tensor:
    return F.interpolate(hr, scale_factor=1.0 / scale, mode="bicubic",
                         align_corners=False, antialias=True).clamp(0, 1)


def upsample_bicubic(lr: torch.Tensor, scale: int = 4) -> torch.Tensor:
    return F.interpolate(lr, scale_factor=scale, mode="bicubic",
                         align_corners=False).clamp(0, 1)


DEGRADATIONS = {
    "id": dict(scale=4, sigma=0.8, noise_sigma=0.0),
    "blur1.4": dict(scale=4, sigma=1.4, noise_sigma=0.0),
    "noise": dict(scale=4, sigma=0.8, noise_sigma=0.02),
    "mild": dict(scale=4, sigma=0.5, noise_sigma=0.0),
}


def make_lr(hr: torch.Tensor, kind: str = "id", scale: int = 4) -> torch.Tensor:
    if kind == "bicubic":
        return degrade_bicubic(hr, scale)
    cfg = DEGRADATIONS[kind]
    return degrade(hr, scale=scale, **{k: v for k, v in cfg.items() if k != "scale"})
