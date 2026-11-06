"""Image-quality metrics for MRI super-resolution.

All functions take float tensors in [0, 1] with shape (B, 1, H, W).
Implementations are batched and device-agnostic so the same code runs on
MPS during development and on CPU during final evaluation.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def _gaussian_window(size: int = 11, sigma: float = 1.5, device=None, dtype=torch.float32):
    x = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.view(1, 1, 1, -1)


def ssim(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Mean SSIM per image (Gaussian window, 11x11, sigma=1.5)."""
    c = x.shape[1]
    win = _gaussian_window(11, 1.5, x.device, x.dtype)
    win2 = win.transpose(-1, -2)
    k = (c, 1, 1, 11)
    wx = win.expand(k)
    wy = win2.expand(c, 1, 11, 1)
    mu_x = F.conv2d(x, wx, padding=(0, 5), groups=c)
    mu_y = F.conv2d(y, wx, padding=(0, 5), groups=c)
    mu_x = F.conv2d(mu_x, wy, padding=(5, 0), groups=c)
    mu_y = F.conv2d(mu_y, wy, padding=(5, 0), groups=c)
    xx = F.conv2d(x * x, wx, padding=(0, 5), groups=c)
    yy = F.conv2d(y * y, wx, padding=(0, 5), groups=c)
    xy = F.conv2d(x * y, wx, padding=(0, 5), groups=c)
    xx = F.conv2d(xx, wy, padding=(5, 0), groups=c)
    yy = F.conv2d(yy, wy, padding=(5, 0), groups=c)
    xy = F.conv2d(xy, wy, padding=(5, 0), groups=c)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    sx = xx - mu_x ** 2
    sy = yy - mu_y ** 2
    sxy = xy - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sx + sy + c2)
    return (num / den).mean(dim=(1, 2, 3))


def psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    mse = ((x - y) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-12)
    return 10.0 * torch.log10(data_range ** 2 / mse)


def sobel_mag(x: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device)
    ky = kx.transpose(0, 1)
    kx = kx.view(1, 1, 3, 3)
    ky = ky.view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


def laplacian(x: torch.Tensor) -> torch.Tensor:
    k = torch.tensor([[0.0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=x.dtype, device=x.device)
    return F.conv2d(x, k.view(1, 1, 3, 3), padding=1)


def hfen(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """High-frequency error norm (normalised by the reference HF energy)."""
    lx, ly = laplacian(x), laplacian(y)
    num = torch.sqrt(((lx - ly) ** 2).sum(dim=(1, 2, 3)) + 1e-8)
    den = torch.sqrt((ly ** 2).sum(dim=(1, 2, 3)) + 1e-8)
    return num / den


def edge_ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    gx, gy = sobel_mag(x), sobel_mag(y)
    scale = gy.flatten(1).amax(dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
    return ssim(gx / scale, gy / scale)


def false_edge_rate(x: torch.Tensor, y: torch.Tensor, brain: torch.Tensor | None = None) -> torch.Tensor:
    """Fraction of brain pixels that are strong edges in x but weak in y.

    Thresholds are relative to the *reference* gradient distribution, so
    the metric does not reward or punish global sharpness.
    """
    gx, gy = sobel_mag(x), sobel_mag(y)
    if brain is None:
        brain = (y > 0.02)
    vals = gy[brain]
    # quantiles per batch are approximated with a global sample for speed
    if vals.numel() == 0:
        return torch.zeros(x.shape[0], device=x.device)
    hi = torch.quantile(vals.float(), 0.95)
    lo = torch.quantile(vals.float(), 0.30)
    false = (gx > hi) & (gy < lo) & brain
    return false.flatten(1).sum(dim=1).float() / brain.flatten(1).sum(dim=1).clamp_min(1).float() * 1e3


def flat_leak(x: torch.Tensor, y: torch.Tensor, q: float = 0.10) -> torch.Tensor:
    """Gradient energy that SR places in regions that are flat in the reference.

    Normalised by the mean reference gradient, so the value is comparable
    across slices.  A blurry output is close to zero; fabricated texture in
    CSF or background increases it.
    """
    gx, gy = sobel_mag(x), sobel_mag(y)
    brain = y > 0.02
    vals = gy[brain]
    if vals.numel() == 0:
        return torch.zeros(x.shape[0], device=x.device)
    thr = torch.quantile(vals.float(), q)
    leak = (gx * ((gy < thr) & brain).float()).flatten(1).sum(dim=1)
    denom = brain.flatten(1).sum(dim=1).clamp_min(1).float()
    ref = gy[brain].mean().clamp_min(1e-8)
    return leak / denom / ref * 100.0


class LPIPS:
    """Thin wrapper around the official LPIPS package (AlexNet backbone)."""

    def __init__(self, device="cpu", net="alex"):
        import lpips
        self.fn = lpips.LPIPS(net=net, verbose=False).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        x = x * 2 - 1
        y = y * 2 - 1
        return self.fn(x, y).flatten()


def all_metrics_vec(sr: torch.Tensor, hr: torch.Tensor) -> dict:
    """Per-image metric tensors (numpy arrays of length B)."""
    return {
        "psnr": psnr(sr, hr).detach().cpu().numpy(),
        "ssim": ssim(sr, hr).detach().cpu().numpy(),
        "hfen": hfen(sr, hr).detach().cpu().numpy(),
        "edge_ssim": edge_ssim(sr, hr).detach().cpu().numpy(),
    }


def all_metrics(sr: torch.Tensor, hr: torch.Tensor, lpips_fn: LPIPS | None = None) -> dict:
    out = {
        "psnr": psnr(sr, hr).mean().item(),
        "ssim": ssim(sr, hr).mean().item(),
        "hfen": hfen(sr, hr).mean().item(),
        "edge_ssim": edge_ssim(sr, hr).mean().item(),
        "fer": false_edge_rate(sr, hr).mean().item(),
        "leak": flat_leak(sr, hr).mean().item(),
    }
    if lpips_fn is not None:
        out["lpips"] = lpips_fn(sr, hr).mean().item()
    return out
