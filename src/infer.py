"""Checkpoint loading and batched inference helpers."""
from __future__ import annotations

import os

import torch

from .models import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_checkpoint(path: str, device="cpu", uncertainty: bool = False):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    arch = args.get("arch", "rrdb")
    model = build_model(arch, uncertainty=uncertainty)
    sd = ckpt["model"]
    if uncertainty and not any(k.startswith("g.") for k in sd):
        # a plain generator checkpoint: wrap it
        model.g.load_state_dict(sd)
    else:
        model.load_state_dict(sd)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, args


@torch.no_grad()
def sr_forward(model, lr: torch.Tensor, beta: float = 0.0):
    out = model(lr)
    if isinstance(out, tuple):
        sr, u = out[0].clamp(0, 1), out[1]
        if beta > 0:
            w = torch.exp(-beta * u)
            base = torch.nn.functional.interpolate(
                lr, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1)
            sr = (w * sr + (1 - w) * base).clamp(0, 1)
        return sr, u
    return out.clamp(0, 1), None
