#!/usr/bin/env python3
"""Post-hoc training of the error-prediction (uncertainty) head.

The SR generator is frozen.  The head learns to map its pre-upsampling
feature map to a normalised per-pixel error proxy
    t = e / (e + tau),  e = |SR - HR|,
which is monotone in the error and bounded in [0, 1].
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import TrainDataset
from .degradation import degrade
from .infer import load_checkpoint
from .models import SRWithUncertainty

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    g, gargs = load_checkpoint(args.ckpt, device)
    model = SRWithUncertainty(g, 64).to(device)
    for p in model.g.parameters():
        p.requires_grad_(False)
    model.g.eval()
    opt = torch.optim.Adam(model.unc.parameters(), lr=args.lr)
    ds = TrainDataset(patch=192, sigmas=(0.8,))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)

    step = 0
    while step < args.steps:
        for b in dl:
            hr = b["hr"].to(device)
            lr = degrade(hr, scale=4, sigma=0.8)
            with torch.no_grad():
                base = F.interpolate(lr, scale_factor=model.g.scale, mode="bicubic", align_corners=False)
                f = model.g.head(lr)
                f = f + model.g.body_tail(model.g.body(f))
                sr = (base + model.g.tail(model.g.up(f))).clamp(0, 1)
            u = model.unc(f)
            u = F.interpolate(u, size=sr.shape[-2:], mode="bilinear", align_corners=False)
            e = (sr - hr).abs()
            target = e / (e + args.tau)
            loss = F.l1_loss(u, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 100 == 0:
                print(f"[unc {args.tag}] {step} loss={loss.item():.4f}", flush=True)
            step += 1
            if step >= args.steps:
                break

    ckpt = {"model": model.state_dict(), "args": {**gargs, "tau": args.tau}}
    out = os.path.join(ROOT, "checkpoints", f"{args.tag}_unc.pt")
    torch.save(ckpt, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
