#!/usr/bin/env python3
"""Train an MRI SR model with a configurable objective.

Objectives
----------
l1        : pixel L1 (baseline)
l1_dsc    : L1 + degradation self-consistency (our proposal)
l1_perc   : L1 + AlexNet perceptual loss (sharpness-oriented comparison)
l1_dsc_perc: all three
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import TrainDataset, EvalDataset
from .degradation import degrade
from .metrics import psnr, ssim
from .models import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PerceptualLoss(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        import torchvision
        net = torchvision.models.alexnet(weights="DEFAULT").features[:12]
        self.net = net.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.to(device)

    def forward(self, sr, hr):
        x = sr.repeat(1, 3, 1, 1)
        y = hr.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        return F.l1_loss(self.net(x), self.net(y))


def make_loss(kind, device):
    perc = PerceptualLoss(device) if "perc" in kind else None
    def loss_fn(sr, hr, lr, sigma):
        loss = F.l1_loss(sr, hr)
        parts = {"l1": loss.item()}
        if "dsc" in kind:
            d = F.l1_loss(degrade(sr, scale=4, sigma=sigma), lr)
            loss = loss + 0.5 * d
            parts["dsc"] = d.item()
        if perc is not None:
            p = perc(sr, hr)
            loss = loss + 0.02 * p
            parts["perc"] = p.item()
        return loss, parts
    return loss_fn


@torch.no_grad()
def validate(model, device, n=24, scale=4):
    ds = EvalDataset("val")
    idx = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int).tolist()
    ps, ss = [], []
    for i in idx:
        b = ds[int(i)]
        hr = b["hr"][None].to(device)
        lr = degrade(hr, scale=scale, sigma=0.8)
        sr = model(lr).clamp(0, 1)
        ps.append(psnr(sr, hr).item())
        ss.append(ssim(sr, hr).item())
    return float(np.mean(ps)), float(np.mean(ss))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arch", default="rrdb", choices=["rrdb", "srresnet"])
    ap.add_argument("--loss", default="l1", choices=["l1", "l1_dsc", "l1_perc", "l1_dsc_perc"])
    ap.add_argument("--sigmas", default="0.8", help="comma separated blur sigmas; 'multi' = 0.5,0.8,1.1,1.4")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patch", type=int, default=192)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if args.smoke:
        args.steps = 20
        args.val_every = 10

    sigmas = (0.5, 0.8, 1.1, 1.4) if args.sigmas == "multi" else tuple(float(s) for s in args.sigmas.split(","))
    ds = TrainDataset(patch=args.patch, sigmas=sigmas)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True, persistent_workers=False)

    model = build_model(args.arch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    warm = 200

    def lr_at(step):
        if step < warm:
            return args.lr * (step + 1) / warm
        t = (step - warm) / max(1, args.steps - warm)
        return 1e-6 + 0.5 * (args.lr - 1e-6) * (1 + np.cos(np.pi * t))

    loss_fn = make_loss(args.loss, device)
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "checkpoints"), exist_ok=True)
    out = args.out or os.path.join(ROOT, "checkpoints", f"{args.tag}.pt")
    logf = open(os.path.join(ROOT, "logs", f"{args.tag}.jsonl"), "w")

    step, best = 0, -1e9
    t0 = time.time()
    running = {}
    while step < args.steps:
        for batch in dl:
            if step >= args.steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            hr = batch["hr"].to(device)
            sigma = float(np.random.choice(sigmas))
            lr = degrade(hr, scale=4, sigma=sigma)
            sr = model(lr)
            loss, parts = loss_fn(sr, hr, lr, sigma)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            for k, v in parts.items():
                running[k] = 0.98 * running.get(k, v) + 0.02 * v
            if step % 50 == 0:
                msg = {"step": step, "loss": float(loss.item()), "lr": lr_at(step), **{k: round(v, 5) for k, v in running.items()}}
                print(f"[{args.tag}] {step} loss={loss.item():.4f} " + " ".join(f"{k}={v:.4f}" for k, v in parts.items()), flush=True)
                logf.write(json.dumps(msg) + "\n"); logf.flush()
            if step > 0 and step % args.val_every == 0:
                p, s = validate(model, device)
                print(f"[{args.tag}] step {step} val psnr={p:.3f} ssim={s:.4f} ({time.time()-t0:.0f}s)", flush=True)
                logf.write(json.dumps({"step": step, "val_psnr": p, "val_ssim": s}) + "\n"); logf.flush()
                if p > best:
                    best = p
                    torch.save({"model": model.state_dict(), "args": vars(args), "val_psnr": p}, out)
            step += 1
    p, s = validate(model, device)
    if p > best:
        best = p
        torch.save({"model": model.state_dict(), "args": vars(args), "val_psnr": p}, out)
    torch.save({"model": model.state_dict(), "args": vars(args), "val_psnr": p}, os.path.join(ROOT, "checkpoints", f"{args.tag}_last.pt"))
    print(f"done {args.tag}: val psnr={p:.3f} ssim={s:.4f} best={best:.3f} time={time.time()-t0:.0f}s", flush=True)
    logf.write(json.dumps({"done": True, "val_psnr": p, "val_ssim": s, "best": best, "time": time.time() - t0}) + "\n")
    logf.close()


if __name__ == "__main__":
    main()
