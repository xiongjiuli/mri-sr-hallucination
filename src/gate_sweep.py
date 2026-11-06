#!/usr/bin/env python3
"""Validation sweep for the uncertainty gate strength beta.

For each beta we compute PSNR/SSIM on unmodified validation slices and the
COP hallucination energy on occluded validation slices.  The operating
point is the smallest HE subject to a PSNR drop of at most 0.05 dB.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .data import EvalDataset
from .degradation import degrade
from .infer import load_checkpoint, sr_forward
from .metrics import psnr, ssim, sobel_mag
from .probes import pick_patches

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="trust")
    ap.add_argument("--betas", default="0,1,2,4,8,16,32")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--n_cop", type=int, default=80)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _ = load_checkpoint(args.ckpt, device, uncertainty=True)
    betas = [float(b) for b in args.betas.split(",")]
    ds = EvalDataset("val")
    idx = np.linspace(0, len(ds) - 1, min(args.n, len(ds))).astype(int)

    alphas = [round(a, 1) for a in np.arange(0.1, 1.0, 0.1)]
    ps = {b: [] for b in betas}; ss = {b: [] for b in betas}
    aps = {a: [] for a in alphas}; ass = {a: [] for a in alphas}
    for i in idx:
        b = ds[int(i)]
        hr = b["hr"][None].to(device)
        lr = degrade(hr, scale=4, sigma=0.8)
        sr, u = sr_forward(model, lr, beta=0.0)
        base = F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1)
        for beta in betas:
            w = torch.exp(-beta * u) if beta > 0 else torch.ones_like(u)
            out = (w * sr + (1 - w) * base).clamp(0, 1)
            ps[beta].append(psnr(out, hr).item()); ss[beta].append(ssim(out, hr).item())
        for a in alphas:
            out = (a * sr + (1 - a) * base).clamp(0, 1)
            aps[a].append(psnr(out, hr).item()); ass[a].append(ssim(out, hr).item())

    rng = np.random.default_rng(11)
    cop_idx = np.linspace(0, len(ds) - 1, min(args.n_cop, len(ds))).astype(int)
    hes = {b: [] for b in betas}; vrs = {b: [] for b in betas}
    ahes = {a: [] for a in alphas}; avrs = {a: [] for a in alphas}
    for i in cop_idx:
        b = ds[int(i)]
        hr_np = b["hr"][0].numpy()
        patches = pick_patches(hr_np, 4, rng=rng)
        if not patches:
            continue
        occ = hr_np.copy()
        for y, x, _ in patches:
            occ[y:y + 48, x:x + 48] = hr_np[y:y + 48, x:x + 48].mean()
        hr = torch.from_numpy(hr_np)[None, None].to(device)
        occ_t = torch.from_numpy(occ)[None, None].to(device)
        lr = degrade(occ_t, scale=4, sigma=0.8)
        sr, u = sr_forward(model, lr, beta=0.0)
        base = F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1)
        g_hr = sobel_mag(hr)[0, 0].cpu().numpy()
        for beta in betas:
            w = torch.exp(-beta * u) if beta > 0 else torch.ones_like(u)
            out = (w * sr + (1 - w) * base).clamp(0, 1)
            g = sobel_mag(out)[0, 0].cpu().numpy()
            for y, x, _ in patches:
                s_, t_ = slice(y + 8, y + 40), slice(x + 8, x + 40)
                hes[beta].append(float(g[s_, t_].mean() / (g_hr[s_, t_].mean() + 1e-8)))
                vrs[beta].append(float(out[0, 0, s_, t_].std().item() / (hr_np[s_, t_].std() + 1e-8)))
        for a in alphas:
            out = (a * sr + (1 - a) * base).clamp(0, 1)
            g = sobel_mag(out)[0, 0].cpu().numpy()
            for y, x, _ in patches:
                s_, t_ = slice(y + 8, y + 40), slice(x + 8, x + 40)
                ahes[a].append(float(g[s_, t_].mean() / (g_hr[s_, t_].mean() + 1e-8)))
                avrs[a].append(float(out[0, 0, s_, t_].std().item() / (hr_np[s_, t_].std() + 1e-8)))

    res = {str(b): {"psnr": float(np.mean(ps[b])), "ssim": float(np.mean(ss[b])),
                    "he": float(np.mean(hes[b])), "vr": float(np.mean(vrs[b]))} for b in betas}
    base_psnr = res[str(betas[0])]["psnr"]
    feas = [b for b in betas if base_psnr - res[str(b)]["psnr"] <= args.tol]
    best = min(feas, key=lambda b: res[str(b)]["he"]) if feas else 0.0
    uniform = {str(a): {"psnr": float(np.mean(aps[a])), "ssim": float(np.mean(ass[a])),
                        "he": float(np.mean(ahes[a])), "vr": float(np.mean(avrs[a]))} for a in alphas}
    out = {"betas": res, "uniform": uniform, "beta_star": best, "psnr_base": base_psnr, "tol": args.tol}
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "gate_sweep.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
