#!/usr/bin/env python3
"""Counterfactual occlusion probe (COP) for hallucinated structure.

For each test slice we replace a few textured 48x48 patches with their
local mean in the *reference*, degrade the modified slice, and let the
model reconstruct it.  Any high-frequency energy the model places inside
the (now flat) patch is, by construction, not supported by the input.

Reported per patch (inner 32x32, away from the seam):
  he    : mean |grad SR| / mean |grad original HR|   (hallucinated texture)
  vr    : std(SR) / std(original HR)                  (fabricated variance)
  mae   : |SR - flat reference|                       (fidelity to the probe)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

from .data import EvalDataset
from .degradation import DEGRADATIONS, degrade, degrade_bicubic
from .infer import load_checkpoint, sr_forward
from .metrics import sobel_mag

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pick_patches(hr: np.ndarray, n: int = 4, size: int = 48, rng=None):
    g = ndi.gaussian_filter(hr, 1.0)
    gx = ndi.sobel(g, axis=0); gy = ndi.sobel(g, axis=1)
    gm = np.hypot(gx, gy)
    brain = ndi.binary_erosion(hr > 0.05, iterations=3)
    h, w = hr.shape
    cand = []
    for y in range(8, h - size - 8, 12):
        for x in range(8, w - size - 8, 12):
            if brain[y:y + size, x:x + size].mean() < 0.98:
                continue
            cand.append((y, x, float(gm[y:y + size, x:x + size].mean())))
    if not cand:
        return []
    vals = np.array([c[2] for c in cand])
    thr = np.percentile(vals, 60)
    cand = [c for c in cand if c[2] >= thr]
    rng.shuffle(cand)
    chosen = []
    for y, x, _ in cand:
        if all(abs(y - yy) >= size and abs(x - xx) >= size for yy, xx, _ in chosen):
            chosen.append((y, x, _))
        if len(chosen) == n:
            break
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--patches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gate", type=float, default=0.0)
    ap.add_argument("--deg", default="id")
    ap.add_argument("--shrink", type=float, default=1.0)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _ = load_checkpoint(args.ckpt, device, uncertainty=args.gate > 0)
    ds = EvalDataset(args.split)
    rng = np.random.default_rng(args.seed)
    n = min(args.n, len(ds))
    idx = np.linspace(0, len(ds) - 1, n).astype(int)

    rows = []
    examples = []
    for k, i in enumerate(idx):
        b = ds[int(i)]
        hr = b["hr"][0].numpy()
        patches = pick_patches(hr, args.patches, rng=rng)
        if not patches:
            continue
        occ = hr.copy()
        for y, x, _ in patches:
            occ[y:y + 48, x:x + 48] = hr[y:y + 48, x:x + 48].mean()
        hr_t = torch.from_numpy(hr)[None, None].to(device)
        occ_t = torch.from_numpy(occ)[None, None].to(device)
        if args.deg == "bicubic":
            lr = degrade_bicubic(occ_t, 4)
        else:
            cfg = dict(DEGRADATIONS[args.deg]); sc = cfg.pop("scale", 4)
            lr = degrade(occ_t, scale=sc, **cfg)
        sr, u = sr_forward(model, lr, beta=args.gate)
        if args.shrink < 1.0:
            base = torch.nn.functional.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1)
            sr = (args.shrink * sr + (1 - args.shrink) * base).clamp(0, 1)
        g_sr = sobel_mag(sr)[0, 0].cpu().numpy()
        g_hr = sobel_mag(hr_t)[0, 0].cpu().numpy()
        for y, x, _ in patches:
            s = slice(y + 8, y + 40); t = slice(x + 8, x + 40)
            he = float(g_sr[s, t].mean() / (g_hr[s, t].mean() + 1e-8))
            vr = float(sr[0, 0, s, t].std().item() / (hr[s, t].std() + 1e-8))
            mae = float((sr[0, 0, s, t] - occ_t[0, 0, s, t]).abs().mean().item())
            rows.append({"index": int(i), "subject": b["subject"], "slice": b["slice"],
                         "y": int(y), "x": int(x), "he": he, "vr": vr, "mae": mae,
                         "u": (float(u[0, 0, s, t].mean().item()) if u is not None else None)})
        if len(examples) < 3:
            examples.append({"index": int(i), "hr": hr, "occ": occ,
                             "lr": lr[0, 0].cpu().numpy(), "sr": sr[0, 0].cpu().numpy(),
                             "u": (u[0, 0].cpu().numpy() if u is not None else None),
                             "patches": [(int(y), int(x)) for y, x, _ in patches]})
        if k % 40 == 0:
            print(f"{args.tag} COP {k}/{n} he={rows[-1]['he']:.3f}", flush=True)

    vals = {k: float(np.mean([r[k] for r in rows])) for k in ("he", "vr", "mae")}
    vals["n_patches"] = len(rows)
    vals["n_slices"] = int(len(set(r["index"] for r in rows)))
    if rows[0]["u"] is not None:
        vals["u_mean"] = float(np.mean([r["u"] for r in rows]))
    out = {"tag": args.tag, "summary": vals, "patches": rows}
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, f"cop_{args.tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(os.path.join(args.outdir, f"cop_examples_{args.tag}.npz"),
                        **{f"{e['index']}_hr": e["hr"] for e in examples},
                        **{f"{e['index']}_occ": e["occ"] for e in examples},
                        **{f"{e['index']}_sr": e["sr"] for e in examples},
                        **{f"{e['index']}_lr": e["lr"] for e in examples},
                        **({f"{e['index']}_u": e["u"] for e in examples} if examples[0]["u"] is not None else {}))
    print(json.dumps(vals, indent=1))


if __name__ == "__main__":
    main()
