#!/usr/bin/env python3
"""Bicubic baseline: metrics and hallucination probe (no learned model)."""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch
from scipy import ndimage as ndi

from .data import EvalDataset
from .degradation import DEGRADATIONS, degrade, degrade_bicubic, upsample_bicubic
from .metrics import LPIPS, all_metrics_vec, sobel_mag
from .probes import pick_patches

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_lr(hr, kind):
    if kind == "bicubic":
        return degrade_bicubic(hr)
    cfg = dict(DEGRADATIONS[kind]); scale = cfg.pop("scale", 4)
    return degrade(hr, scale=scale, **cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--degs", default="id,blur1.4,noise,bicubic")
    ap.add_argument("--lpips", action="store_true")
    ap.add_argument("--cop_n", type=int, default=240)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ds = EvalDataset(args.split)
    os.makedirs(args.outdir, exist_ok=True)
    lfn = LPIPS(device) if args.lpips else None

    from torch.utils.data import DataLoader
    rows = []
    for kind in args.degs.split(","):
        dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        for batch in dl:
            hr = batch["hr"].to(device)
            lr = make_lr(hr, kind)
            sr = upsample_bicubic(lr)
            mv = all_metrics_vec(sr, hr)
            for j, i in enumerate(batch["index"].tolist()):
                m = {"model": "bicubic", "deg": kind, "index": int(i),
                     "subject": ds.meta[int(i)]["subject"], "slice": ds.meta[int(i)]["slice"]}
                for k, arr in mv.items():
                    m[k] = float(arr[j])
                rows.append(m)
    if lfn is not None:
        vals = []
        step = max(1, len(ds) // 240)
        for i in range(0, len(ds), step):
            b = ds[i]; hr = b["hr"][None].to(device)
            sr = upsample_bicubic(make_lr(hr, "id"))
            vals.append(lfn(sr, hr).mean().item())
        for r in rows:
            if r["deg"] == "id":
                r["lpips"] = float(np.mean(vals))
    with open(os.path.join(args.outdir, "metrics_bicubic.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary = {}
    for kind in args.degs.split(","):
        sub = [r for r in rows if r["deg"] == kind]
        keys = [k for k in sub[0] if k not in ("model", "deg", "index", "subject", "slice")]
        summary[kind] = {k: float(np.mean([r[k] for r in sub])) for k in keys}
        summary[kind]["n"] = len(sub)
    with open(os.path.join(args.outdir, "summary_bicubic.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))

    # COP control
    rng = np.random.default_rng(7)
    idx = np.linspace(0, len(ds) - 1, min(args.cop_n, len(ds))).astype(int)
    vals = []
    for i in idx:
        b = ds[int(i)]
        hr_np = b["hr"][0].numpy()
        patches = pick_patches(hr_np, 4, rng=rng)
        if not patches:
            continue
        occ = hr_np.copy()
        for y, x, _ in patches:
            occ[y:y + 48, x:x + 48] = hr_np[y:y + 48, x:x + 48].mean()
        occ_t = torch.from_numpy(occ)[None, None].to(device)
        sr = upsample_bicubic(degrade(occ_t, scale=4, sigma=0.8))
        g_sr = sobel_mag(sr)[0, 0].cpu().numpy()
        g_hr = sobel_mag(torch.from_numpy(hr_np)[None, None])[0, 0].numpy()
        for y, x, _ in patches:
            s_ = slice(y + 8, y + 40); t_ = slice(x + 8, x + 40)
            vals.append({"he": float(g_sr[s_, t_].mean() / (g_hr[s_, t_].mean() + 1e-8)),
                         "vr": float(sr[0, 0, s_, t_].std().item() / (hr_np[s_, t_].std() + 1e-8)),
                         "mae": float((sr[0, 0, s_, t_] - occ_t[0, 0, s_, t_]).abs().mean().item())})
    cop = {k: float(np.mean([v[k] for v in vals])) for k in ("he", "vr", "mae")}
    cop["n_patches"] = len(vals)
    with open(os.path.join(args.outdir, "cop_bicubic.json"), "w") as f:
        json.dump({"tag": "bicubic", "summary": cop}, f, indent=1)
    print("COP", cop)


if __name__ == "__main__":
    main()
