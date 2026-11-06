#!/usr/bin/env python3
"""Evaluate one checkpoint on full test slices under several degradations."""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import EvalDataset
from .degradation import DEGRADATIONS, degrade, degrade_bicubic
from .infer import load_checkpoint, sr_forward
from .metrics import LPIPS, all_metrics_vec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_lr(hr, kind):
    if kind == "bicubic":
        return degrade_bicubic(hr)
    cfg = dict(DEGRADATIONS[kind])
    scale = cfg.pop("scale", 4)
    return degrade(hr, scale=scale, **cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--degs", default="id")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ood_limit", type=int, default=600)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lpips", action="store_true")
    ap.add_argument("--lpips_n", type=int, default=240)
    ap.add_argument("--save_preds", default="")
    ap.add_argument("--gate", type=float, default=0.0)
    ap.add_argument("--shrink", type=float, default=1.0)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _ = load_checkpoint(args.ckpt, device, uncertainty=args.gate > 0)
    ds = EvalDataset(args.split)
    n = len(ds) if not args.limit else min(args.limit, len(ds))
    lfn = LPIPS(device) if args.lpips else None
    pred_idx = [int(x) for x in args.save_preds.split(",") if x.strip()]

    os.makedirs(args.outdir, exist_ok=True)
    rows = []
    lpips_vals = []
    for kind in args.degs.split(","):
        n_kind = n if (kind == "id" or args.limit) else min(args.ood_limit, n)
        sub = torch.utils.data.Subset(ds, list(range(n_kind)))
        dl = DataLoader(sub, batch_size=args.batch, shuffle=False, num_workers=0)
        for batch in dl:
            hr = batch["hr"].to(device)
            lr = make_lr(hr, kind).to(device)
            sr, u = sr_forward(model, lr, beta=args.gate)
            if args.shrink < 1.0:
                base = torch.nn.functional.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1)
                sr = (args.shrink * sr + (1 - args.shrink) * base).clamp(0, 1)
            mv = all_metrics_vec(sr, hr)
            if lfn is not None and kind == "id":
                lpips_vals.extend(lfn(sr, hr).cpu().numpy().tolist())
            idxs = batch["index"].tolist()
            for j, i in enumerate(idxs):
                meta = ds.meta[int(i)]
                r = {"model": args.tag, "deg": kind, "index": int(i),
                     "subject": meta["subject"], "slice": meta["slice"]}
                for k, arr in mv.items():
                    r[k] = float(arr[j])
                if u is not None:
                    err = (sr[j:j + 1] - hr[j:j + 1]).abs()
                    r["u_mean"] = float(u[j].mean())
                    r["err_mean"] = float(err.mean())
                    uu = u[j].flatten().float(); ee = err.flatten().float()
                    ru = uu.argsort().argsort().float(); re = ee.argsort().argsort().float()
                    ru = (ru - ru.mean()) / (ru.std() + 1e-8)
                    re = (re - re.mean()) / (re.std() + 1e-8)
                    r["u_err_spearman"] = float((ru * re).mean())
                rows.append(r)
                if int(i) in pred_idx:
                    d = os.path.join(args.outdir, "preds")
                    os.makedirs(d, exist_ok=True)
                    np.savez_compressed(
                        os.path.join(d, f"{args.tag}_{kind}_{int(i):04d}.npz"),
                        sr=sr[j, 0].float().cpu().numpy(),
                        hr=hr[j, 0].float().cpu().numpy(),
                        lr=lr[j, 0].float().cpu().numpy(),
                        u=(u[j, 0].float().cpu().numpy() if u is not None else np.zeros((1,))))
        print(f"{args.tag} {kind} done ({n_kind})", flush=True)

    if lpips_vals:
        lp = float(np.mean(lpips_vals))
        for r in rows:
            if r["deg"] == "id":
                r["lpips"] = lp
        print(f"LPIPS({len(lpips_vals)}) = {lp:.4f}")

    csv_path = os.path.join(args.outdir, f"metrics_{args.tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    summary = {}
    for kind in args.degs.split(","):
        s = [r for r in rows if r["deg"] == kind]
        keys = [k for k in s[0] if k not in ("model", "deg", "index", "subject", "slice")]
        summary[kind] = {k: {"mean": float(np.mean([r[k] for r in s])),
                             "std": float(np.std([r[k] for r in s]))} for k in keys}
        summary[kind]["n"] = len(s)
    with open(os.path.join(args.outdir, f"summary_{args.tag}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: {m: round(v["mean"], 4) for m, v in s.items() if isinstance(v, dict)}
                      for k, s in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
