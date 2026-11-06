#!/usr/bin/env python3
"""Build subject-level train/val/test slice stores from the IXI T1w archive.

Preprocessing per volume
------------------------
1. load the volume in canonical (RAS) orientation;
2. brain extraction with deepbet (T1w U-Net, bundled weights);
3. intensity normalisation with the 1st/99th brain percentiles;
4. keep axial slices covering the brain (>=10 % of maximal brain area);
5. resample each axial slice to 1 mm isotropic and pad to 256x256;
6. zero the background and store float16 slices.

The split is by subject, stratified by acquisition site, so no subject
(and therefore no slice) can appear in two splits.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "IXI-T1")
OUT = os.path.join(ROOT, "data", "processed")
SIZE = 256
N_SUBJECTS = 72
SEED = 2026


def normalize(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = vol[mask]
    lo, hi = np.percentile(v, 1), np.percentile(v, 99)
    vol = (vol - lo) / max(hi - lo, 1e-6)
    return np.clip(vol, 0, 1)


def to_1mm_pad(sl: np.ndarray, zooms, order: int = 1, size: int = SIZE) -> np.ndarray:
    """Resample an axial slice to 1 mm isotropic and centre-pad to size x size."""
    fz = (zooms[0] / 1.0, zooms[1] / 1.0)
    out = zoom(sl, fz, order=order) if (abs(fz[0] - 1) > 1e-3 or abs(fz[1] - 1) > 1e-3) else sl
    h, w = out.shape
    if h > size or w > size:
        y0 = max(0, (h - size) // 2); x0 = max(0, (w - size) // 2)
        out = out[y0:y0 + size, x0:x0 + size]
        h, w = out.shape
    py, px = size - h, size - w
    return np.pad(out, ((py // 2, py - py // 2), (px // 2, px - px // 2)), mode="constant")


def process_subject(path: str, bet) -> tuple[np.ndarray, list[int]] | None:
    img = nib.load(path)
    can = nib.as_closest_canonical(img)
    vol = can.get_fdata(dtype=np.float32)
    if vol.ndim != 3:
        return None
    mask = bet.run_model(vol.copy()) > 0.5
    if mask.sum() < 5e5:
        print("  small mask", path, mask.sum(), flush=True)
    vol = normalize(vol, mask)
    vol[~mask] = 0.0
    areas = mask.sum(axis=(0, 1))
    thr = 0.10 * areas.max()
    idx = np.where(areas >= thr)[0]
    if len(idx) == 0:
        return None
    lo, hi = idx.min(), idx.max()
    idx = np.arange(lo, hi + 1)
    zooms = can.header.get_zooms()[:2]
    slices = np.stack([to_1mm_pad(vol[:, :, z], zooms, 1) for z in idx]).astype(np.float16)
    return slices, idx.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n", type=int, default=N_SUBJECTS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.raw, "*.nii.gz")))
    if not files:
        raise SystemExit(f"no NIfTI files under {args.raw}")
    print(f"found {len(files)} volumes")

    rng = random.Random(args.seed)
    rng.shuffle(files)
    files = sorted(files[: args.n])
    site = lambda f: os.path.basename(f).split("-")[1] if "-" in os.path.basename(f) else "NA"
    groups: dict[str, list[str]] = {}
    for f in files:
        groups.setdefault(site(f), []).append(f)
    train, val, test = [], [], []
    for s, fs in sorted(groups.items()):
        rng.shuffle(fs)
        n = len(fs)
        n_test = max(3, round(0.22 * n))
        n_val = max(2, round(0.14 * n))
        test += fs[:n_test]
        val += fs[n_test:n_test + n_val]
        train += fs[n_test + n_val:]
        print(s, n, "->", len(fs[n_test + n_val:]), len(fs[n_test:n_test + n_val]), n_test)

    from deepbet import BrainExtraction
    bet = BrainExtraction(no_gpu=False)

    os.makedirs(args.out, exist_ok=True)
    splits = {"train": train, "val": val, "test": test}
    meta = {}
    for split, fs in splits.items():
        arrs, rows = [], []
        for f in fs:
            res = process_subject(f, bet)
            if res is None:
                continue
            sl, idx = res
            arrs.append(sl)
            sid = os.path.basename(f).replace("-T1.nii.gz", "")
            rows += [{"subject": sid, "slice": int(i)} for i in idx]
            print(split, sid, sl.shape[0], flush=True)
        arr = np.concatenate(arrs, 0)
        np.save(os.path.join(args.out, f"{split}.npy"), arr)
        meta[split] = rows
        print(split, arr.shape, f"{arr.nbytes/1e6:.0f} MB")
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    with open(os.path.join(args.out, "splits.json"), "w") as f:
        json.dump({k: [os.path.basename(x) for x in v] for k, v in splits.items()}, f, indent=1)


if __name__ == "__main__":
    main()
