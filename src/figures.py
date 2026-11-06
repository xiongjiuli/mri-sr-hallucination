#!/usr/bin/env python3
"""Generate all manuscript figures from the result files."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

MODELS = [
    ("bicubic", "Bicubic", "bicubic", "#7f7f7f"),
    ("rrdb_l1", "RRDB", "rrdb_l1", "#1f77b4"),
    ("rrdb_l1_mkda", "RRDB+MKDA", "rrdb_l1_mkda", "#17becf"),
    ("rrdb_perc", "RRDB+Perc", "rrdb_perc", "#d62728"),
    ("rrdb_dsc_mkda", "RRDB+DSC+MKDA", "trust", "#9467bd"),
    ("rrdb_l1_s90", "RRDB+Shrink", "shr", "#8c564b"),
]
LABEL = {m[0]: m[1] for m in MODELS}
COLOR = {m[0]: m[3] for m in MODELS}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=300)


def load_json(path, default=None):
    return json.load(open(path)) if os.path.exists(path) else default


def summary(tag):
    return load_json(os.path.join(RES, f"summary_{tag}.json"))


def cop(tag):
    return load_json(os.path.join(RES, f"cop_{tag}.json"))


def val(s, deg="id", metric="psnr"):
    if not s or deg not in s or metric not in s[deg]:
        return np.nan
    v = s[deg][metric]
    return v["mean"] if isinstance(v, dict) else v


def fig_method():
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
    box = dict(fc="white", ec="0.25", lw=0.8)

    def b(x, y, w, h, text):
        ax.add_patch(Rectangle((x, y), w, h, **box))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5)

    b(0.1, 2.5, 1.7, 0.9, "LR input\n$y$")
    b(2.4, 2.5, 2.0, 0.9, "RRDB + global\nresidual $G_\\theta$")
    b(5.0, 2.5, 1.7, 0.9, "SR estimate\n$\\hat{x}$")
    b(7.3, 2.5, 2.0, 0.9, "gated fusion\n$w\\hat{x}+(1-w)\\mathcal{B}(y)$")
    b(9.9, 2.5, 1.7, 0.9, "output\n$\\hat{x}_g$")

    b(2.4, 0.6, 2.0, 0.8, "degradation $D_\\sigma$\n+ MKDA")
    b(5.0, 0.6, 1.7, 0.8, "DSC loss\n$\\|D_\\sigma(\\hat{x})-y\\|_1$")
    b(7.3, 0.6, 2.0, 0.8, "error head\n$u=H_\\phi(f)$")

    for x0, x1 in [(1.8, 2.4), (4.4, 5.0), (6.7, 7.3), (9.3, 9.9)]:
        ax.annotate("", xy=(x1, 2.95), xytext=(x0, 2.95),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.2"))
    ax.annotate("", xy=(5.85, 1.5), xytext=(5.85, 2.5),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.2", ls=":"))
    ax.annotate("", xy=(3.4, 1.4), xytext=(3.4, 2.5),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.2", ls=":"))
    ax.annotate("", xy=(8.3, 2.5), xytext=(8.3, 1.4),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.2"))
    ax.annotate("", xy=(5.0, 1.0), xytext=(4.4, 1.0),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.2"))
    ax.annotate("", xy=(7.3, 1.0), xytext=(6.7, 1.0),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.2"))
    ax.text(0.1, 3.75, "(a) training and inference", fontsize=8.5, weight="bold")
    fig.tight_layout(pad=0.2)
    save(fig, "fig_method")
    plt.close(fig)


def fig_pareto():
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    handles, labels = [], []
    for tag, label, key, color in MODELS:
        s = summary(tag); c = cop(tag)
        if s is None or c is None:
            continue
        x, y = val(s), c["summary"]["he"]
        if np.isnan(x):
            continue
        h = ax.scatter(x, y, s=36, color=color, edgecolor="k", linewidth=0.4, zorder=3)
        handles.append(h); labels.append(label)
    ax.set_xlim(30.2, 34.1)
    ax.set_ylim(0, 0.034)
    ax.set_xlabel("PSNR (dB)")
    ax.set_ylabel("COP hallucination energy $\\downarrow$")
    ax.grid(alpha=0.25, lw=0.4)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=2, frameon=False, fontsize=6.2, handletextpad=0.3, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    save(fig, "fig_pareto")
    plt.close(fig)


def fig_ood():
    degs = [("blur1.4", "blur 1.4"), ("noise", "noise 0.02"), ("bicubic", "bicubic")]
    tags = [("rrdb_l1", "RRDB"), ("rrdb_l1_mkda", "RRDB+MKDA"),
            ("rrdb_dsc_mkda", "RRDB+DSC+MKDA"), ("rrdb_perc", "RRDB+Perc")]
    base = summary("rrdb_l1")
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    width = 0.8 / len(tags)
    for k, (tag, label) in enumerate(tags):
        s = summary(tag)
        vals = [val(s, d) - val(base, d) for d, _ in degs]
        ax.bar(np.arange(len(degs)) + k * width - 0.4 + width / 2, vals, width,
               label=label, color=COLOR.get(tag, None), edgecolor="k", linewidth=0.3)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(np.arange(len(degs)))
    ax.set_xticklabels([n for _, n in degs])
    ax.set_ylabel("$\\Delta$PSNR vs RRDB (dB)")
    ax.legend(frameon=False, fontsize=6.0, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.22))
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, "fig_ood")
    plt.close(fig)


def crop(img, cy, cx, size=160):
    h, w = img.shape
    y0 = int(np.clip(cy - size // 2, 0, h - size))
    x0 = int(np.clip(cx - size // 2, 0, w - size))
    return img[y0:y0 + size, x0:x0 + size]


def fig_qual():
    idxs = [int(x) for x in os.environ.get("QUAL_IDX", "1080,1560").split(",")]
    tags = [("rrdb_l1", "RRDB"), ("rrdb_perc", "RRDB+Perc"),
            ("rrdb_dsc_mkda", "DSC+MKDA"), ("rrdb_l1_s90", "Shrink")]
    rows = []
    for idx in idxs:
        d = {}
        for tag, _ in tags:
            p = os.path.join(RES, "preds", f"{tag}_id_{idx:04d}.npz")
            if os.path.exists(p):
                d[tag] = np.load(p)
        if d:
            rows.append((idx, d))
    if not rows:
        return
    ncol = 2 + len(tags) + 2
    fig, axes = plt.subplots(len(rows), ncol, figsize=(1.55 * ncol, 1.6 * len(rows)))
    if len(rows) == 1:
        axes = axes[None]
    for r, (idx, d) in enumerate(rows):
        first = next(iter(d.values()))
        hr = first["hr"]
        ys, xs = np.where(hr > 0.08)
        cy, cx = int(ys.mean()), int(xs.mean())
        import torch, torch.nn.functional as F
        up = F.interpolate(torch.from_numpy(first["lr"])[None, None], scale_factor=4,
                           mode="bicubic", align_corners=False)[0, 0].numpy().clip(0, 1)
        imgs = [("LR", crop(up, cy, cx, 144))]
        for tag, label in tags:
            if tag in d:
                imgs.append((label, crop(d[tag]["sr"], cy, cx, 144)))
            else:
                imgs.append((label, np.zeros((144, 144))))
        imgs.append(("HR", crop(hr, cy, cx, 144)))
        if "rrdb_l1_s90" in d:
            err = crop(np.abs(d["rrdb_l1_s90"]["sr"] - hr), cy, cx, 144)
        elif "rrdb_dsc_mkda" in d:
            err = crop(np.abs(d["rrdb_dsc_mkda"]["sr"] - hr), cy, cx, 144)
        else:
            err = np.zeros((144, 144))
        imgs.append(("$|e|$ perc", crop(np.abs(d["rrdb_perc"]["sr"] - hr), cy, cx, 144) if "rrdb_perc" in d else err))
        imgs.append(("$|e|$ shrink", err))
        for c, (lab, im) in enumerate(imgs):
            ax = axes[r, c]
            if im.max() == 0 and lab not in ("LR",):
                ax.axis("off")
                continue
            if lab.startswith("$|e|$"):
                ax.imshow(im, cmap="magma", vmin=0, vmax=0.05)
            else:
                ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(lab, fontsize=7)
        axes[r, 0].set_ylabel(f"slice {idx}", fontsize=7)
    fig.tight_layout(pad=0.05, w_pad=0.05, h_pad=0.05)
    save(fig, "fig_qual")
    plt.close(fig)


def fig_cop():
    idx = int(os.environ.get("COP_IDX", "0"))
    files = {tag: os.path.join(RES, f"cop_examples_{tag}.npz") for tag, _, _, _ in MODELS}
    avail = {t: np.load(p) for t, p in files.items() if os.path.exists(p)}
    if not avail:
        return
    keys = [k for k in avail["rrdb_l1"].files if k.endswith("_hr")]
    if not keys:
        return
    key = keys[min(idx, len(keys) - 1)]
    i = key.split("_")[0]
    hr = avail["rrdb_l1"][f"{i}_hr"]
    occ = avail["rrdb_l1"][f"{i}_occ"]
    patches = None
    # locate patches from the COP json
    cj = cop("rrdb_l1")
    if cj:
        ps = [r for r in cj["patches"] if str(r["index"]) == i]
        if ps:
            patches = [(r["y"], r["x"]) for r in ps]
    if patches:
        y, x = patches[0]
        cy, cx = y + 24, x + 24
    else:
        cy, cx = hr.shape[0] // 2, hr.shape[1] // 2
    panels = [("HR", hr), ("occluded HR", occ)]
    for tag, label in [("rrdb_l1", "RRDB"), ("rrdb_perc", "RRDB+Perc"),
                       ("rrdb_dsc_mkda", "DSC+MKDA"), ("rrdb_l1_s90", "Shrink")]:
        if tag in avail:
            panels.append((label, avail[tag][f"{i}_sr"]))
            if f"{i}_u" in avail[tag].files and tag == "rrdb_dsc_mkda":
                panels.append(("uncertainty $u$", avail[tag][f"{i}_u"]))
    fig, axes = plt.subplots(1, len(panels), figsize=(1.7 * len(panels), 1.9))
    for ax, (lab, im) in zip(axes, panels):
        cmap = "magma" if "uncertainty" in lab else "gray"
        vmin, vmax = (0, 1) if "uncertainty" not in lab else (None, None)
        ax.imshow(crop(im, cy, cx, 128), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(lab, fontsize=7); ax.set_xticks([]); ax.set_yticks([])
        if patches:
            rect = Rectangle((x - (cx - 64) + 8, y - (cy - 64) + 8), 32, 32,
                             fill=False, ec="lime", lw=0.8)
            ax.add_patch(rect)
    fig.tight_layout(pad=0.05, w_pad=0.05)
    save(fig, "fig_cop")
    plt.close(fig)


def fig_failure():
    import csv
    rows = []
    p = os.path.join(RES, "metrics_rrdb_l1_s90.csv")
    if os.path.exists(p):
        with open(p) as f:
            rows = [r for r in csv.DictReader(f) if r["deg"] == "id"]
    rows = [r for r in rows if 600 <= int(r["index"]) <= 2100
            and os.path.exists(os.path.join(RES, "preds", f"rrdb_l1_s90_id_{int(r['index']):04d}.npz"))]
    if not rows:
        return
    worst = sorted(rows, key=lambda r: float(r["psnr"]))[:2]
    fig, axes = plt.subplots(len(worst), 4, figsize=(7.0, 1.9 * len(worst)))
    if len(worst) == 1:
        axes = axes[None]
    for r, row in enumerate(worst):
        i = int(row["index"])
        d = np.load(os.path.join(RES, "preds", f"rrdb_l1_s90_id_{i:04d}.npz"))
        hr, sr = d["hr"], d["sr"]
        err = np.abs(sr - hr)
        axes[r, 0].imshow(hr, cmap="gray", vmin=0, vmax=1)
        axes[r, 1].imshow(sr, cmap="gray", vmin=0, vmax=1)
        axes[r, 2].imshow(err, cmap="magma", vmax=0.05)
        axes[r, 3].imshow(np.abs(np.gradient(sr)[0]), cmap="magma")
        axes[r, 0].set_ylabel(f"slice {i}", fontsize=7)
        for a, t in zip(axes[r], ["HR", "Shrink", "$|e|$", "$|\\nabla\\hat{x}|$"]):
            a.set_title(t, fontsize=7)
            a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(pad=0.1)
    save(fig, "fig_failure")
    plt.close(fig)


def main():
    fig_method()
    fig_pareto()
    fig_ood()
    fig_qual()
    fig_cop()
    fig_failure()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
