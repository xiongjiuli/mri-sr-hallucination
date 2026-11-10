#!/usr/bin/env python3
"""Reproduce the manuscript's archived HE and controlled PDRE ablations.

Run from the repository root with `python3 -m src.paired_ablation`.
Only the committed COP JSON and NPZ archives are needed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import astuple, dataclass
import json
from math import ceil
from pathlib import Path
import random
from statistics import mean, median

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "rrdb_l1"
CONTRASTS = (
    ("Perc vs. L1", "rrdb_perc", BASELINE),
    ("MKDA vs. L1", "rrdb_l1_mkda", BASELINE),
    ("DSC+MKDA vs. L1", "rrdb_dsc_mkda", BASELINE),
    ("DSC+MKDA vs. MKDA", "rrdb_dsc_mkda", "rrdb_l1_mkda"),
    ("Shrink vs. L1", "rrdb_l1_s90", BASELINE),
)
BOOTSTRAP_SEED = 2026
BOOTSTRAP_DRAWS = 10_000
OUTER_SIZE = 48
CORE_SIZE = 32
EPS = 1e-6


@dataclass(frozen=True)
class PairScores:
    he: float
    pdre: float
    amplitude_error: float
    response_mae: float
    original_gradient_error: float
    flat_mae: float


def sobel(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sobel components on the interior pixels, using the one-pixel halo."""
    center_rows = image[:-2] + 2 * image[1:-1] + image[2:]
    center_columns = image[:, :-2] + 2 * image[:, 1:-1] + image[:, 2:]
    return center_rows[:, 2:] - center_rows[:, :-2], center_columns[2:] - center_columns[:-2]


def score_pair(
    original: np.ndarray,
    flat: np.ndarray,
    original_output: np.ndarray,
    flat_output: np.ndarray,
) -> PairScores:
    """Score aligned 2D arrays that include a one-pixel Sobel halo."""
    arrays = tuple(np.asarray(image, dtype=np.float64)
                   for image in (original, flat, original_output, flat_output))
    original, flat, original_output, flat_output = arrays
    if original.ndim != 2 or min(original.shape) < 3 or any(
        image.shape != original.shape for image in arrays
    ):
        raise ValueError("Expected four aligned 2D images with a one-pixel halo")

    core = np.s_[1:-1, 1:-1]
    reference = original[core] - flat[core]
    response = original_output[core] - flat_output[core]
    detail_norm = np.abs(reference).sum()
    error = np.abs(response - reference)
    ref_x, ref_y = sobel(original)
    flat_x, flat_y = sobel(flat_output)
    orig_x, orig_y = sobel(original_output)
    # Match the stabilised Sobel magnitude used in src.metrics and src.probes.
    he_ref = np.sqrt(ref_x**2 + ref_y**2 + 1e-8).mean()
    original_gradient_error = (
        np.hypot(orig_x - ref_x, orig_y - ref_y).sum()
        / (np.hypot(ref_x, ref_y).sum() + EPS)
    )
    return PairScores(
        he=float(np.sqrt(flat_x**2 + flat_y**2 + 1e-8).mean() / (he_ref + 1e-8)),
        pdre=float(error.sum() / (detail_norm + EPS)),
        amplitude_error=float(abs(np.abs(response).sum() - detail_norm) / (detail_norm + EPS)),
        response_mae=float(error.mean()),
        original_gradient_error=float(original_gradient_error),
        flat_mae=float(np.abs(flat_output[core] - flat[core]).mean()),
    )


def load_patch_rows(results: Path, tag: str) -> dict:
    path = results / f"cop_{tag}.json"
    rows = {}
    for row in json.loads(path.read_text(encoding="utf-8"))["patches"]:
        key = (row["subject"], row["slice"], row["y"], row["x"])
        if key in rows:
            raise ValueError(f"Duplicate patch key in {path.name}: {key}")
        rows[key] = row
    if not rows:
        raise ValueError(f"No patch records in {path.name}")
    return rows


def subject_reanalysis(results: Path, baseline: dict):
    """Pair archives by patch and bootstrap equal-weight subject differences."""
    by_tag = {BASELINE: baseline}
    comparison_tags = {tag for _, first, second in CONTRASTS for tag in (first, second)}
    for tag in sorted(comparison_tags - {BASELINE}):
        rows = load_patch_rows(results, tag)
        if rows.keys() != baseline.keys():
            raise ValueError(f"Patch identifiers in {tag} do not match {BASELINE}")
        by_tag[tag] = rows

    by_subject = defaultdict(list)
    for key in sorted(baseline):
        by_subject[key[0]].append(key)
    keys_by_subject = list(by_subject.values())
    n_subjects = len(keys_by_subject)
    rng = random.Random(BOOTSTRAP_SEED)
    draws = [rng.choices(range(n_subjects), k=n_subjects) for _ in range(BOOTSTRAP_DRAWS)]
    lower_index = ceil(0.025 * BOOTSTRAP_DRAWS) - 1
    upper_index = ceil(0.975 * BOOTSTRAP_DRAWS) - 1

    comparisons = []
    for label, first, second in CONTRASTS:
        differences = [
            mean(by_tag[first][key]["he"] - by_tag[second][key]["he"] for key in keys)
            for keys in keys_by_subject
        ]
        boot = sorted(mean(differences[i] for i in draw) for draw in draws)
        comparisons.append((
            label, mean(differences), boot[lower_index], boot[upper_index],
            sum(d > 0 for d in differences),
        ))
    return comparisons, [len(keys) for keys in keys_by_subject]


def controlled_ablation(results: Path, baseline: dict):
    """Construct known outputs on the archived references, without inference."""
    border = (OUTER_SIZE - CORE_SIZE) // 2
    scores = defaultdict(list)
    contrast_checks = defaultdict(list)
    preview = None
    sample_indices = []
    with np.load(results / f"cop_examples_{BASELINE}.npz", allow_pickle=False) as archive:
        indices = sorted(int(key.split("_")[0]) for key in archive.files if key.endswith("_hr"))
        for index in indices:
            reference_slice = archive[f"{index}_hr"]
            flat_slice = archive[f"{index}_occ"]
            for patch in baseline.values():
                if patch["index"] != index:
                    continue
                y, x = patch["y"] + border - 1, patch["x"] + border - 1
                size = CORE_SIZE + 2
                original = reference_slice[y:y + size, x:x + size].astype(np.float64)
                flat = flat_slice[y:y + size, x:x + size].astype(np.float64)
                if original.shape != (size, size) or not np.all(flat == flat[0, 0]):
                    raise ValueError(f"Missing flat interior and halo for archived slice {index}")
                detail = original - flat
                shifted = reference_slice[y:y + size, x - 2:x + size - 2].astype(np.float64)
                texture = 0.01 * np.cos(np.pi * np.arange(border - 1, border - 1 + size)[None, :] / 2)
                pairs = {
                    "Ideal": (original, flat),
                    "Both outputs flat": (flat, flat),
                    "Half-amplitude response": (flat + 0.5 * detail, flat),
                    "Response displaced by two pixels": (shifted, flat),
                    "Contrast reversed and clipped": (np.clip(flat - detail, 0, 1), flat),
                    "Texture on flat output only": (original, flat + texture),
                    "Texture shared by both outputs": (original + texture, flat + texture),
                    "Offset shared by both outputs": (original - 0.02, flat - 0.02),
                }
                for name, (pred, flat_pred) in pairs.items():
                    scores[name].append(score_pair(original, flat, pred, flat_pred))
                for contrast in (1.0, 0.5):
                    scaled_reference = flat + contrast * detail
                    attenuated = flat + 0.5 * contrast * detail
                    contrast_checks[contrast].append(
                        score_pair(scaled_reference, flat, attenuated, flat)
                    )
                if preview is None:
                    preview = pairs
                sample_indices.append(index)
    if not sample_indices:
        raise ValueError("No matching archived reference pairs")
    return scores, contrast_checks, sample_indices, preview


def write_tex(path, comparisons, patch_counts, scores, contrast_checks, indices):
    n_subjects = len(patch_counts)
    avg = {}
    for name, values in scores.items():
        columns = zip(*(astuple(row) for row in values))
        avg[name] = PairScores(*(mean(column) for column in columns))

    macros = {
        "ArchivePatches": sum(patch_counts),
        "ArchiveSubjects": n_subjects,
        "ArchiveMinPatches": min(patch_counts),
        "ArchiveMedianPatches": f"{median(patch_counts):g}",
        "ArchiveMaxPatches": max(patch_counts),
        "ArchiveBootstrapDraws": f"{BOOTSTRAP_DRAWS:,}".replace(",", "{,}"),
        "ArchiveBootstrapSeed": BOOTSTRAP_SEED,
        "ArchiveDSCIncrement": f"{comparisons[3][1]*1000:.2f}",
        "ArchiveShrinkDecrease": f"{-comparisons[4][1]*1000:.2f}",
        "ControlRegions": len(indices),
        "ControlIndices": ", ".join(map(str, sorted(set(indices)))),
        "ControlPreviewIndex": indices[0],
        "ControlShiftAmplitude": f"{avg['Response displaced by two pixels'].amplitude_error:.3f}",
        "ControlShiftPDRE": f"{avg['Response displaced by two pixels'].pdre:.3f}",
        "ControlIdealHE": f"{avg['Ideal'].he:.5f}",
        "ControlSharedHE": f"{avg['Texture shared by both outputs'].he:.5f}",
        "ControlContrastFullMAE": f"{mean(x.response_mae for x in contrast_checks[1.0]):.5f}",
        "ControlContrastHalfMAE": f"{mean(x.response_mae for x in contrast_checks[0.5]):.5f}",
        "ControlContrastPDRE": f"{mean(x.pdre for x in contrast_checks[1.0]):.3f}",
    }
    lines = ["% Generated by python3 -m src.paired_ablation. Do not edit."]
    lines.extend(fr"\newcommand{{\{key}}}{{{value}}}" for key, value in macros.items())

    lines.append(r"\newcommand{\ArchiveComparisonRows}{%")
    for label, difference, lower, upper, higher in comparisons:
        lines.append(
            fr"{label} & ${difference*1000:+.2f}$ & $[{lower*1000:.2f},{upper*1000:.2f}]$ "
            fr"& {higher}/{n_subjects} \\"
        )
    lines.append("}")

    lines.append(r"\newcommand{\ControlledAblationRows}{%")
    for label, row in avg.items():
        lines.append(
            fr"{label} & {row.he:.5f} & {row.pdre:.3f} & {row.amplitude_error:.3f} "
            fr"& {row.response_mae*1000:.2f} & {row.original_gradient_error:.3f} "
            fr"& {row.flat_mae*1000:.2f} \\"
        )
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(path: Path, pairs: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    original, flat = pairs["Ideal"]
    flat_level = float(flat[0, 0])
    core = np.s_[1:-1, 1:-1]
    delta_limit = float(np.max(np.abs((original - flat)[core])))
    cases = (
        ("Ideal pair", "Ideal"),
        ("Erased detail", "Both outputs flat"),
        ("Displaced detail", "Response displaced by two pixels"),
        ("Shared texture", "Texture shared by both outputs"),
    )
    fig, axes = plt.subplots(3, 4, figsize=(6.9, 4.62))
    scales = [(0, 1), (flat_level - 0.015, flat_level + 0.015), (-delta_limit, delta_limit)]
    for column, (title, key) in enumerate(cases):
        original_output, flat_output = pairs[key]
        axes[0, column].set_title(title, fontsize=10, fontweight="semibold", pad=5)
        images = (original_output, flat_output, original_output - flat_output)
        for row, (data, (low, high)) in enumerate(zip(images, scales)):
            ax = axes[row, column]
            panel = ax.imshow(data[core], cmap="RdBu_r" if row == 2 else "gray",
                              vmin=low, vmax=high, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(".75")
                spine.set_linewidth(0.4)
            if column == 3:
                bar = fig.colorbar(panel, cax=ax.inset_axes((1.08, 0.03, 0.09, 0.94)),
                                   ticks=[low, high], format="%.2f")
                bar.ax.tick_params(labelsize=6.5, length=2, pad=2)
                bar.outline.set_linewidth(0.4)
    for row, title in enumerate(("Original\nbranch", "Flat\nbranch", "Original\nminus flat")):
        axes[row, 0].set_ylabel(title, fontsize=9, rotation=0, ha="right", va="center", labelpad=8)
    fig.subplots_adjust(left=0.135, right=0.893, top=0.935, bottom=0.04, hspace=0.10, wspace=0.14)
    fig.savefig(path, metadata={
        "Title": "Controlled paired outputs from an archived IXI region", "CreationDate": None,
    })
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "paper",
                        help="directory for the generated LaTeX and figures")
    args = parser.parse_args()
    results = ROOT / "results"
    baseline = load_patch_rows(results, BASELINE)
    comparisons, patch_counts = subject_reanalysis(results, baseline)
    scores, contrast_checks, indices, preview = controlled_ablation(results, baseline)
    figures = args.out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    write_tex(args.out_dir / "paired_ablation.tex", comparisons, patch_counts,
              scores, contrast_checks, indices)
    write_figure(figures / "fig_paired_controls.pdf", preview)
    print(f"Paired {sum(patch_counts)} patches across {len(patch_counts)} subjects. "
          f"Scored {len(indices)} controlled regions. Wrote assets to {args.out_dir}.")


if __name__ == "__main__":
    main()
