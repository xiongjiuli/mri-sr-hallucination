# Measuring Hallucination in MRI Super-Resolution

Code, data splits and experiment logs for the course project
*AI-assisted research: a controlled probe for hallucinated structure in MRI
super-resolution, and an empirical study of fidelity, sharpness and
fabrication*.

The repository contains everything needed to reproduce the main results:
subject-level data splits, preprocessing, four trained models, the
counterfactual occlusion probe (COP), all evaluation scripts and the
scripts that regenerate every number and figure in the paper.

---

## 1. What is in the box

| Path | Content |
|---|---|
| `src/prepare_ixi.py` | IXI download post-processing: deepbet brain extraction, 1 mm resampling, axial slicing, subject-level split |
| `src/models.py` | SRResNet, RRDB, uncertainty head, gated wrapper |
| `src/degradation.py` | blur + average-pool degradation, bicubic degradation, OOD variants |
| `src/train.py` | training loop (L1 / L1+DSC / L1+perceptual) |
| `src/train_unc.py` | post-hoc error head (used for the gate control) |
| `src/evaluate.py` | full-slice metrics, shrinkage, several degradations |
| `src/probes.py` | counterfactual occlusion probe |
| `src/baselines.py` | bicubic baseline |
| `src/gate_sweep.py` | validation sweep for the gate strength β |
| `src/make_numbers.py` | generates `paper/numbers.tex` from the results |
| `src/figures.py` | generates all paper figures |
| `scripts/download_ixi.py` | parallel range download of IXI-T1.tar (4.84 GB) |
| `scripts/train_all.sh` | trains the four models sequentially and the error head |
| `configs/` | resolved training configurations |
| `data/processed/` | slices and subject-level splits (generated) |
| `checkpoints/` | trained weights (generated) |
| `results/` | metrics, COP JSON, predictions and figures |
| `paper/` | IEEEtran manuscript and bibliography |

## 2. Environment

Tested with Python 3.12.7, PyTorch 2.12, torchvision 0.27 on an Apple M3 Pro
(MPS). The code also runs on CPU (slower) and on CUDA by changing the device
selection.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the packages that were used. `nibabel` and
`torchvision` are only needed for preprocessing, LPIPS and the perceptual
loss. If you install into a local directory instead of a venv, set
`PYTHONPATH` accordingly (the supplied scripts assume `./.pylibs` when it
exists, see `scripts/run_all.sh`).

## 3. Data

The public IXI T1w archive is downloaded from the original host:

```bash
python3 scripts/download_ixi.py          # 4.84 GB, ~7-10 min
mkdir -p data/IXI-T1
tar -xf data/IXI-T1.tar -C data/IXI-T1
python3 -m src.prepare_ixi --n 72
```

Preprocessing uses `deepbet` for brain extraction and stores 256×256
float16 axial slices at 1 mm isotropic resolution. The split is by subject
and stratified by acquisition site (43 train / 11 validation / 18 test).
`data/processed/splits.json` lists the exact subjects; the split is fixed
by `--seed 2026` and is included in the repository so that the experiment
can be reproduced without re-downloading the archive.

**Leakage control.** No subject appears in two splits; slices are never
split randomly. This is checked automatically in `src/make_numbers.py`
and by the assert in `scripts/run_all.sh`.

## 4. Training

```bash
bash scripts/train_all.sh
```

This trains, for 5,000 iterations each with identical budgets:

| tag | objective |
|---|---|
| `rrdb_l1` | L1, σ = 0.8 (baseline) |
| `rrdb_l1_mkda` | L1, σ ~ U{0.5,0.8,1.1,1.4} (MKDA ablation) |
| `rrdb_dsc` | L1 + 0.5·DSC, σ = 0.8 |
| `rrdb_dsc_mkda` | L1 + 0.5·DSC, MKDA |
| `rrdb_perc` | L1 + 0.02·AlexNet perceptual |
| `srresnet_l1` | SRResNet-16, L1 |

then the post-hoc error head:

```bash
python3 -m src.train_unc --ckpt checkpoints/rrdb_dsc_mkda.pt --tag rrdb_dsc_mkda
```

Each run writes `logs/<tag>.jsonl` (loss, learning rate, validation PSNR)
and `checkpoints/<tag>.pt` (best validation PSNR). The resolved
configuration is stored inside the checkpoint.

## 5. Evaluation

```bash
# in-distribution + three mismatched degradations, with LPIPS
python3 -m src.evaluate --ckpt checkpoints/rrdb_l1.pt --tag rrdb_l1 \
    --degs id,blur1.4,noise,bicubic --ood_limit 600 --lpips --save_preds 120,420,760

# risk control: shrink the SR output toward bicubic (alpha = 0.9)
python3 -m src.evaluate --ckpt checkpoints/rrdb_l1.pt --tag rrdb_l1_s90 \
    --degs id --shrink 0.9 --lpips --save_preds 120,420,760

# hallucination probe
python3 -m src.probes --ckpt checkpoints/rrdb_l1.pt --tag rrdb_l1 --n 240

# bicubic control
python3 -m src.baselines --lpips

# gate sweep (validation only); used as a negative control against shrinkage
python3 -m src.gate_sweep --ckpt checkpoints/rrdb_dsc_mkda_unc.pt
```

Then regenerate the paper numbers and figures:

```bash
python3 -m src.make_numbers
python3 -m src.figures
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## 6. Results at a glance

See `results/summary_*.json`, `results/cop_*.json`, `results/figures/` and
`paper/main.pdf`. The main findings are:

1. pixel-loss SR is conservative: it improves PSNR over bicubic by ~3 dB
   while raising COP hallucination from 0.002 to 0.022;
2. perceptual training improves LPIPS but increases COP hallucination by
   about 18%;
3. multi-kernel degradation training improves out-of-distribution PSNR by
   ~0.5 dB but also increases hallucination, and adding DSC does not help;
4. shrinking the output toward bicubic lowers hallucination by ~10% for a
   0.03 dB PSNR cost, and a learned error gate is no better than this
   uniform shrinkage.

## 7. Reproducibility notes

- All random seeds are fixed (`--seed 0` for training, `2026` for the split).
- The split file is versioned; do not regenerate it with a different seed.
- Checkpoints are selected on validation PSNR only; the test split is
  evaluated once.
- Metrics are computed per slice and aggregated; `results/metrics_*.csv`
  contains the per-slice values so that all statistics can be recomputed.
- Hardware: one Apple M3 Pro (14-core GPU). Training the four models takes
  about 1.5 h; full evaluation takes about 30 min.

## 8. AI-assisted workflow

This project was developed with an AI coding assistant (DeepSeek Harness).
The assistant wrote the initial code, ran the experiments and drafted the
manuscript. All numbers in the paper are generated from the result files by
`src/make_numbers.py`; the group is responsible for checking the code, the
splits, the metrics and the claims. See `docs/AI_USAGE.md` for the detailed
record.
