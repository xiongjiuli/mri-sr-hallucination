# Processed data

This directory contains the **subject lists and slice metadata** used by the
experiments:

- `splits.json` — the exact 43/11/18 subject-level split (fixed seed 2026);
- `meta.json` — one record per slice (`subject`, axial slice index).

The actual slice arrays (`train.npy`, `val.npy`, `test.npy`, about 1.2 GB)
are **not** committed because of their size and because they are fully
reproducible:

```bash
python3 scripts/download_ixi.py
mkdir -p data/IXI-T1 && tar -xf data/IXI-T1.tar -C data/IXI-T1
python3 -m src.prepare_ixi --n 72
```

The preprocessing script uses deepbet for brain extraction, resamples each
axial slice to 1 mm isotropic and pads to 256×256. The split file is
versioned so that the reported test set can be recovered exactly.
