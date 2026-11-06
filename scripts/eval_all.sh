#!/usr/bin/env bash
# Evaluate every trained model and regenerate the paper artefacts.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/.pylibs:$PWD"
export TORCH_HOME="$PWD/.cache/torch"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export MPLCONFIGDIR="$PWD/.cache/mpl"

PRED_IDX=120,360,600,840,1080,1320,1560,1800,2040,2280

python3 -m src.baselines --lpips

for M in rrdb_l1 rrdb_l1_mkda rrdb_perc rrdb_dsc_mkda; do
  python3 -m src.evaluate --ckpt checkpoints/$M.pt --tag $M \
    --degs id,blur1.4,noise,bicubic --ood_limit 600 --lpips --save_preds $PRED_IDX
  python3 -m src.probes --ckpt checkpoints/$M.pt --tag $M --n 400
done

# risk control: constant shrinkage toward bicubic
python3 -m src.evaluate --ckpt checkpoints/rrdb_l1.pt --tag rrdb_l1_s90 \
  --degs id --shrink 0.9 --lpips --save_preds $PRED_IDX
python3 -m src.probes --ckpt checkpoints/rrdb_l1.pt --tag rrdb_l1_s90 --n 400 --shrink 0.9

# learned gate (negative control, validation sweep only)
python3 -m src.gate_sweep --ckpt checkpoints/rrdb_dsc_mkda_unc.pt

python3 -m src.make_numbers
python3 -m src.figures
cd paper && pdflatex -interaction=nonstopmode main >/dev/null && bibtex main >/dev/null && \
  pdflatex -interaction=nonstopmode main >/dev/null && pdflatex -interaction=nonstopmode main >/dev/null
cd ..
echo "DONE"
