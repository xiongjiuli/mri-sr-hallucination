#!/usr/bin/env bash
# End-to-end reproduction of the main results.
# Usage: bash scripts/run_all.sh [prepare|train|eval|all]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/.pylibs:$PWD"
export TORCH_HOME="$PWD/.cache/torch"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export MPLCONFIGDIR="$PWD/.cache/mpl"
mkdir -p "$MPLCONFIGDIR" logs checkpoints results

STAGE="${1:-all}"

if [[ "$STAGE" == "prepare" || "$STAGE" == "all" ]]; then
  if [[ ! -f data/IXI-T1.tar ]]; then
    python3 scripts/download_ixi.py
  fi
  if [[ ! -d data/IXI-T1 ]]; then
    mkdir -p data/raw && tar -xf data/IXI-T1.tar -C data/raw
  fi
  python3 -m src.prepare_ixi --n 72
fi

TRAIN=${TRAIN:-1}
if [[ "$STAGE" == "train" || "$STAGE" == "all" ]]; then
  if [[ "$TRAIN" == "1" ]]; then
    python3 -m src.train --tag rrdb_l1          --arch rrdb     --loss l1          --sigmas 0.8 --steps 6000 --batch 8 --lr 2e-4
    python3 -m src.train --tag rrdb_l1_mkda     --arch rrdb     --loss l1          --sigmas multi --steps 6000 --batch 8 --lr 2e-4
    python3 -m src.train --tag rrdb_dsc         --arch rrdb     --loss l1_dsc      --sigmas 0.8 --steps 6000 --batch 8 --lr 2e-4
    python3 -m src.train --tag rrdb_dsc_mkda    --arch rrdb     --loss l1_dsc      --sigmas multi --steps 6000 --batch 8 --lr 2e-4
    python3 -m src.train --tag rrdb_perc        --arch rrdb     --loss l1_perc     --sigmas 0.8 --steps 6000 --batch 8 --lr 2e-4
    python3 -m src.train --tag srresnet_l1      --arch srresnet --loss l1          --sigmas 0.8 --steps 6000 --batch 8 --lr 2e-4
  fi
  python3 -m src.train_unc --ckpt checkpoints/rrdb_dsc_mkda.pt --tag rrdb_dsc_mkda --steps 1500
fi

if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
  for M in bicubic rrdb_l1 rrdb_l1_mkda rrdb_dsc rrdb_dsc_mkda rrdb_perc srresnet_l1; do
    if [[ "$M" == "bicubic" ]]; then continue; fi
    python3 -m src.evaluate --ckpt checkpoints/$M.pt --tag $M --degs id,blur1.4,noise,bicubic --lpips
    python3 -m src.probes --ckpt checkpoints/$M.pt --tag $M --n 240
  done
  python3 -m src.baselines
fi
