#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/.pylibs:$PWD"
export TORCH_HOME="$PWD/.cache/torch"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export MPLCONFIGDIR="$PWD/.cache/mpl"
mkdir -p logs checkpoints
for spec in \
  "rrdb_l1 rrdb l1 0.8" \
  "rrdb_l1_mkda rrdb l1 multi" \
  "rrdb_dsc_mkda rrdb l1_dsc multi" \
  "rrdb_perc rrdb l1_perc 0.8" ; do
  set -- $spec
  echo "=== training $1 ($(date)) ==="
  python3 -m src.train --tag "$1" --arch "$2" --loss "$3" --sigmas "$4" --steps 5000 --batch 8 --lr 2e-4
done
python3 -m src.train_unc --ckpt checkpoints/rrdb_dsc_mkda.pt --tag rrdb_dsc_mkda --steps 1500
echo "ALL DONE $(date)"
