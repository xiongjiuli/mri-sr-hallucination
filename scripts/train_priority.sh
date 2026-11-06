#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/.pylibs:$PWD"
export TORCH_HOME="$PWD/.cache/torch"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export MPLCONFIGDIR="$PWD/.cache/mpl"
python3 -m src.train --tag rrdb_dsc_mkda --arch rrdb --loss l1_dsc --sigmas multi --steps 5000 --batch 8 --lr 2e-4
python3 -m src.train --tag rrdb_perc --arch rrdb --loss l1_perc --sigmas 0.8 --steps 5000 --batch 8 --lr 2e-4
python3 -m src.train --tag rrdb_dsc --arch rrdb --loss l1_dsc --sigmas 0.8 --steps 3000 --batch 8 --lr 2e-4
