#!/bin/bash
P=$HOME/aiaa3201_p2; cd $P
export TMPDIR=$P/.tmp XDG_CACHE_HOME=$P/.cache PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TORCH_HOME=$P/.cache/torch
date; nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
if [ "$FREE" -lt 4000 ]; then echo "NOT_STARTED low GPU memory"; exit 3; fi
$P/env/bin/python part2_official/run_dino_formal.py --out-dir $P/outputs/dino
echo "grid_exit=$?"; date; df -h $P | tail -1
echo GRID_FINISHED
