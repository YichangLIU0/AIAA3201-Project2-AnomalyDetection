#!/bin/bash
# queued behind the DINO grid: only one of our GPU jobs at a time; candle first, then the rest
P=$HOME/aiaa3201_p2; cd $P
export TMPDIR=$P/.tmp XDG_CACHE_HOME=$P/.cache PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TORCH_HOME=$P/.cache/torch
while tmux has-session -t p2_dino 2>/dev/null; do sleep 15; done
date; nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
if [ "$FREE" -lt 8000 ]; then echo "NOT_STARTED low GPU memory"; exit 3; fi
$P/env/bin/python part2_official/run_anomalyclip_formal.py --out-dir $P/outputs/anomalyclip
echo "clip_exit=$?"; date; df -h $P | tail -1
echo CLIP_FINISHED
