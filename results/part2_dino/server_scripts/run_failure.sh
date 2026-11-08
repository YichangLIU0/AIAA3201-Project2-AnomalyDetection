#!/bin/bash
P=$HOME/aiaa3201_p2; cd $P
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=$P/.cache/mpl
while tmux has-session -t p2_agg 2>/dev/null; do sleep 10; done
$P/env/bin/python part2_official/failure_analysis.py; echo "failure_exit=$?"
