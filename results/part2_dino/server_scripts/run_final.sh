#!/bin/bash
P=$HOME/aiaa3201_p2; cd $P
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TMPDIR=$P/.tmp XDG_CACHE_HOME=$P/.cache TORCH_HOME=$P/.cache/torch MPLCONFIGDIR=$P/.cache/mpl
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
if [ "$FREE" -gt 4000 ]; then $P/env/bin/python part2_official/dino_interface.py --run outputs/dino/candle/1shot_seed0 --images 5; echo "interface_exit=$?"; else echo INTERFACE_NOT_RUN low GPU; fi
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 $P/env/bin/python part2_official/failure_analysis.py; echo "failure_exit=$?"
echo FINAL_DONE
