#!/bin/bash
P=$HOME/aiaa3201_p2; cd $P
export TMPDIR=$P/.tmp XDG_CACHE_HOME=$P/.cache PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TORCH_HOME=$P/.cache/torch
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
echo "gpu_free_mib_before=$FREE"; if [ "$FREE" -lt 4000 ]; then echo "NOT_STARTED low GPU memory"; exit 3; fi
mkdir -p outputs/server_parity/dino
# sample this process's GPU memory from nvidia-smi alongside the run
( while sleep 1; do nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits; done > logs/parity_gpu_samples.txt ) &
SAMPLER=$!
START=$(date +%s.%N)
$P/env/bin/python part2_official/run_dino_formal.py --categories candle --shots 1 --seeds 0 --out-dir $P/outputs/server_parity/dino &
PID=$!; echo "python_pid=$PID"; wait $PID; RC=$?
END=$(date +%s.%N); kill $SAMPLER
echo "exit=$RC wall_seconds=$(echo "$END - $START" | bc 2>/dev/null || python3 -c "print($END-$START)")"
echo "peak_nvsmi_mib=$(awk -F', ' -v p=$PID '$1==p && $2>m {m=$2} END{print m+0}' logs/parity_gpu_samples.txt)"
echo PARITY_RUN_DONE
