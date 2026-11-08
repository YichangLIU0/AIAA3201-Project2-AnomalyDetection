#!/bin/bash
P=$HOME/aiaa3201_p2; cd $P
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
date
$P/env/bin/python part2_official/aggregate_dino_formal.py 2>&1 | tail -22
$P/env/bin/python - <<'P'
import json,subprocess,sys
from pathlib import Path
sys.path.insert(0,'.')
from shared_protocol.metrics import aggregate_runs
from shared_protocol.protocol import CATEGORIES,save_json
rows,mis=[],[]
for c in CATEGORIES:
    d=Path('outputs/anomalyclip')/c/'zero_shot_seed0'; m=json.loads((d/'metrics.json').read_text(encoding='utf-8'))
    r=subprocess.run([sys.executable,'-m','shared_protocol.evaluate_predictions','--run',str(d),'--output',str(d/'metrics_reeval.json')],capture_output=True,text=True)
    if r.returncode or json.loads((d/'metrics_reeval.json').read_text(encoding='utf-8'))!=m: mis.append(c)
    rows.append(m)
a=aggregate_runs(rows,CATEGORIES); Path('outputs/anomalyclip/summary').mkdir(exist_ok=True); save_json('outputs/anomalyclip/summary/aggregate.json',a)
print('CLIP reeval_mismatch',mis); print(json.dumps(a,indent=0)[:2500])
P
date; echo AGG_DONE
