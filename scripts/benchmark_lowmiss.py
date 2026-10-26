"""Single-process warmed demo timing; fixed inputs and randomized method order."""
from pathlib import Path
import sys,json,time
import numpy as np
import torch
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'coursework'))
from common import OUT,DATA,CATEGORIES
from demo import Detector
methods=['coreset25','hi448','multiscale','core224_peak','hi448_peak'];categories=CATEGORIES+['macaroni1','macaroni2','pcb1','pcb2','pcb3','pcb4']
detector=Detector();first=sorted((DATA/'candle/test/good').glob('*.JPG'))[0]
for method in methods:detector.infer(first,'candle',method)
rng=np.random.default_rng(20261026);records=[]
for category in categories:
 for kind in ['good','bad']:
  path=sorted((DATA/category/'test'/kind).glob('*.JPG'))[0]
  for method in rng.permutation(methods):
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
   detector.infer(path,category,str(method))
   torch.cuda.synchronize();seconds=time.perf_counter()-start
   records.append({'category':category,'input':str(path.relative_to(DATA)),'method':str(method),'seconds':seconds,'peak_gpu_allocated_mb':torch.cuda.max_memory_allocated()/1024**2})
summary=[]
for method in methods:
 rr=[x for x in records if x['method']==method];ts=[x['seconds'] for x in rr]
 summary.append({'method':method,'n':len(rr),'mean_seconds':float(np.mean(ts)),'median_seconds':float(np.median(ts)),'p95_seconds':float(np.quantile(ts,.95)),'max_gpu_allocated_mb':max(x['peak_gpu_allocated_mb'] for x in rr)})
result={'protocol':'One shared initialized frozen backbone. Five warmup calls. First good and first bad filename per category. Fixed randomized method order per image. Synchronized wall time includes image decode, bank load, feature extraction, PCA, exact matching and map/score generation; excludes model initialization and rendering. No cached feature reuse. Peak allocated CUDA memory includes resident backbone, not total device occupancy. 24 images per method; no repeated-trial confidence claim.','records':records,'summary':summary}
(OUT/'online_benchmark.json').write_text(json.dumps(result,indent=2));print(json.dumps(summary,indent=2))
