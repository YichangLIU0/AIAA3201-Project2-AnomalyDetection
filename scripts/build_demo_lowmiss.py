"""Re-infer filename-selected examples, including mistakes, without cached queries."""
from pathlib import Path
import sys,json,subprocess,zipfile
import numpy as np
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'coursework'))
from common import OUT,DATA,CATEGORIES
from demo import Detector,panel
methods=['hi448','multiscale','core224_peak','hi448_peak']
all_categories=CATEGORIES+['macaroni1','macaroni2','pcb1','pcb2','pcb3','pcb4']
folder=r/'deliverables/demo_lowmiss';folder.mkdir(parents=True,exist_ok=True)
detector=Detector();records=[]
for category in all_categories:
 for method in methods:
  family='peak' if method.endswith('_peak') else 'lowmiss'
  base=OUT/family/category/f'8shot_seed0_{method}'
  info=json.loads((base/'metrics.json').read_text());source=OUT/info['map_source'] if family=='peak' else base
  with np.load(source/'predictions.npz') as saved:
   labels=saved['masks'].reshape(len(saved['scores']),-1).any(1)
   scores=saved['scores']
   if family=='peak':
    with np.load(base/'scores.npz') as scored:scores=scored['scores']
   for kind,idx in [('normal',int(np.flatnonzero(~labels)[0])),('anomalous',int(np.flatnonzero(labels)[0]))]:
    path=DATA/str(saved['paths'][idx]);im,am,result=detector.infer(path,category,method,review_quantile=.9)
    assert abs(result['score']-float(scores[idx]))<1e-5,(category,method,result['score'],scores[idx])
    assert np.allclose(am,saved['maps'][idx],atol=1e-5)
    result.update(input=str(path.relative_to(DATA)),ground_truth=kind,matches_saved_prediction=True)
    target=folder/method;target.mkdir(exist_ok=True)
    number=sum(x['method']==method for x in records)
    panel(im,am,result).save(target/f'frame_{number:02d}.png');records.append(result)
  print('DEMO_VERIFIED',category,method,flush=True)
(folder/'predictions.json').write_text(json.dumps(records,indent=2))
(folder/'README.txt').write_text('Actual fresh inference on the first normal and first anomalous test image by filename for all twelve categories, under four variants. Includes errors; examples were not selected for success. Maps and scores match formal saved predictions within 1e-5. The normal-calibration 0.90 review threshold is an illustrative predefined setting, not a test-optimized choice. REVIEW means deferred decision, not a correctly detected defect. Each video frame lasts three seconds. Input paths are relative to the official one-class dataset. Timing includes per-query bank loading, feature extraction and matching, but excludes model initialization and rendering; these demonstration timings are not a controlled benchmark.\n')
ffmpeg=Path('C:/Program Files/ffmpeg-2025-11-06-git-222127418b-essentials_build/bin/ffmpeg.exe')
for method in methods:
 subprocess.run([str(ffmpeg),'-y','-framerate','1/3','-i',str(folder/method/'frame_%02d.png'),'-c:v','libx264','-r','25','-pix_fmt','yuv420p',str(folder/method/'anomaly_demo.mp4')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
with zipfile.ZipFile(r/'deliverables/anomaly_demo.zip','w',zipfile.ZIP_DEFLATED) as z:
 for sub in ['demo','demo_coreset','demo_lowmiss']:
  for p in (r/'deliverables'/sub).rglob('*'):
   if p.is_file():z.write(p,p.relative_to(r/'deliverables'))
print('LOWMISS_DEMO_COMPLETE',len(records),flush=True)
