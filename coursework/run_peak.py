"""Peak-of-smoothed-map ablation. Threshold is fitted only to normal map maxima."""
import sys,json,argparse,time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from common import *
from metrics import normal_threshold
ALL=CATEGORIES+['macaroni1','macaroni2','pcb1','pcb2','pcb3','pcb4']
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--categories',nargs='+',default=ALL);args=parser.parse_args()
 for category in args.categories:
  for seed in range(3):
   for family,parent,method in [('coreset','coreset25','core224_peak'),('lowmiss','hi448','hi448_peak')]:
    source=OUT/family/category/f'8shot_seed{seed}_{parent}';target=OUT/'peak'/category/f'8shot_seed{seed}_{method}'
    if (target/'metrics.json').exists():continue
    base=json.loads((source/'metrics.json').read_text())
    with np.load(source/'predictions.npz') as d:
     maps=d['maps'];masks=d['masks'];paths=d['paths'];truth=masks.reshape(len(masks),-1).any(1)
     start=time.perf_counter();scores=maps.reshape(len(maps),-1).max(1);extra_time=(time.perf_counter()-start)/len(maps)
    with np.load(source/'calibration.npz') as d:
     cm=d['maps'];cal_scores=cm.reshape(len(cm),-1).max(1)
    threshold=normal_threshold(cal_scores);pred=scores>threshold
    base.update(method=method,image_auroc=float(roc_auc_score(truth,scores)),image_ap=float(average_precision_score(truth,scores)),image_fpr=float(pred[~truth].mean()),image_fnr=float((~pred[truth]).mean()),image_threshold=threshold,calibration_fpr=float((cal_scores>threshold).mean()),scoring_seconds_mean=base['scoring_seconds_mean']+extra_time,map_source=str(source.relative_to(OUT)).replace('\\','/'),protocol='parent maps unchanged; max smoothed-map image score; independently normal-calibrated threshold')
    target.mkdir(parents=True,exist_ok=True);np.savez_compressed(target/'scores.npz',scores=scores,paths=paths,labels=truth);np.savez_compressed(target/'calibration.npz',scores=cal_scores)
    (target/'metrics.json').write_text(json.dumps(base,indent=2));print('PEAK_RESULT',category,seed,method,base['image_fnr'],base['image_fpr'],flush=True)
 print('PEAK_COMPLETE',flush=True)
