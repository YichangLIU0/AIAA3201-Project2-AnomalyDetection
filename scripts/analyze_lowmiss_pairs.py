"""Paired decisions and costs; counts are image-seed events, not independent samples."""
from pathlib import Path
import sys,json
import numpy as np
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'coursework'))
from common import OUT,CATEGORIES
ALL=CATEGORIES+['macaroni1','macaroni2','pcb1','pcb2','pcb3','pcb4']
methods={'coreset25':'coreset','hi448':'lowmiss','multiscale':'lowmiss','core224_peak':'peak','hi448_peak':'peak'}
rows=[];paired=[];costs=[];halves=[]
for category in ALL:
 for seed in range(3):
  base=OUT/'coreset'/category/f'8shot_seed{seed}_coreset25';bm=json.loads((base/'metrics.json').read_text())
  with np.load(base/'predictions.npz') as d:
   paths=d['paths'];truth=d['masks'].reshape(len(paths),-1).any(1);old=d['scores']>bm['image_threshold']
  for method,family in methods.items():
   folder=OUT/family/category/f'8shot_seed{seed}_{method}';m=json.loads((folder/'metrics.json').read_text())
   rows.append({k:m[k] for k in ['category','seed','method','image_auroc','image_ap','pixel_ap','aupro_03','image_fpr','image_fnr']})
   with np.load(folder/('scores.npz' if family=='peak' else 'predictions.npz')) as d:
    assert np.array_equal(paths,d['paths']);new=d['scores']>m['image_threshold']
   paired.append({'category':category,'seed':seed,'method':method,'fn_fixed':int((truth&~old&new).sum()),'fn_introduced':int((truth&old&~new).sum()),'fp_fixed':int((~truth&old&~new).sum()),'fp_introduced':int((~truth&~old&new).sum()),'normal_count':int((~truth).sum()),'anomaly_count':int(truth.sum())})
   costs.append({k:m[k] for k in ['category','seed','method','bank_bytes','feature_seconds_mean','scoring_seconds_mean']})
  h=json.loads((OUT/'lowmiss'/category/f'8shot_seed{seed}_multiscale'/'half_ablations.json').read_text())
  halves.append({'category':category,'seed':seed,**h})
result={'per_run':rows,'paired_against_core224':paired,'costs':costs,'half_bank_score_ablations':halves,'scope':'Paired event counts repeat each test image across three seeds. Component timings come from separate experiment runs and are not a controlled end-to-end benchmark.'}
(OUT/'lowmiss_paired_analysis.json').write_text(json.dumps(result,indent=2))
print('PAIRED_ANALYSIS_COMPLETE',len(rows))
