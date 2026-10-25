"""Fixed 25% coreset extension; original runs are retained unchanged."""
import time,json,argparse
import numpy as np
import torch,faiss
from common import *
sys.path.insert(0,str(ROOT/"third_party/AnomalyDINO"))
from run_dino import bank_from,predict
from coreset import greedy_coreset_indices
from metrics import evaluate,normal_threshold
faiss.omp_set_num_threads(4);torch.set_num_threads(4)

def coverage(full,bank):
 index=faiss.IndexFlatL2(full.shape[1]);index.add(bank)
 distances=index.search(full,1)[0][:,0]/2
 return {'mean_cosine_distance':float(distances.mean()),'p95_cosine_distance':float(np.quantile(distances,.95)),'max_cosine_distance':float(distances.max())}

def run(category):
 _,cal,refs,test=split(category);masks=masks_for(test)
 queries=[];cal_queries=[];ft=[]
 for p in cal+test:
  f,m,t=features(None,p)
  (cal_queries if p in cal else queries).append((f[0],m[0]));ft.append(t)
 for seed in range(3):
  folder=OUT/'coreset'/category/f'8shot_seed{seed}_coreset25'
  if (folder/'metrics.json').exists() and (folder/'diagnosis.json').exists():continue
  reference=[features(None,p,True)[0] for p in refs[str(seed)]]
  full=bank_from(reference,'full',seed);random=bank_from(reference,'random25',seed);pooled=bank_from(reference,'pooled25',seed)
  print('CORESET_SELECT',category,seed,flush=True)
  torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
  indices=greedy_coreset_indices(full,len(full)//4,seed=seed+2000,projection_dim=128)
  bank=np.ascontiguousarray(full[indices]);torch.cuda.synchronize();selection=time.perf_counter()-start
  peak=torch.cuda.max_memory_allocated()/1024**2
  index=faiss.IndexFlatL2(bank.shape[1]);index.add(bank)
  cm,cs,_=predict(index,cal_queries);threshold=normal_threshold(cs)
  maps,scores,timing=predict(index,queries)
  result=evaluate(maps,masks,scores,threshold)
  result.update(category=category,seed=seed,shots=8,method='coreset25',reference_paths=[str(p.relative_to(DATA)) for p in refs[str(seed)]],input_size=SIZE,metric_size=SIZE,pixel_threshold=float(np.quantile(cm,.995)),calibration_n=len(cal),calibration_fpr=float((cs>threshold).mean()),feature_seconds_mean=float(np.mean(ft[len(cal):])),scoring_seconds_mean=timing,bank_build_seconds=selection,bank_bytes=int(bank.nbytes),bank_tokens=len(bank),selection_peak_gpu_allocated_mb=peak,selection_projection_dim=128,selection_seed=seed+2000,protocol='same frozen cached features, projected greedy k-center, 25 percent original vectors')
  diagnosis={'category':category,'seed':seed,'coverage':{name:coverage(full,b) for name,b in [('random25',random),('pooled25',pooled),('coreset25',bank)]},'comparisons':{}}
  truth=masks.reshape(len(masks),-1).any(1);new_pred=scores>threshold
  for method in ['full','random25','pooled25']:
   base=OUT/'dinov2'/category/f'8shot_seed{seed}_{method}'
   old=json.loads((base/'metrics.json').read_text());old_scores=np.load(base/'predictions.npz')['scores'];old_pred=old_scores>old['image_threshold']
   diagnosis['comparisons'][method]={'old_fp':int((old_pred&~truth).sum()),'new_fp':int((new_pred&~truth).sum()),'fp_fixed':int((old_pred&~new_pred&~truth).sum()),'fp_introduced':int((~old_pred&new_pred&~truth).sum()),'old_fn':int((~old_pred&truth).sum()),'new_fn':int((~new_pred&truth).sum()),'fn_fixed':int((~old_pred&new_pred&truth).sum()),'fn_introduced':int((old_pred&~new_pred&truth).sum()),'normal_score_mean_old':float(old_scores[~truth].mean()),'normal_score_mean_new':float(scores[~truth].mean()),'old_threshold':old['image_threshold'],'new_threshold':threshold}
  save_result(folder,maps,masks,scores,test,result,cm,cs)
  np.savez_compressed(folder/'demo_bank.npz',bank=bank,selected_indices=indices)
  (folder/'diagnosis.json').write_text(json.dumps(diagnosis,indent=2))
  print('CORESET_DIAGNOSIS',json.dumps(diagnosis),flush=True)
 print('CORESET_CATEGORY_DONE',category,flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--categories',nargs='+',default=CATEGORIES);args=parser.parse_args()
 for category in args.categories:run(category)
 print('CORESET_EXTENSION_COMPLETE',flush=True)
