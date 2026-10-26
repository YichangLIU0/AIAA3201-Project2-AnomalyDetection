"""Official DINOv2/AnomalyDINO primitives; explicit random reference manifests."""
import argparse,time,json
import numpy as np
import faiss
import torch
from scipy.ndimage import gaussian_filter
from common import *
from metrics import evaluate,normal_threshold
faiss.omp_set_num_threads(4)

def bank_from(ref_features,method,seed):
 chunks=[]
 for f in ref_features:
  # [8 rotations,256 tokens,384 channels]
  if method=='pooled25':
   f=f.reshape(8,8,2,8,2,384).mean((2,4)).reshape(-1,384)
  else: f=f.reshape(-1,384)
  chunks.append(f)
 bank=np.ascontiguousarray(np.concatenate(chunks),dtype=np.float32)
 if method=='random25':
  bank=bank[np.random.default_rng(seed+1000).choice(len(bank),len(bank)//4,replace=False)].copy()
 faiss.normalize_L2(bank)
 return bank

def predict(index,queries):
 maps=[];scores=[];times=[]
 for f,mask in queries:
  start=time.perf_counter()
  q=np.ascontiguousarray(f[mask],dtype=np.float32)
  faiss.normalize_L2(q)
  d,_=index.search(q,1)
  patch=np.zeros(len(f),dtype=np.float32);patch[mask]=np.maximum(d[:,0]/2,0)
  scores.append(top_score(patch))
  maps.append(gaussian_filter(cv2.resize(patch.reshape(16,16),(SIZE,SIZE)),sigma=4))
  times.append(time.perf_counter()-start)
 return np.stack(maps),np.array(scores),float(np.mean(times))

def run(category,model):
 pool,cal,refs,test=split(category)
 masks=masks_for(test)
 print('CACHE',category,'test',len(test),'calibration',len(cal),flush=True)
 queries=[];cal_queries=[];feature_times=[]
 for i,p in enumerate(cal+test):
  f,m,t=features(model,p)
  (cal_queries if i<len(cal) else queries).append((f[0],m[0]))
  if i>=len(cal): feature_times.append(t)
  if i%40==0: print('FEATURE',category,i,'/',len(cal+test),flush=True)
 for seed in range(3):
  reference_features=[features(model,p,rotate=True)[0] for p in refs[str(seed)]]
  for shots in [1,2,4,8]:
   for method in (['full','random25','pooled25'] if shots==8 else ['full']):
    folder=OUT/'dinov2'/category/f'{shots}shot_seed{seed}_{method}'
    if all((folder/name).exists() for name in ['metrics.json','predictions.npz','calibration.npz']) and (seed!=0 or shots!=8 or (folder/'demo_bank.npz').exists()): continue
    start=time.perf_counter();bank=bank_from(reference_features[:shots],method,seed)
    index=faiss.IndexFlatL2(bank.shape[1]);index.add(bank)
    bank_time=time.perf_counter()-start
    cm,cs,ct=predict(index,cal_queries)
    threshold=normal_threshold(cs)
    pixel_threshold=float(np.quantile(cm,.995))
    maps,scores,score_time=predict(index,queries)
    result=evaluate(maps,masks,scores,threshold)
    result.update(category=category,seed=seed,shots=shots,method=method,reference_paths=[str(p.relative_to(DATA)) for p in refs[str(seed)][:shots]],input_size=SIZE,metric_size=SIZE,pixel_threshold=pixel_threshold,calibration_fpr=float((cs>threshold).mean()),feature_seconds_mean=float(np.mean(feature_times)),scoring_seconds_mean=score_time,bank_build_seconds=bank_time,bank_bytes=int(bank.nbytes),bank_tokens=len(bank),peak_gpu_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,protocol='square224_fp32_8rot_queryPCA_cal40',calibration_n=len(cal))
    save_result(folder,maps,masks,scores,test,result,cm,cs)
    if seed==0 and shots==8:
     np.savez_compressed(folder/'demo_bank.npz',bank=bank)
    del maps,cm,index,bank
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--categories',nargs='+',default=CATEGORIES);args=p.parse_args()
 model=load_backbone()
 for category in args.categories: run(category,model)
 print('ALL_DINO_EXPERIMENTS_COMPLETE',flush=True)
