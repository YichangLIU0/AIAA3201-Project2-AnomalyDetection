"""Fixed-budget 448 and 224+448 experiments; no test-label model selection."""
import time,argparse
import torch,faiss
from common import *
from coreset import greedy_coreset_indices
from multiscale import score_features,fuse_scales,fit_reference_scale,reference_groups
from metrics import evaluate,normal_threshold
from sklearn.metrics import roc_auc_score,average_precision_score
sys.path.insert(0,str(ROOT/'third_party/AnomalyDINO'))
from src.utils import augment_image
faiss.omp_set_num_threads(4);torch.set_num_threads(4)
ALL_CATEGORIES=CATEGORIES+['macaroni1','macaroni2','pcb1','pcb2','pcb3','pcb4']

def features448(model,path,rotate=False):
 key=str(path.relative_to(DATA)).replace('/','_').replace('\\','_')
 cache=OUT/'cache448'/f'{key}_{"rot" if rotate else "query"}.npz';cache.parent.mkdir(exist_ok=True)
 if cache.exists():
  with np.load(cache) as d:return d['features'],d['mask'],float(d['seconds'])
 image=np.asarray(Image.open(path).convert('RGB').resize((448,448),Image.Resampling.BICUBIC)).copy()
 views=augment_image(image) if rotate else [image];fs=[];ms=[]
 torch.cuda.synchronize();start=time.perf_counter()
 for view in views:
  tensor,grid=model.prepare_image(view);assert grid==(32,32),grid
  f=model.extract_features(tensor).astype(np.float32);np.random.seed(0)
  mask=model.compute_background_mask(f,grid,10,True)
  if not mask.any():mask=np.ones(len(f),bool)
  fs.append(f);ms.append(mask)
 torch.cuda.synchronize();elapsed=time.perf_counter()-start
 f=np.stack(fs);m=np.stack(ms);np.savez(cache,features=f,mask=m,seconds=elapsed)
 return f,m,elapsed

def predict(bank,queries,grid):
 index=faiss.IndexFlatL2(384);index.add(np.ascontiguousarray(bank));maps=[];scores=[];times=[]
 for f,mask in queries:
  start=time.perf_counter();amap,score=score_features(index,f,mask,grid);times.append(time.perf_counter()-start);maps.append(amap);scores.append(score)
 return np.stack(maps),np.asarray(scores),float(np.mean(times))

def image_metrics(scores,masks,cal_scores):
 labels=masks.reshape(len(masks),-1).any(1);threshold=normal_threshold(cal_scores);prediction=scores>threshold
 return {'image_auroc':float(roc_auc_score(labels,scores)),'image_ap':float(average_precision_score(labels,scores)),'image_fpr':float(prediction[~labels].mean()),'image_fnr':float((~prediction[labels]).mean()),'image_threshold':threshold}

def run(category,model):
 _,cal,refs,test=split(category);masks=masks_for(test);q_low=[];q_high=[];feature_times=[]
 print('HIGHRES_CACHE',category,len(cal)+len(test),flush=True)
 for i,path in enumerate(cal+test):
  f,m,_=features(None,path);q_low.append((f[0],m[0]))
  f,m,t=features448(model,path);q_high.append((f[0],m[0]));feature_times.append(t)
  if i%50==0:print('HIGHRES_FEATURE',category,i,flush=True)
 ncal=len(cal)
 for seed in range(3):
  hi_dir=OUT/'lowmiss'/category/f'8shot_seed{seed}_hi448'
  fused_dir=OUT/'lowmiss'/category/f'8shot_seed{seed}_multiscale'
  if (hi_dir/'metrics.json').exists() and (fused_dir/'metrics.json').exists() and (fused_dir/'half_ablations.json').exists():continue
  low_source=OUT/'coreset'/category/f'8shot_seed{seed}_coreset25'
  with np.load(low_source/'demo_bank.npz') as d:low_bank=d['bank'];low_ids=d['selected_indices']
  high_refs=[features448(model,p,True) for p in refs[str(seed)]]
  low_refs=[features(None,p,True) for p in refs[str(seed)]]
  full=np.ascontiguousarray(np.concatenate([f.reshape(-1,384) for f,_,_ in high_refs]));faiss.normalize_L2(full)
  torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
  ids=greedy_coreset_indices(full,4096,seed=seed+2000,projection_dim=128)
  high_bank=np.ascontiguousarray(full[ids]);torch.cuda.synchronize();selection_time=time.perf_counter()-start
  low_half=low_bank[:2048];high_half=high_bank[:2048]
  low_scale,low_loo=fit_reference_scale(low_half,reference_groups(low_ids[:2048],256),[(f[0],m[0]) for f,m,_ in low_refs],16)
  high_scale,high_loo=fit_reference_scale(high_half,reference_groups(ids[:2048],1024),[(f[0],m[0]) for f,m,_ in high_refs],32)
  print('HIGHRES_EVALUATE',category,seed,'scales',low_scale,high_scale,flush=True)
  if not (hi_dir/'metrics.json').exists():
   maps,scores,timing=predict(high_bank,q_high,32);cm,cs=maps[:ncal],scores[:ncal];tm,ts=maps[ncal:],scores[ncal:]
   result=evaluate(tm,masks,ts,normal_threshold(cs))
   result.update(category=category,seed=seed,shots=8,method='hi448',input_size=448,metric_size=224,bank_tokens=4096,bank_bytes=6291456,bank_build_seconds=selection_time,feature_seconds_mean=float(np.mean(feature_times[ncal:])),scoring_seconds_mean=timing,peak_gpu_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,pixel_threshold=float(np.quantile(cm,.995)),calibration_n=ncal,reference_paths=[str(p.relative_to(DATA)) for p in refs[str(seed)]],protocol='448 native resize; 4096 greedy selected original vectors; unchanged cal40')
   save_result(hi_dir,tm,masks,ts,test,result,cm,cs);np.savez_compressed(hi_dir/'demo_bank.npz',bank=high_bank,selected_indices=ids)
   del maps,scores,cm,cs,tm,ts
  lm,ls,lt=predict(low_half,q_low,16);hm,hs,ht=predict(high_half,q_high,32)
  fused,fs=fuse_scales(lm,hm,ls,hs,low_scale,high_scale);cm,cs=fused[:ncal],fs[:ncal];tm,ts=fused[ncal:],fs[ncal:]
  result=evaluate(tm,masks,ts,normal_threshold(cs))
  result.update(category=category,seed=seed,shots=8,method='multiscale',input_size=[224,448],metric_size=224,bank_tokens=4096,bank_bytes=6291456,feature_seconds_mean=float(np.mean(feature_times[ncal:]))+json.loads((low_source/'metrics.json').read_text())['feature_seconds_mean'],scoring_seconds_mean=lt+ht,peak_gpu_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,pixel_threshold=float(np.quantile(cm,.995)),calibration_n=ncal,reference_paths=[str(p.relative_to(DATA)) for p in refs[str(seed)]],low_scale=low_scale,high_scale=high_scale,protocol='224+448; 2048 vectors per scale; reference-only leave-one-image-out scaling; max fusion; unchanged cal40')
  save_result(fused_dir,tm,masks,ts,test,result,cm,cs)
  ablations={'low_half':image_metrics(ls[ncal:],masks,ls[:ncal]),'high_half':image_metrics(hs[ncal:],masks,hs[:ncal]),'low_reference_loo':low_loo,'high_reference_loo':high_loo,'scale_fit_images':[str(p.relative_to(DATA)) for p in refs[str(seed)]]}
  (fused_dir/'half_ablations.json').write_text(json.dumps(ablations,indent=2))
  np.savez_compressed(fused_dir/'demo_bank.npz',low_bank=low_half,high_bank=high_half,low_scale=low_scale,high_scale=high_scale)
  np.savez_compressed(fused_dir/'half_scores.npz',low_scores=ls,high_scores=hs)
  del full,lm,hm,fused,cm,cs,tm,ts
 print('LOWMISS_CATEGORY_COMPLETE',category,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--categories',nargs='+',default=ALL_CATEGORIES);a=p.parse_args();model=load_backbone(448)
 for category in a.categories:run(category,model)
 print('LOWMISS_ALL_COMPLETE',flush=True)
