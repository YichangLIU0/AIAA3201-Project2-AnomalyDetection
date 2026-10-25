"""Bounded illumination stress test; thresholds are kept fixed from clean normal calibration."""
import torch,faiss
from run_dino import predict
from common import *
from metrics import evaluate
faiss.omp_set_num_threads(4)
model=load_backbone()
for c in CATEGORIES:
 _,_,_,test=split(c)
 selected=sorted([p for p in test if p.parent.name=='good'])[:10]+sorted([p for p in test if p.parent.name=='bad'])[:10]
 masks=masks_for(selected)
 for factor in [0.8,1.,1.2]:
  queries=[]
  for p in selected:
   image=np.clip(rgb(p).astype(np.float32)*factor,0,255).astype(np.uint8)
   tensor,grid=model.prepare_image(image);f=model.extract_features(tensor).astype(np.float32)
   np.random.seed(0);mask=model.compute_background_mask(f,grid,threshold=10,masking_type=True)
   if not mask.any():mask=np.ones(len(f),dtype=bool)
   queries.append((f,mask))
  for method in ['full','random25','pooled25']:
   base=OUT/'dinov2'/c/f'8shot_seed0_{method}'
   bank=np.load(base/'demo_bank.npz')['bank'];index=faiss.IndexFlatL2(384);index.add(bank)
   info=json.loads((base/'metrics.json').read_text());maps,scores,t=predict(index,queries)
   r=evaluate(maps,masks,scores,info['image_threshold'])
   r.update(category=c,method=method,brightness_factor=factor,seed=0,shots=8,pixel_threshold=info['pixel_threshold'],scope='20 fixed test images per category; synthetic brightness shift; thresholds unchanged')
   save_result(OUT/'robustness'/c/f'{factor}_{method}',maps,masks,scores,selected,r)
print('ROBUSTNESS_COMPLETE',flush=True)
