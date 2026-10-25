from pathlib import Path
import os,sys,json,time,hashlib
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np
import cv2
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/VisA_pytorch/1cls'
OUT=ROOT/'outputs/coursework'
OUT.mkdir(parents=True,exist_ok=True)
CATEGORIES=['candle','capsules','cashew','chewinggum','fryum','pipe_fryum']
SIZE=224

def rgb(path):
 return np.asarray(Image.open(path).convert('RGB').resize((SIZE,SIZE),Image.Resampling.BICUBIC)).copy()

def split(category):
 manifest=OUT/'manifests'/f'{category}.json'
 manifest.parent.mkdir(exist_ok=True)
 train=sorted((DATA/category/'train/good').glob('*.JPG'))
 assert len(train)>64
 rng=np.random.default_rng(20261024)
 shuffled=[train[i] for i in rng.permutation(len(train))]
 cal=shuffled[:40];pool=shuffled[40:]
 refs={str(s):[pool[i] for i in np.random.default_rng(s).choice(len(pool),8,replace=False)] for s in range(3)}
 test=sorted((DATA/category/'test').glob('*/*.JPG'))
 assert set(cal).isdisjoint(pool)
 assert all(set(r).isdisjoint(cal) and len(set(r))==8 for r in refs.values())
 result={'category':category,'calibration':[str(x.relative_to(DATA)) for x in cal],
 'normal_training_pool':[str(x.relative_to(DATA)) for x in pool],
 'references':{s:[str(x.relative_to(DATA)) for x in r] for s,r in refs.items()},
 'test':[str(x.relative_to(DATA)) for x in test],'input_size':SIZE}
 if manifest.exists(): assert json.loads(manifest.read_text())==result,'Manifest changed'
 else: manifest.write_text(json.dumps(result,indent=2))
 return pool,cal,refs,test

def masks_for(paths):
 out=[]
 for p in paths:
  if p.parent.name=='good': out.append(np.zeros((SIZE,SIZE),dtype=bool))
  else:
   m=DATA/p.relative_to(DATA).parts[0]/'ground_truth/bad'/f'{p.stem}.png'
   out.append(np.asarray(Image.open(m).resize((SIZE,SIZE),Image.Resampling.NEAREST))>0)
 return np.stack(out)

def load_backbone():
 import torch
 sys.path.insert(0,str(ROOT/'third_party/AnomalyDINO'))
 from src.backbones import DINOv2Wrapper
 class LocalWrapper(DINOv2Wrapper):
  def load_model(self):
   from torchvision import transforms
   m=torch.hub.load(str(ROOT/'third_party/dinov2'),'dinov2_vits14',source='local',pretrained=False)
   m.load_state_dict(torch.load(ROOT/'checkpoints/dinov2_vits14_pretrain.pth',map_location='cpu',weights_only=True),strict=True)
   self.transform=transforms.Compose([transforms.Resize(self.smaller_edge_size,interpolation=transforms.InterpolationMode.BICUBIC,antialias=True),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
   return m.eval().to(self.device)
 torch.set_num_threads(4)
 return LocalWrapper('dinov2_vits14','cuda',SIZE)

def features(model,path,rotate=False):
 import torch
 from src.utils import augment_image
 key=str(path.relative_to(DATA)).replace('/','_').replace('\\','_')
 cache=OUT/'cache224'/f'{key}_{"rot" if rotate else "query"}.npz'
 cache.parent.mkdir(exist_ok=True)
 if cache.exists():
  with np.load(cache) as d: return d['features'],d['mask'],float(d['seconds'])
 image=rgb(path);images=augment_image(image) if rotate else [image]
 fs=[];ms=[]
 torch.cuda.synchronize();start=time.perf_counter()
 for image in images:
  inp,grid=model.prepare_image(image)
  f=model.extract_features(inp).astype(np.float32)
  np.random.seed(0)
  mask=model.compute_background_mask(f,grid,threshold=10,masking_type=True)
  if not mask.any(): mask=np.ones(len(f),dtype=bool)
  fs.append(f);ms.append(mask)
 torch.cuda.synchronize();seconds=time.perf_counter()-start
 f=np.stack(fs);m=np.stack(ms)
 np.savez(cache,features=f,mask=m,seconds=seconds)
 return f,m,seconds

def top_score(x):
 x=np.asarray(x).ravel();n=max(1,int(len(x)*.01))
 return float(np.partition(x,len(x)-n)[-n:].mean())

def save_result(folder,maps,masks,scores,paths,metrics,cal_maps=None,cal_scores=None):
 folder.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(folder/'predictions.npz',maps=maps.astype(np.float32),scores=scores,masks=masks,paths=np.array([str(p.relative_to(DATA)) for p in paths]))
 if cal_maps is not None: np.savez_compressed(folder/'calibration.npz',maps=cal_maps.astype(np.float32),scores=cal_scores)
 (folder/'metrics.json').write_text(json.dumps(metrics,indent=2))
 print(str(folder.relative_to(OUT)),json.dumps(metrics),flush=True)
