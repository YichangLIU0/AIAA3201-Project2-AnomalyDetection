"""Eight-shot classical and reconstruction baselines, same reference lists as DINO."""
import time,argparse
import torch
from torch import nn
from scipy.ndimage import gaussian_filter
from common import *
from metrics import evaluate,normal_threshold
cv2.setNumThreads(4);torch.set_num_threads(4)
class Autoencoder(nn.Module):
 def __init__(self):
  super().__init__()
  self.net=nn.Sequential(nn.Conv2d(3,16,4,2,1),nn.ReLU(),nn.Conv2d(16,32,4,2,1),nn.ReLU(),nn.Conv2d(32,64,4,2,1),nn.ReLU(),nn.ConvTranspose2d(64,32,4,2,1),nn.ReLU(),nn.ConvTranspose2d(32,16,4,2,1),nn.ReLU(),nn.ConvTranspose2d(16,3,4,2,1),nn.Sigmoid())
 def forward(self,x): return self.net(x)

def template_predict(images,refs):
 desc=np.stack([cv2.resize(x,(28,28)).astype(np.float32) for x in refs])
 maps=[];scores=[];failures=0;start=time.perf_counter()
 for image in images:
  nearest=int(np.argmin(((desc-cv2.resize(image,(28,28)))**2).mean((1,2,3))))
  reference=refs[nearest]
  gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
  target=cv2.cvtColor(reference,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
  warp=np.eye(2,3,dtype=np.float32)
  try:
   _,warp=cv2.findTransformECC(gray,target,warp,cv2.MOTION_EUCLIDEAN,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,1e-4),None,3)
   aligned=cv2.warpAffine(reference,warp,(SIZE,SIZE),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REFLECT)
  except cv2.error:
   aligned=reference;failures+=1
  error=np.abs(image.astype(np.float32)-aligned.astype(np.float32)).mean(2)/255.
  error=gaussian_filter(error,sigma=2)
  maps.append(error);scores.append(top_score(error))
 return np.stack(maps),np.array(scores),(time.perf_counter()-start)/len(images),failures

def ae_predict(model,images):
 maps=[];scores=[];torch.cuda.synchronize();start=time.perf_counter()
 with torch.inference_mode():
  for image in images:
   x=torch.from_numpy(image.transpose(2,0,1).copy()).float()[None].cuda()/255
   pred=model(x)
   error=(x-pred).abs().mean(1)[0].cpu().numpy()
   error=gaussian_filter(error,2)
   maps.append(error);scores.append(top_score(error))
 torch.cuda.synchronize()
 return np.stack(maps),np.array(scores),(time.perf_counter()-start)/len(images)

def run(category):
 pool,cal,refs,test=split(category)
 cal_images=[rgb(p) for p in cal];test_images=[rgb(p) for p in test];masks=masks_for(test)
 # Same eight rotation angles and interpolation as the foundation baseline.
 sys.path.insert(0,str(ROOT/'third_party/AnomalyDINO'))
 from src.utils import augment_image
 for seed in range(3):
  reference_images=[rgb(p) for p in refs[str(seed)]]
  augmented=[a for x in reference_images for a in augment_image(x)]
  for method in ['template_ecc','autoencoder']:
   folder=OUT/'baselines'/category/f'8shot_seed{seed}_{method}'
   if (folder/'metrics.json').exists(): continue
   start=time.perf_counter();losses=[]
   if method=='autoencoder':
    torch.manual_seed(seed);np.random.seed(seed)
    torch.cuda.reset_peak_memory_stats()
    model=Autoencoder().cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    train=torch.from_numpy(np.stack(augmented).transpose(0,3,1,2).copy()).float()/255
    gen=torch.Generator().manual_seed(seed)
    for epoch in range(15):
     loss_sum=0.
     for indices in torch.randperm(len(train),generator=gen).split(8):
      x=train[indices].cuda();optimizer.zero_grad();loss=(model(x)-x).square().mean();loss.backward();optimizer.step();loss_sum+=loss.item()
     losses.append(loss_sum/(len(train)/8))
    model.eval();torch.cuda.synchronize();train_seconds=time.perf_counter()-start
    cm,cs,ct=ae_predict(model,cal_images);maps,scores,timing=ae_predict(model,test_images)
    folder.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),folder/'autoencoder.pt')
    (folder/'training_loss.json').write_text(json.dumps(losses))
    gpu=torch.cuda.max_memory_allocated()/1024**2;failures=0
   else:
    train_seconds=time.perf_counter()-start
    cm,cs,ct,cf=template_predict(cal_images,augmented)
    maps,scores,timing,failures=template_predict(test_images,augmented);gpu=0.
   threshold=normal_threshold(cs);result=evaluate(maps,masks,scores,threshold)
   result.update(category=category,method=method,seed=seed,shots=8,reference_paths=[str(p.relative_to(DATA)) for p in refs[str(seed)]],input_size=SIZE,metric_size=SIZE,epochs=15 if method=='autoencoder' else 0,pixel_threshold=float(np.quantile(cm,.995)),calibration_fpr=float((cs>threshold).mean()),inference_seconds_mean=timing,training_seconds=train_seconds,peak_gpu_allocated_mb=gpu,ecc_failed_test_alignments=failures)
   save_result(folder,maps,masks,scores,test,result,cm,cs)
   if method=='autoencoder': del model,optimizer,train;torch.cuda.empty_cache()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--categories',nargs='+',default=CATEGORIES);a=p.parse_args()
 for c in a.categories: run(c)
 print('ALL_CLASSICAL_BASELINES_COMPLETE',flush=True)
