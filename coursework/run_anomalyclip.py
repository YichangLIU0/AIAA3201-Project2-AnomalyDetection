"""Official AnomalyCLIP checkpoint, adapted only for shared data/metrics protocol."""
import argparse,time,hashlib
import torch
from torchvision import transforms
from scipy.ndimage import gaussian_filter
from common import *
from metrics import evaluate,normal_threshold
sys.path.insert(0,str(ROOT/'third_party/AnomalyCLIP'))
import AnomalyCLIP_lib
from prompt_ensemble import AnomalyCLIP_PromptLearner

def main():
 torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
 weight=ROOT/'checkpoints/ViT-L-14-336px.pt'
 expected='3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02'
 assert hashlib.file_digest(weight.open('rb'),'sha256').hexdigest()==expected,'CLIP backbone checksum mismatch'
 details={'Prompt_length':12,'learnabel_text_embedding_depth':9,'learnabel_text_embedding_length':4}
 model,_=AnomalyCLIP_lib.load('ViT-L/14@336px',device='cpu',design_details=details,download_root=str(ROOT/'checkpoints'))
 learner=AnomalyCLIP_PromptLearner(model,details)
 checkpoint=ROOT/'third_party/AnomalyCLIP/checkpoints/9_12_4_multiscale_visa/epoch_15.pth'
 learner.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['prompt_learner'])
 model.eval().cuda();learner.eval().cuda();model.visual.DAPM_replace(DPAM_layer=20)
 for p in model.parameters():p.requires_grad_(False)
 for p in learner.parameters():p.requires_grad_(False)
 with torch.inference_mode():
  prompts,tokens,compound=learner(cls_id=None)
  text=model.encode_text_learn(prompts,tokens,compound).float()
  text=torch.stack(torch.chunk(text,dim=0,chunks=2),dim=1)
  text=text/text.norm(dim=-1,keepdim=True)
 transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
 def infer(paths):
  maps=[];scores=[];times=[]
  for i,p in enumerate(paths):
   image=transform(rgb(p))[None].cuda();torch.cuda.synchronize();start=time.perf_counter()
   with torch.inference_mode():
    image_f,patches=model.encode_image(image,[24],DPAM_layer=20)
    image_f=image_f/image_f.norm(dim=-1,keepdim=True)
    prob=((image_f@text.permute(0,2,1))/.07).softmax(-1)[:,0,1]
    patch=patches[0]/patches[0].norm(dim=-1,keepdim=True)
    similarity,_=AnomalyCLIP_lib.compute_similarity(patch,text[0])
    sm=AnomalyCLIP_lib.get_similarity_map(similarity[:,1:,:],SIZE)
    amap=((sm[...,1]+1-sm[...,0])/2)[0].cpu().numpy()
   amap=gaussian_filter(amap,4)
   scores.append(float(prob.item()));maps.append(amap);torch.cuda.synchronize();times.append(time.perf_counter()-start)
   if i%40==0:print('CLIP_IMAGE',i,'/',len(paths),flush=True)
  return np.stack(maps),np.array(scores),float(np.mean(times))
 for category in CATEGORIES:
  folder=OUT/'anomalyclip'/category/'zero_shot'
  if (folder/'metrics.json').exists():continue
  _,cal,_,test=split(category);print('CLIP_CATEGORY',category,flush=True)
  torch.cuda.reset_peak_memory_stats()
  cm,cs,ct=infer(cal);maps,scores,t=infer(test)
  result=evaluate(maps,masks_for(test),scores,normal_threshold(cs))
  result.update(category=category,method='anomalyclip',seed=0,shots=0,input_size=SIZE,metric_size=SIZE,pixel_threshold=float(np.quantile(cm,.995)),calibration_n=40,inference_seconds_mean=t,peak_gpu_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,checkpoint=str(checkpoint.relative_to(ROOT)),protocol='official target-VisA prompt checkpoint; no target fitting; 40 target normal images for threshold calibration only')
  save_result(folder,maps,masks_for(test),scores,test,result,cm,cs)
 print('ANOMALYCLIP_COMPLETE',flush=True)
if __name__=='__main__':main()
