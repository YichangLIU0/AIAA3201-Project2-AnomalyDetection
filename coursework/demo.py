"""Runnable local image demo, sharing the evaluated DINO scoring path."""
import argparse,json,time
import numpy as np
import faiss
from PIL import Image,ImageDraw,ImageFont,ImageTk
from run_dino import predict
from common import *
faiss.omp_set_num_threads(4)
class Detector:
 def __init__(self):
  self.model=load_backbone()
 def _query(self,path,size):
  import torch
  from torchvision import transforms
  image=np.asarray(Image.open(path).convert('RGB').resize((size,size),Image.Resampling.BICUBIC)).copy()
  tensor=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])(image)
  features=self.model.extract_features(tensor).astype(np.float32);np.random.seed(0)
  grid=(size//14,size//14);mask=self.model.compute_background_mask(features,grid,10,True)
  if not mask.any():mask=np.ones(len(features),dtype=bool)
  return features,mask
 def infer(self,path,category,method='hi448_peak',review_quantile=None):
  from multiscale import score_features,fuse_scales
  from metrics import normal_threshold
  from selective import decisions
  peak_parent={'core224_peak':'coreset25','hi448_peak':'hi448'}.get(method)
  family='peak' if peak_parent else 'lowmiss' if method in ['hi448','multiscale','core224_peak','hi448_peak'] else ('coreset' if method=='coreset25' else 'dinov2')
  folder=OUT/family/category/f'8shot_seed0_{method}'
  info=json.loads((folder/'metrics.json').read_text())
  start=time.perf_counter();image=rgb(path)
  bank_folder=OUT/info['map_source'] if peak_parent else folder
  with np.load(bank_folder/'demo_bank.npz') as saved:
   if method=='multiscale':
    outputs=[]
    for size,key in [(224,'low_bank'),(448,'high_bank')]:
     f,mask=self._query(path,size);bank=saved[key];index=faiss.IndexFlatL2(384);index.add(bank)
     outputs.append(score_features(index,f,mask,size//14))
    amap,score=fuse_scales(outputs[0][0],outputs[1][0],outputs[0][1],outputs[1][1],float(saved['low_scale']),float(saved['high_scale']))
   else:
    size=448 if (peak_parent or method)=='hi448' else 224
    f,mask=self._query(path,size);bank=saved['bank'];index=faiss.IndexFlatL2(384);index.add(bank)
    amap,score=score_features(index,f,mask,size//14)
  if peak_parent:score=float(amap.max())
  decision='NG' if score>info['image_threshold'] else 'OK'
  result={'input':str(path),'category':category,'method':method,'score':float(score),'threshold':info['image_threshold'],'decision':decision,'pixel_threshold':info['pixel_threshold'],'seconds':time.perf_counter()-start}
  if review_quantile is not None:
   with np.load(folder/'calibration.npz') as calibration:low=normal_threshold(calibration['scores'],alpha=1-review_quantile)
   result.update(decision=str(decisions(np.array([score]),low,info['image_threshold'])[0]),review_threshold=low,review_quantile=review_quantile)
  return image,amap,result

def panel(image,anomaly,result):
 import matplotlib
 cmap=matplotlib.colormaps['magma']
 # Class-specific fixed calibration scale; no per-image min/max normalization.
 heat=(cmap(np.clip(anomaly/max(result['pixel_threshold']*2,1e-6),0,1))[:,:,:3]*255).astype(np.uint8)
 overlay=(.55*image+.45*heat).astype(np.uint8)
 binary=np.repeat((anomaly>result['pixel_threshold'])[:,:,None]*255,3,axis=2).astype(np.uint8)
 canvas=Image.new('RGB',(1280,500),'#f8fafc');draw=ImageDraw.Draw(canvas)
 font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
 draw.text((25,15),f"{result['category']} | {result['method']} | {result['decision']} | score {result['score']:.4f} / threshold {result['threshold']:.4f}",fill='#183044',font=font)
 for i,(arr,title) in enumerate(zip([image,heat,overlay,binary],['RGB input','Anomaly heatmap','Overlay','Calibrated pixel mask'])):
  canvas.paste(Image.fromarray(arr).resize((288,288)),(25+i*315,95));draw.text((25+i*315,60),title,fill='#183044',font=small)
 draw.text((25,410),Path(result['input']).name+' | '+f"inference {result['seconds']:.3f} s",fill='#183044',font=small)
 footer=(f"REVIEW above {result['review_threshold']:.4f}; NG above {result['threshold']:.4f}. Review is not counted as successful detection." if 'review_threshold' in result else 'Course prototype. Thresholds use normal calibration images; OK does not guarantee defect-free.')
 draw.text((25,450),footer,fill='#394a5c',font=small)
 return canvas

def main():
 p=argparse.ArgumentParser();p.add_argument('--image');p.add_argument('--category',choices=CATEGORIES+['macaroni1','macaroni2','pcb1','pcb2','pcb3','pcb4'],default='candle');p.add_argument('--method',choices=['full','random25','pooled25','coreset25','hi448','multiscale','core224_peak','hi448_peak'],default='hi448_peak');p.add_argument('--output',default=str(OUT/'demo_prediction.png'));p.add_argument('--review-quantile',type=float,choices=[.8,.85,.9,.95]);args=p.parse_args()
 detector=Detector()
 if args.image:
  image,amap,r=detector.infer(Path(args.image),args.category,args.method,args.review_quantile);target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
  panel(image,amap,r).save(target);target.with_suffix('.json').write_text(json.dumps(r,indent=2));np.save(target.with_suffix('.npy'),amap)
  print(json.dumps(r));return
 import tkinter as tk
 from tkinter import ttk,filedialog,messagebox
 window=tk.Tk();window.title('Project 2 — Product inspection')
 category=tk.StringVar(value=args.category);method=tk.StringVar(value=args.method)
 ttk.Label(window,text='Product category').pack();ttk.Combobox(window,textvariable=category,values=CATEGORIES+['macaroni1','macaroni2','pcb1','pcb2','pcb3','pcb4'],state='readonly').pack()
 ttk.Label(window,text='Detector variant').pack();ttk.Combobox(window,textvariable=method,values=['full','random25','pooled25','coreset25','hi448','multiscale','core224_peak','hi448_peak'],state='readonly').pack()
 label=ttk.Label(window);label.pack()
 def choose():
  name=filedialog.askopenfilename(filetypes=[('Images','*.jpg *.jpeg *.png *.bmp')])
  if not name:return
  try:
   im,am,r=detector.infer(Path(name),category.get(),method.get(),args.review_quantile);photo=ImageTk.PhotoImage(panel(im,am,r).resize((1024,400)));label.configure(image=photo);label.image=photo
  except Exception as e:messagebox.showerror('Inference error',str(e))
 ttk.Button(window,text='Choose product image',command=choose).pack();window.mainloop()
if __name__=='__main__':main()
