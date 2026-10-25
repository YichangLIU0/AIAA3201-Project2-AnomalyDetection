"""Runnable local image demo, sharing the evaluated DINO scoring path."""
import argparse,json,time
import numpy as np
import faiss
from PIL import Image,ImageDraw,ImageFont,ImageTk
from run_dino import predict
from common import *
faiss.omp_set_num_threads(4)
class Detector:
 def __init__(self): self.model=load_backbone()
 def infer(self,path,category,method='random25'):
  folder=OUT/('coreset' if method=='coreset25' else 'dinov2')/category/f'8shot_seed0_{method}'
  info=json.loads((folder/'metrics.json').read_text());bank=np.load(folder/'demo_bank.npz')['bank']
  index=faiss.IndexFlatL2(384);index.add(bank)
  start=time.perf_counter();image=rgb(path);x,grid=self.model.prepare_image(image)
  f=self.model.extract_features(x).astype(np.float32);np.random.seed(0)
  mask=self.model.compute_background_mask(f,grid,10,True)
  if not mask.any():mask=np.ones(len(f),dtype=bool)
  maps,scores,_=predict(index,[(f,mask)])
  result={'input':str(path),'category':category,'method':method,'score':float(scores[0]),'threshold':info['image_threshold'],'decision':'NG' if scores[0]>info['image_threshold'] else 'OK','pixel_threshold':info['pixel_threshold'],'seconds':time.perf_counter()-start}
  return image,maps[0],result

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
 draw.text((25,450),'Course prototype. Thresholds use normal calibration images; OK does not guarantee defect-free.',fill='#394a5c',font=small)
 return canvas

def main():
 p=argparse.ArgumentParser();p.add_argument('--image');p.add_argument('--category',choices=CATEGORIES,default='candle');p.add_argument('--method',choices=['full','random25','pooled25','coreset25'],default='random25');p.add_argument('--output',default=str(OUT/'demo_prediction.png'));args=p.parse_args()
 detector=Detector()
 if args.image:
  image,amap,r=detector.infer(Path(args.image),args.category,args.method);target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
  panel(image,amap,r).save(target);target.with_suffix('.json').write_text(json.dumps(r,indent=2));np.save(target.with_suffix('.npy'),amap)
  print(json.dumps(r));return
 import tkinter as tk
 from tkinter import ttk,filedialog,messagebox
 window=tk.Tk();window.title('Project 2 — Product inspection')
 category=tk.StringVar(value=args.category);method=tk.StringVar(value=args.method)
 ttk.Label(window,text='Product category').pack();ttk.Combobox(window,textvariable=category,values=CATEGORIES,state='readonly').pack()
 ttk.Label(window,text='Memory bank').pack();ttk.Combobox(window,textvariable=method,values=['full','random25','pooled25','coreset25'],state='readonly').pack()
 label=ttk.Label(window);label.pack()
 def choose():
  name=filedialog.askopenfilename(filetypes=[('Images','*.jpg *.jpeg *.png *.bmp')])
  if not name:return
  try:
   im,am,r=detector.infer(Path(name),category.get(),method.get());photo=ImageTk.PhotoImage(panel(im,am,r).resize((1024,400)));label.configure(image=photo);label.image=photo
  except Exception as e:messagebox.showerror('Inference error',str(e))
 ttk.Button(window,text='Choose product image',command=choose).pack();window.mainloop()
if __name__=='__main__':main()
