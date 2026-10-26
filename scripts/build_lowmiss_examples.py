"""Fixed-rule paired low-miss examples; no best-looking manual selection."""
from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'coursework'))
from common import OUT,DATA,rgb
fig,axes=plt.subplots(4,5,figsize=(10,8));records=[]
for row,category in enumerate(['candle','capsules','pcb1','pcb4']):
 low=OUT/'coreset'/category/'8shot_seed0_coreset25';high=OUT/'lowmiss'/category/'8shot_seed0_hi448'
 a=json.loads((low/'metrics.json').read_text());b=json.loads((high/'metrics.json').read_text())
 with np.load(low/'predictions.npz') as old,np.load(high/'predictions.npz') as new:
  assert np.array_equal(old['paths'],new['paths'])
  y=old['masks'].reshape(len(old['scores']),-1).any(1)
  op=old['scores']>a['image_threshold'];npred=new['scores']>b['image_threshold']
  recovered=np.flatnonzero(y&~op&npred)
  introduced=np.flatnonzero(y&op&~npred)
  selection='first recovered FN' if len(recovered) else ('first introduced FN' if len(introduced) else 'first anomalous image')
  idx=int((recovered if len(recovered) else introduced if len(introduced) else np.flatnonzero(y))[0])
  im=rgb(DATA/str(old['paths'][idx]));mask=old['masks'][idx];yy,xx=np.where(mask)
  x0,x1=max(0,int(xx.min())-8),min(224,int(xx.max())+9);y0,y1=max(0,int(yy.min())-8),min(224,int(yy.max())+9)
  arrays=[im,mask,old['maps'][idx]/a['pixel_threshold'],new['maps'][idx]/b['pixel_threshold'],im[y0:y1,x0:x1]]
  titles=[category+' / RGB','Ground truth',f"224: {'NG' if op[idx] else 'OK'}",f"448: {'NG' if npred[idx] else 'OK'}",'Defect zoom']
  for col,(arr,title) in enumerate(zip(arrays,titles)):
   ax=axes[row,col]
   if col in [2,3]:ax.imshow(arr,cmap='magma',vmin=0,vmax=2)
   elif col==1:ax.imshow(arr,cmap='gray',vmin=0,vmax=1)
   else:ax.imshow(arr)
   ax.set_title(title,fontsize=9);ax.axis('off')
  axes[row,0].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='yellow',linewidth=1))
  records.append({'category':category,'seed':0,'input':str(old['paths'][idx]),'selection_rule':selection,'old_score':float(old['scores'][idx]),'new_score':float(new['scores'][idx]),'old_threshold':a['image_threshold'],'new_threshold':b['image_threshold'],'defect_area_fraction':float(mask.mean()),'crop_xyxy':[x0,y0,x1,y1]})
fig.tight_layout();fig.savefig(r/'reports/figures/lowmiss_examples.pdf');fig.savefig(r/'reports/figures/lowmiss_examples.png',dpi=150);plt.close(fig)
(OUT/'lowmiss_example_records.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
