"""Additional-category high-resolution errors for visual review, selected by filename."""
from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'coursework'))
from common import OUT,DATA,rgb
records=[]
for pair in [('macaroni1','macaroni2'),('pcb1','pcb2'),('pcb3','pcb4')]:
 examples=[]
 for category in pair:
  folder=OUT/'lowmiss'/category/'8shot_seed0_hi448';m=json.loads((folder/'metrics.json').read_text())
  with np.load(folder/'predictions.npz') as d:
   y=d['masks'].reshape(len(d['scores']),-1).any(1);pred=d['scores']>m['image_threshold']
   fp=np.flatnonzero(~y&pred);fn=np.flatnonzero(y&~pred);indices=[]
   for values in [fp,fn]:
    if len(values):indices.append(int(values[0]))
   for i in np.flatnonzero(y!=pred):
    if len(indices)>=2:break
    if int(i) not in indices:indices.append(int(i))
   for i in indices:
    record={'category':category,'method':'hi448','seed':0,'input':str(d['paths'][i]),'type':'FN' if y[i] else 'FP','score':float(d['scores'][i]),'threshold':m['image_threshold'],'defect_area_fraction':float(d['masks'][i].mean()),'selection':'first FP and FN by filename, fill a shortage with other errors'}
    records.append(record);examples.append((rgb(DATA/record['input']),d['masks'][i],d['maps'][i]/m['pixel_threshold'],record))
 if not examples:continue
 fig,axes=plt.subplots(len(examples),3,figsize=(6,2*len(examples)),squeeze=False)
 for row,(im,gt,am,record) in enumerate(examples):
  for col,arr in enumerate([im,gt,am]):
   ax=axes[row,col];ax.imshow(arr,cmap='gray' if col==1 else 'magma',vmin=0,vmax=1 if col==1 else 2) if col else ax.imshow(arr);ax.axis('off')
  axes[row,0].set_title(record['category']+' '+record['type'],fontsize=10)
  axes[row,1].set_title('GT',fontsize=10);axes[row,2].set_title(f"score {record['score']:.3f} / {record['threshold']:.3f}",fontsize=9)
 fig.tight_layout();fig.savefig(r/'reports/figures'/('lowmiss_errors_'+pair[0]+'.png'),dpi=140);plt.close(fig)
(OUT/'lowmiss_additional_errors.json').write_text(json.dumps(records,indent=2));print('ADDITIONAL_ERROR_REVIEW',len(records))
