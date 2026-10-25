import json,csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
FIG=ROOT/'reports/figures';FIG.mkdir(parents=True,exist_ok=True)
rows=[]
for family in ['dinov2','baselines','anomalyclip','coreset']:
 for p in (OUT/family).rglob('metrics.json') if (OUT/family).exists() else []:
  x=json.loads(p.read_text());x['family']=family;rows.append(x)
(OUT/'all_metrics.json').write_text(json.dumps(rows,indent=2))
keys=['image_auroc','image_ap','pixel_auroc','pixel_ap','aupro_03','image_fpr','image_fnr']
summary=[]
for method in sorted({x['method'] for x in rows}):
 for shot in sorted({x['shots'] for x in rows if x['method']==method}):
  rr=[x for x in rows if x['method']==method and x['shots']==shot]
  seeds=sorted({x['seed'] for x in rr})
  s={'method':method,'shots':shot,'runs':len(rr),'categories':len({x['category'] for x in rr})}
  for key in keys:
   macro=[np.mean([x[key] for x in rr if x['seed']==seed]) for seed in seeds]
   s[key]=float(np.mean(macro));s[key+'_std']=float(np.std(macro,ddof=1)) if len(macro)>1 else 0.
  summary.append(s)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
with (OUT/'summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(10,2.8))
for ax,key,title in zip(axs,['image_auroc','pixel_ap','aupro_03'],['Image AUROC','Pixel AP','AUPRO @ FPR 0.3']):
 for c in CATEGORIES:
  means=[];stds=[]
  for shot in [1,2,4,8]:
   a=[x[key] for x in rows if x['method']=='full' and x['category']==c and x['shots']==shot]
   means.append(np.mean(a));stds.append(np.std(a,ddof=1))
  ax.errorbar([1,2,4,8],means,yerr=stds,marker='o',markersize=3,capsize=2,label=c)
 ax.set_title(title);ax.set_xlabel('Normal reference images');ax.set_xticks([1,2,4,8]);ax.grid(alpha=.2)
axs[-1].legend(fontsize=7,loc='lower right');fig.tight_layout();fig.savefig(FIG/'shot_curves.pdf');fig.savefig(FIG/'shot_curves.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(9,2.6));methods=['full','random25','pooled25'];colors=['#254e70','#388f82','#cf7354']
for ax,key,title,mult in zip(axs,['bank_bytes','scoring_seconds_mean','image_auroc'],['Memory bank (MiB)','Matching + map (ms)','Image AUROC'],[1/1024**2,1000,1]):
 vals=[np.mean([x[key] for x in rows if x['method']==m and x['shots']==8])*mult for m in methods]
 ax.bar(methods,vals,color=colors);ax.set_title(title);ax.tick_params(axis='x',labelrotation=15)
 for i,v in enumerate(vals):ax.text(i,v,f'{v:.3f}' if key=='image_auroc' else f'{v:.1f}',ha='center',va='bottom',fontsize=8)
 ax.set_ylim(0,max(vals)*1.2)
fig.tight_layout();fig.savefig(FIG/'compression.pdf');fig.savefig(FIG/'compression.png',dpi=160);plt.close(fig)
errors=[]
for c in CATEGORIES:
 folder=OUT/'dinov2'/c/'8shot_seed0_full';metric=json.loads((folder/'metrics.json').read_text())
 with np.load(folder/'predictions.npz') as d:
  maps=d['maps'];masks=d['masks'];scores=d['scores'];paths=d['paths']
  y=masks.reshape(len(masks),-1).any(1);pred=scores>metric['image_threshold']
  fp=np.flatnonzero(pred&~y).tolist();fn=np.flatnonzero(~pred&y).tolist()
  chosen=fp[:2]+fn[:2]
  chosen+=(fp[2:]+fn[2:])[:max(0,4-len(chosen))]
  fig,axs=plt.subplots(len(chosen),4,figsize=(9,2.3*len(chosen)),squeeze=False)
  for row,i in enumerate(chosen):
   ident=f'{c}_{row+1}';image=rgb(DATA/paths[i]);thresholded=maps[i]>metric['pixel_threshold']
   for ax,arr,title in zip(axs[row],[image,masks[i],maps[i],thresholded],['RGB','Ground truth','Anomaly map','Fixed-threshold mask']):
    ax.imshow(arr,cmap=None if arr.ndim==3 else 'magma',vmin=None if arr.ndim==3 else 0,vmax=None if arr.ndim==3 else (max(metric['pixel_threshold']*2,.1) if title=='Anomaly map' else 1));ax.axis('off')
    if row==0:ax.set_title(title)
   typ='FP' if not y[i] else 'FN';axs[row,0].set_ylabel(ident)
   axs[row,0].set_title(f'{ident} {typ}: {scores[i]:.3f} / {metric["image_threshold"]:.3f}',fontsize=9)
   errors.append({'id':ident,'category':c,'path':str(paths[i]),'error':typ,'score':float(scores[i]),'threshold':metric['image_threshold'],'mask_area_fraction':float(masks[i].mean()),'cause':'pending visual inspection','interpretation':'hypothesis, not causal proof'})
  fig.tight_layout();fig.savefig(FIG/f'errors_{c}.png',dpi=130);plt.close(fig)
(OUT/'failure_cases.json').write_text(json.dumps(errors,indent=2))
# Single representative per class; ground truth and prediction are separately shown.
fig,axs=plt.subplots(6,4,figsize=(8,11))
for row,c in enumerate(CATEGORIES):
 folder=OUT/'dinov2'/c/'8shot_seed0_full';m=json.loads((folder/'metrics.json').read_text())
 with np.load(folder/'predictions.npz') as d:
  y=d['masks'].reshape(len(d['scores']),-1).any(1);i=np.flatnonzero(y)[0]
  arrays=[rgb(DATA/d['paths'][i]),d['masks'][i],d['maps'][i],d['maps'][i]>m['pixel_threshold']]
  for ax,arr in zip(axs[row],arrays):ax.imshow(arr,cmap='magma' if arr.ndim==2 else None);ax.axis('off')
  axs[row,0].set_title(c,fontsize=9)
for ax,title in zip(axs[0],['RGB','GT','Score map','Thresholded']):ax.set_title(title)
fig.tight_layout();fig.savefig(FIG/'qualitative.pdf');fig.savefig(FIG/'qualitative.png',dpi=130);plt.close(fig)
print('Summary groups',len(summary),'failure cases',len(errors),flush=True)
