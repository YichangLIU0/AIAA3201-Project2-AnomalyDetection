"""Aggregate the fixed coreset extension, preserving paired seed/category evidence."""
from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'coursework'))
from common import OUT,DATA,CATEGORIES,rgb
rows=[json.loads(p.read_text()) for c in CATEGORIES for p in (OUT/'coreset'/c).rglob('metrics.json')]
diags=[json.loads(p.read_text()) for c in CATEGORIES for p in (OUT/'coreset'/c).rglob('diagnosis.json')]
assert len(rows)==len(diags)==18,(len(rows),len(diags))
methods=['full','random25','pooled25','coreset25'];metrics={}
for method in methods:
 family='coreset' if method=='coreset25' else 'dinov2'
 rr=[json.loads((OUT/family/c/f'8shot_seed{s}_{method}'/'metrics.json').read_text()) for c in CATEGORIES for s in range(3)]
 record={}
 for key in ['image_auroc','pixel_ap','aupro_03','image_fpr','image_fnr','bank_build_seconds','scoring_seconds_mean','bank_bytes']:
  macro=[np.mean([x[key] for x in rr if x['seed']==seed]) for seed in range(3)]
  record[key]=float(np.mean(macro));record[key+'_std']=float(np.std(macro,ddof=1))
 if method!='full':
  for key in ['mean_cosine_distance','p95_cosine_distance','max_cosine_distance']:
   record[key]=float(np.mean([x['coverage'][method][key] for x in diags]))
 metrics[method]=record
paired={}
for method in ['full','random25','pooled25']:
 keys=['old_fp','new_fp','fp_fixed','fp_introduced','old_fn','new_fn','fn_fixed','fn_introduced']
 paired[method]={k:int(sum(d['comparisons'][method][k] for d in diags)) for k in keys}
 paired[method]['count_scope']='image-seed events, not unique images; 1080 normal and 1800 anomalous evaluations'
per_class={}
for c in CATEGORIES:
 per_class[c]={m:{k:float(np.mean([json.loads((OUT/('coreset' if m=='coreset25' else 'dinov2')/c/f'8shot_seed{s}_{m}'/'metrics.json').read_text())[k] for s in range(3)])) for k in ['image_auroc','pixel_ap','image_fpr','image_fnr']} for m in methods}
result={'metrics':metrics,'paired_decisions':paired,'per_category':per_class,'selection_peak_gpu_allocated_mb_mean':float(np.mean([x['selection_peak_gpu_allocated_mb'] for x in rows]))}
(OUT/'coreset_extension_summary.json').write_text(json.dumps(result,indent=2))
fig,axs=plt.subplots(1,3,figsize=(10,3));compress=['random25','pooled25','coreset25'];colors=['#388f82','#cf7354','#7655a5']
for ax,key,label in zip(axs,['max_cosine_distance','image_auroc','image_fnr'],['Mean worst reference distance','Image AUROC','FNR at calibrated threshold']):
 vals=[metrics[m][key] for m in compress];ax.bar(compress,vals,color=colors);ax.set_title(label,fontsize=10);ax.tick_params(axis='x',labelrotation=20);ax.spines[['top','right']].set_visible(False)
 for i,v in enumerate(vals):ax.text(i,v,f'{v:.3f}',ha='center',va='bottom',fontsize=9)
 ax.set_ylim(0,max(vals)*1.2)
fig.tight_layout();fig.savefig(r/'reports/figures/coreset_diagnosis.pdf');fig.savefig(r/'reports/figures/coreset_diagnosis.png',dpi=160);plt.close(fig)
# Select one example of each decision transition, deterministic category/filename order.
cases=[]
for tag in ['fp_fixed','fp_introduced','fn_fixed','fn_introduced']:
 for c in CATEGORIES:
  oldbase=OUT/'dinov2'/c/'8shot_seed0_random25';newbase=OUT/'coreset'/c/'8shot_seed0_coreset25'
  with np.load(oldbase/'predictions.npz') as a,np.load(newbase/'predictions.npz') as b:
   old=json.loads((oldbase/'metrics.json').read_text());new=json.loads((newbase/'metrics.json').read_text())
   y=a['masks'].reshape(len(a['scores']),-1).any(1);op=a['scores']>old['image_threshold'];npred=b['scores']>new['image_threshold']
   choices={'fp_fixed':~y&op&~npred,'fp_introduced':~y&~op&npred,'fn_fixed':y&~op&npred,'fn_introduced':y&op&~npred}
   ids=np.flatnonzero(choices[tag])
   if not len(ids):continue
   i=int(ids[0]);path=str(a['paths'][i]);cases.append({'transition':tag,'category':c,'path':path,'old_score':float(a['scores'][i]),'new_score':float(b['scores'][i]),'old_threshold':old['image_threshold'],'new_threshold':new['image_threshold']})
   fig,axs=plt.subplots(1,4,figsize=(10,3))
   panels=[rgb(DATA/path),a['masks'][i],a['maps'][i]/max(old['pixel_threshold'],1e-8),b['maps'][i]/max(new['pixel_threshold'],1e-8)]
   for ax,im,title in zip(axs,panels,['Input','GT','Random / pixel threshold','Coreset / pixel threshold']):
    ax.imshow(im,cmap=None if im.ndim==3 else 'magma',vmin=None if im.ndim==3 else 0,vmax=None if im.ndim==3 else (1 if title=='GT' else 2));ax.set_title(title,fontsize=9);ax.axis('off')
   fig.suptitle(f'{tag}: {c} | image score/threshold {a["scores"][i]:.3f}/{old["image_threshold"]:.3f} to {b["scores"][i]:.3f}/{new["image_threshold"]:.3f}',fontsize=10)
   fig.tight_layout();fig.savefig(r/f'reports/figures/coreset_{tag}.png',dpi=140);plt.close(fig);break
(OUT/'coreset_transition_examples.json').write_text(json.dumps(cases,indent=2))
print(json.dumps(result,indent=2))
