"""Shared metrics. No test-label threshold optimization."""
import math
import numpy as np
from scipy.ndimage import label
from sklearn.metrics import roc_auc_score,average_precision_score

def normal_threshold(scores,alpha=.05):
 s=np.sort(np.asarray(scores,dtype=float))
 if len(s)==0 or not np.isfinite(s).all(): raise ValueError('Invalid normal calibration scores')
 k=math.ceil((len(s)+1)*(1-alpha))
 return float(s[k-1]) if k<=len(s) else float('inf')

def aupro(maps,masks,limit=.3):
 weights=np.zeros(masks.shape,dtype=np.float32); regions=0
 for i,mask in enumerate(masks):
  labs,n=label(mask)
  sizes=np.bincount(labs.ravel())
  for j in range(1,n+1): weights[i][labs==j]=1./sizes[j]
  regions+=n
 if regions==0 or (~masks).sum()==0: raise ValueError('AUPRO requires positive regions and background')
 scores=maps.ravel();order=np.argsort(-scores,kind='stable')
 s=scores[order]
 ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
 fpr=np.r_[0.,np.cumsum((~masks).ravel()[order],dtype=np.float64)[ends]/(~masks).sum()]
 pro=np.r_[0.,np.cumsum(weights.ravel()[order],dtype=np.float64)[ends]/regions]
 # Preserve vertical steps (same FPR, increasing PRO); interpolate only the crossing segment.
 k=np.searchsorted(fpr,limit,side='right')
 x=fpr[:k];y=pro[:k]
 if x[-1]<limit:
  yend=y[-1]+(pro[k]-y[-1])*(limit-x[-1])/(fpr[k]-x[-1])
  x=np.r_[x,limit];y=np.r_[y,yend]
 return float(np.trapz(y,x)/limit)

def evaluate(maps,masks,scores,threshold):
 maps=np.asarray(maps,dtype=np.float32);masks=np.asarray(masks,dtype=bool);scores=np.asarray(scores)
 assert maps.shape==masks.shape and len(scores)==len(maps)
 if not np.isfinite(maps).all() or not np.isfinite(scores).all(): raise ValueError('Nonfinite predictions')
 y=masks.reshape(len(masks),-1).any(1);pred=scores>threshold
 return {'image_auroc':float(roc_auc_score(y,scores)),'image_ap':float(average_precision_score(y,scores)),
 'image_fpr':float(pred[~y].mean()),'image_fnr':float((~pred[y]).mean()),
 'pixel_auroc':float(roc_auc_score(masks.ravel(),maps.ravel())),
 'pixel_ap':float(average_precision_score(masks.ravel(),maps.ravel())),
 'aupro_03':aupro(maps,masks),'image_threshold':float(threshold),'n_test':len(scores)}
