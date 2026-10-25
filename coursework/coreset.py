"""Projected greedy k-center selection, inspired by PatchCore; no target labels used."""
import numpy as np
import torch

def greedy_coreset_indices(bank,count,seed=0,projection_dim=128,device='cuda'):
 bank=np.asarray(bank,dtype=np.float32)
 if bank.ndim!=2 or not 1<=count<=len(bank):raise ValueError('count must be in [1, number of vectors]')
 if projection_dim<1 or not np.isfinite(bank).all():raise ValueError('finite features and positive projection dimension required')
 with torch.inference_mode():
  x=torch.as_tensor(bank,device=device)
  if projection_dim<x.shape[1]:
   rng=np.random.default_rng(seed)
   projection=rng.standard_normal((x.shape[1],projection_dim)).astype(np.float32)/np.sqrt(projection_dim)
   x=x@torch.as_tensor(projection,device=device)
  x=x-x.mean(0,keepdim=True)
  norm=(x*x).sum(1)
  nearest=torch.full((len(x),),float('inf'),device=device)
  used=torch.zeros(len(x),dtype=torch.bool,device=device)
  selected=[];index=int(norm.argmax())
  for _ in range(count):
   selected.append(index);used[index]=True
   distances=(norm+norm[index]-2*torch.mv(x,x[index])).clamp_min_(0)
   nearest=torch.minimum(nearest,distances)
   nearest.masked_fill_(used,-1.)
   index=int(nearest.argmax())
 return np.asarray(selected,dtype=np.int64)
