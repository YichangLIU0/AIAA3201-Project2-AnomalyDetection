"""Resolution-aware frozen-feature scoring and reference-only scale fusion."""
import numpy as np,faiss,cv2
from scipy.ndimage import gaussian_filter

def reference_groups(indices,patches_per_view):
 return np.asarray(indices,dtype=np.int64)//(8*patches_per_view)

def score_features(index,features,mask,grid,output_size=224):
 features=np.asarray(features,dtype=np.float32);mask=np.asarray(mask,dtype=bool).reshape(-1)
 if len(features)!=grid*grid or len(mask)!=len(features):raise ValueError('Feature grid mismatch')
 if not mask.any():raise ValueError('Empty query mask')
 q=np.ascontiguousarray(features[mask]);faiss.normalize_L2(q)
 distances=index.search(q,1)[0][:,0]
 patch=np.zeros(len(features),np.float32);patch[mask]=np.maximum(distances/2,0)
 k=max(1,int(len(patch)*.01));score=float(np.partition(patch,len(patch)-k)[-k:].mean())
 amap=gaussian_filter(cv2.resize(patch.reshape(grid,grid),(output_size,output_size)),4)
 return amap,score

def fuse_scales(low_maps,high_maps,low_scores,high_scores,low_scale,high_scale):
 if low_scale<=0 or high_scale<=0:raise ValueError('Scales must be positive')
 return np.maximum(low_maps/low_scale,high_maps/high_scale),np.maximum(low_scores/low_scale,high_scores/high_scale)

def fit_reference_scale(bank,groups,reference_queries,grid):
 scores=[]
 for i,(features,mask) in enumerate(reference_queries):
  retained=np.ascontiguousarray(bank[groups!=i]);assert len(retained)>0
  index=faiss.IndexFlatL2(bank.shape[1]);index.add(retained)
  _,score=score_features(index,features,mask,grid);scores.append(score)
 return max(float(np.median(scores)),1e-6),scores
