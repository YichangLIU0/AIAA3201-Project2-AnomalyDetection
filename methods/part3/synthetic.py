"""CutPaste-inspired and procedural texture generators using normal images only.

This is an original small augmentation component, not a reproduction of the
CutPaste paper's representation-training pipeline. The DINO backbone stays frozen.
"""
import numpy as np
import cv2
import torch
from torch import nn
from torch.nn import functional as F

CONFIG={
    'version':2,'modes':['cutpaste','texture'],
    'placement':'default foreground estimated from normal-image border Lab; canvas option retained for future ablation',
    'foreground':'Lab chroma distance from border median > max(12,border95th+8); additionally positive luminance contrast for dark backgrounds L<70; opening5, components>=0.1% canvas; no test GT',
    'rectangle_area_fraction':[.02,.08], 'aspect_ratio':[.5,2.],
    'cutpaste':'same-image donor crop, random location; independent RGB gains .6..1.4 and offset -20..20',
    'texture':'procedural coarse RGB noise, linear interpolation to rectangle, blend alpha .7',
    'mask':'only pixels whose uint8 RGB changed, not every pixel of proposed rectangle',
    'patch_targets':'adaptive-average mask coverage on actual extractor patch grid; no GT test masks',
    'head':'1x1 Conv 384->128, ReLU, 1x1 Conv128->1; backbone features detached',
    'loss':'BCEWithLogits; positive weight from synthetic target mass only',
    'status':'generator and head preparation only; real-DINO training/transfer/ablation not yet run',
    'source':'CutPaste inspiration: https://openaccess.thecvf.com/content/CVPR2021/html/Li_CutPaste_Self-Supervised_Learning_for_Anomaly_Detection_and_Localization_CVPR_2021_paper.html',
}


def estimate_foreground(image):
    h,w=image.shape[:2];bw=max(1,int(min(h,w)*.05))
    lab=cv2.cvtColor(image,cv2.COLOR_RGB2LAB).astype(np.float32)
    border=np.concatenate([lab[:bw].reshape(-1,3),lab[-bw:].reshape(-1,3),
                           lab[:,:bw].reshape(-1,3),lab[:,-bw:].reshape(-1,3)])
    median=np.median(border,axis=0)
    threshold=max(12.,float(np.quantile(np.linalg.norm(border[:,1:]-median[1:],axis=1),.95))+8.)
    mask=np.linalg.norm(lab[...,1:]-median[1:],axis=2)>threshold
    if median[0]<70:
        light_threshold=max(25.,float(np.quantile(border[:,0]-median[0],.95))+15.)
        mask|=(lab[...,0]-median[0])>light_threshold
    mask=mask.astype(np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
    keep=np.zeros(n,dtype=bool)
    if n>1:keep[1:]=stats[1:,cv2.CC_STAT_AREA]>=max(9,int(h*w*.001))
    mask=keep[labels]
    if mask.mean()<.005:raise ValueError('Foreground heuristic failed; inspect normal image rather than invent GT')
    return mask


def generate(image, seed, mode='cutpaste', placement='foreground'):
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3 or min(image.shape[:2])<16:
        raise ValueError('Expected RGB uint8 normal image, at least16 pixels per side')
    if mode not in CONFIG['modes']: raise ValueError('Unsupported synthetic mode')
    if placement not in ['foreground','canvas']:raise ValueError('Unsupported placement')
    rng=np.random.default_rng(seed)
    h,w=image.shape[:2]
    area=rng.uniform(.02,.08)*h*w; aspect=rng.uniform(.5,2.)
    ph=min(h,max(2,int(np.sqrt(area/aspect)))); pw=min(w,max(2,int(np.sqrt(area*aspect))))
    y=int(rng.integers(h-ph+1)); x=int(rng.integers(w-pw+1))
    foreground=estimate_foreground(image) if placement=='foreground' else np.ones((h,w),dtype=bool)
    if placement=='foreground':
        ys,xs=np.where(foreground)
        center=int(rng.integers(len(xs)))
        x=int(np.clip(xs[center]-pw//2,0,w-pw));y=int(np.clip(ys[center]-ph//2,0,h-ph))
    original=image[y:y+ph,x:x+pw].astype(np.float32)
    donor=None
    if mode=='cutpaste':
        dy=int(rng.integers(h-ph+1)); dx=int(rng.integers(w-pw+1))
        donor=[dx,dy,pw,ph]
        patch=image[dy:dy+ph,dx:dx+pw].astype(np.float32)
        patch=patch*rng.uniform(.6,1.4,(1,1,3))+rng.uniform(-20,20,(1,1,3))
    else:
        noise=rng.integers(0,256,(8,8,3),dtype=np.uint8).astype(np.float32)
        patch=.7*cv2.resize(noise,(pw,ph),interpolation=cv2.INTER_LINEAR)+.3*original
    result=image.copy()
    proposed=np.clip(patch,0,255).astype(np.uint8)
    selected=foreground[y:y+ph,x:x+pw]
    result[y:y+ph,x:x+pw][selected]=proposed[selected]
    mask=np.any(result!=image,axis=2)
    if not mask.any(): raise ValueError('Synthetic operation changed no pixels')
    return result,mask,{'seed':int(seed),'mode':mode,'placement':placement,
                        'estimated_foreground_fraction':float(foreground.mean()),'target_rectangle_xywh':[x,y,pw,ph],
                        'donor_rectangle_xywh':donor,'changed_pixels':int(mask.sum()),
                        'changed_fraction':float(mask.mean())}


class ScoringHead(nn.Module):
    def __init__(self,channels=384):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(channels,128,1),nn.ReLU(),nn.Conv2d(128,1,1))

    def forward(self,features):
        if features.ndim!=4: raise ValueError('Expected B,C,Hpatch,Wpatch features, not CLS tokens')
        return self.net(features.detach())[:,0]


def head_loss(head,features,pixel_masks):
    if pixel_masks.ndim!=3 or pixel_masks.shape[0]!=features.shape[0]:
        raise ValueError('Expected aligned B,H,W synthetic masks')
    if not torch.isfinite(pixel_masks).all() or pixel_masks.min()<0 or pixel_masks.max()>1:
        raise ValueError('Synthetic target masks must be finite in0..1')
    targets=F.adaptive_avg_pool2d(pixel_masks[:,None].float(),features.shape[-2:])[:,0]
    positives=targets.sum(); negatives=targets.numel()-positives
    if positives<=0 or negatives<=0:
        raise ValueError('A training batch needs both normal and synthetic pixels')
    weight=(negatives/positives).detach()
    return F.binary_cross_entropy_with_logits(head(features),targets,pos_weight=weight)


def frozen_features(extractor,images):
    """B adapter returns spatial features after its exact official preprocessing.

    Input/output geometry and mask projection must be supplied by that adapter;
    this helper does not guess CLS removal, patch arrangement, normalization or crop.
    """
    extractor.requires_grad_(False); extractor.eval()
    with torch.no_grad(): features=extractor(images)
    if not isinstance(features,torch.Tensor) or features.ndim!=4:
        raise ValueError('DINO adapter must return B,C,Hpatch,Wpatch tensor')
    return features.detach()
