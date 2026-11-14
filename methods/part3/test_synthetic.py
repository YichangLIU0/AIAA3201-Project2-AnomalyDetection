import unittest
import numpy as np
import torch
from torch import nn
from methods.part3.synthetic import generate,estimate_foreground,ScoringHead,head_loss,frozen_features


class SyntheticTest(unittest.TestCase):
    def test_exact_mask_determinism_and_input_unchanged(self):
        image=np.random.default_rng(8).integers(0,256,(70,98,3),dtype=np.uint8)
        backup=image.copy()
        for mode in ['cutpaste','texture']:
            modified,mask,meta=generate(image,4,mode,placement='canvas')
            again,againmask,_=generate(image,4,mode,placement='canvas')
            np.testing.assert_array_equal(image,backup)
            np.testing.assert_array_equal(mask,np.any(modified!=image,axis=2))
            np.testing.assert_array_equal(modified,again)
            np.testing.assert_array_equal(mask,againmask)
            self.assertGreater(meta['changed_pixels'],0)

    def test_default_synthesis_does_not_paint_estimated_background(self):
        image=np.full((140,196,3),20,dtype=np.uint8)
        image[40:100,60:136]=[220,160,100]
        foreground=estimate_foreground(image)
        for mode in ['cutpaste','texture']:
            changed,mask,meta=generate(image,3,mode)
            self.assertFalse(np.any(mask&~foreground))
            self.assertGreater(int(mask.sum()),0)
            np.testing.assert_array_equal(changed[~foreground],image[~foreground])

    def test_only_head_changes_in_optimization_step(self):
        torch.manual_seed(1)
        extractor=nn.Conv2d(3,384,14,14)
        before={k:v.detach().clone() for k,v in extractor.state_dict().items()}
        head=ScoringHead(); initial=head.net[0].weight.detach().clone()
        images=torch.rand(2,3,56,84)
        features=frozen_features(extractor,images)
        masks=torch.zeros(2,56,84);masks[1,14:42,14:42]=1.
        optimizer=torch.optim.Adam(head.parameters(),lr=.001)
        loss=head_loss(head,features,masks);loss.backward();optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(torch.equal(initial,head.net[0].weight))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in extractor.parameters()))
        self.assertTrue(all(torch.equal(before[k],v) for k,v in extractor.state_dict().items()))
        # This Conv2d fixture checks gradient isolation, NOT DINO representation quality.


if __name__=='__main__':unittest.main()
