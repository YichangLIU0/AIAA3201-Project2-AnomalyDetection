import unittest,sys
from pathlib import Path
import numpy as np,faiss
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'coursework'))
from multiscale import score_features,fuse_scales,reference_groups,fit_reference_scale
class MultiscaleTests(unittest.TestCase):
 def test_grid_and_mask(self):
  bank=np.array([[1.,0.]],dtype=np.float32);index=faiss.IndexFlatL2(2);index.add(bank)
  f=np.tile(bank,(16,1));f[-1]=[0,1];mask=np.ones(16,dtype=bool)
  amap,s=score_features(index,f,mask,4);self.assertAlmostEqual(s,1.,places=6);self.assertEqual(amap.shape,(224,224))
  mask[-1]=False;_,s=score_features(index,f,mask,4);self.assertAlmostEqual(s,0.,places=6)
 def test_scale_invariance(self):
  a=np.array([1.,2.]);b=np.array([3.,1.]);m1=np.ones((2,4,4));m2=m1*2
  x,y=fuse_scales(m1,m2,a,b,2.,3.)
  x2,y2=fuse_scales(m1*5,m2*7,a*5,b*7,10.,21.)
  np.testing.assert_allclose(x,x2);np.testing.assert_allclose(y,y2)
 def test_reference_groups(self):
  np.testing.assert_array_equal(reference_groups(np.array([0,2047,2048,4095]),256),[0,0,1,1])
 def test_leave_one_image_out_removes_all_rotated_views(self):
  # A reference would match itself perfectly if even one rotated view leaked.
  bank=np.repeat(np.eye(2,dtype=np.float32),8,axis=0)
  groups=np.repeat(np.arange(2),8)
  queries=[(np.eye(2,dtype=np.float32)[i:i+1],np.ones(1,dtype=bool)) for i in range(2)]
  scale,scores=fit_reference_scale(bank,groups,queries,grid=1)
  np.testing.assert_allclose(scores,[1.,1.],atol=1e-6)
  self.assertAlmostEqual(scale,1.,places=6)
if __name__=='__main__':unittest.main()
