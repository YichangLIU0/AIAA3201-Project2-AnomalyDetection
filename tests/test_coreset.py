import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'coursework'))
from coreset import greedy_coreset_indices

class CoresetTests(unittest.TestCase):
 def test_duplicates_still_produce_unique_budget(self):
  x=np.ones((9,4),np.float32)
  selected=greedy_coreset_indices(x,7,seed=3,projection_dim=4,device='cpu')
  self.assertEqual(len(set(selected)),7)
  self.assertTrue(np.all((selected>=0)&(selected<len(x))))
 def test_deterministic_nested_cover(self):
  x=np.array([[0,0],[1,0],[2,0],[9,0],[10,0],[11,0]],np.float32)
  a=greedy_coreset_indices(x,2,seed=3,projection_dim=2,device='cpu')
  b=greedy_coreset_indices(x,3,seed=3,projection_dim=2,device='cpu')
  np.testing.assert_array_equal(a,b[:2])
  np.testing.assert_array_equal(b,greedy_coreset_indices(x,3,seed=3,projection_dim=2,device='cpu'))
  distance=lambda ids:np.min(((x[:,None]-x[ids][None])**2).sum(-1),axis=1)
  self.assertLessEqual(distance(b).max(),distance(a).max())
  self.assertTrue(any(x[a,0]<3) and any(x[a,0]>8))
 def test_projected_selection_reproducible_and_original_input_unchanged(self):
  x=np.random.default_rng(20).normal(size=(30,12)).astype(np.float32);before=x.copy()
  a=greedy_coreset_indices(x,8,seed=42,projection_dim=3,device='cpu')
  b=greedy_coreset_indices(x,8,seed=42,projection_dim=3,device='cpu')
  np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(x,before)
  self.assertEqual(len(set(a)),8)
 def test_invalid_budget(self):
  with self.assertRaises(ValueError):greedy_coreset_indices(np.ones((3,2),np.float32),4,device='cpu')
if __name__=='__main__':unittest.main()
