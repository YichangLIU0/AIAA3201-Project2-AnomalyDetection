import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'coursework'))
from metrics import evaluate,normal_threshold
class MetricTests(unittest.TestCase):
 def test_perfect(self):
  gt=np.zeros((2,4,4),dtype=bool);gt[1,1:3,1:3]=True
  r=evaluate(gt.astype(float),gt,np.array([0.,1.]),0.5)
  for k in ['image_auroc','image_ap','pixel_auroc','pixel_ap','aupro_03']: self.assertAlmostEqual(r[k],1.)
  self.assertEqual(r['image_fpr'],0.)
 def test_tied(self):
  gt=np.zeros((2,4,4),dtype=bool);gt[1,1:3,1:3]=True
  r=evaluate(np.ones(gt.shape),gt,np.ones(2),1.)
  self.assertAlmostEqual(r['pixel_auroc'],.5)
  self.assertAlmostEqual(r['aupro_03'],.15)
  self.assertEqual(r['image_fpr'],0.)
 def test_calibration(self):
  self.assertEqual(normal_threshold(np.arange(40.)),38.)
if __name__=='__main__': unittest.main()
