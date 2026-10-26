import unittest,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'coursework'))
from metrics import pixel_statistics,aupro
class PixelStatisticsTests(unittest.TestCase):
 def test_matches_reference_with_ties(self):
  rng=np.random.default_rng(72)
  for tied in [False,True]:
   scores=rng.random((4,13,17)).astype(np.float32)
   if tied:scores=np.round(scores,1)
   masks=rng.random(scores.shape)<.12
   result=pixel_statistics(scores,masks)
   self.assertAlmostEqual(result[0],roc_auc_score(masks.ravel(),scores.ravel()),places=12)
   self.assertAlmostEqual(result[1],average_precision_score(masks.ravel(),scores.ravel()),places=12)
   self.assertAlmostEqual(result[2],aupro(scores,masks),places=10)
if __name__=='__main__':unittest.main()
