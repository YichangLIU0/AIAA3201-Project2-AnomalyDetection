import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'coursework'))
from selective import decisions
class SelectiveTests(unittest.TestCase):
 def test_boundaries_and_fixed_ng(self):
  x=np.array([.1,.2,.3,.4,.5]);d=decisions(x,.2,.4)
  np.testing.assert_array_equal(d,['OK','OK','REVIEW','REVIEW','NG'])
  d2=decisions(x,.1,.4)
  np.testing.assert_array_equal(d=='NG',d2=='NG')
 def test_invalid_order(self):
  with self.assertRaises(ValueError):decisions(np.array([.1]),.5,.4)
if __name__=='__main__':unittest.main()
