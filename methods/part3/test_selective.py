import unittest
import numpy as np
from methods.part3.selective import ReviewPolicy, counts, evaluate_selective


class SelectiveTest(unittest.TestCase):
    def test_review_does_not_become_true_positive(self):
        policy=ReviewPolicy(1.,.2)
        scores=np.array([.5,.9,1.1,1.5])
        labels=np.array([1,1,0,1])
        accepted=policy.decisions(scores)!='REVIEW'
        row=counts(scores,labels,policy,accepted)
        self.assertEqual(row['automatic_fn'],1)
        self.assertEqual(row['automatic_tp'],1)
        self.assertEqual(row['review_anomaly'],1)
        self.assertEqual(row['anomaly_accounting_total'],3)
        self.assertAlmostEqual(row['automatic_fnr'],1/3)
        self.assertAlmostEqual(row['risk'],.5)

    def test_test_labels_do_not_change_calibration_or_decisions(self):
        cal=np.linspace(0.,1.,20); scores=np.array([.4,.9,.95,1.2])
        first=evaluate_selective(cal,scores,[0,1,0,1])
        second=evaluate_selective(cal,scores,[1,0,1,0])
        self.assertEqual(first['policy'],second['policy'])
        self.assertEqual([r['coverage'] for r in first['risk_coverage']],
                         [r['coverage'] for r in second['risk_coverage']])

    def test_tied_confidence_is_grouped_and_zero_coverage_has_no_risk(self):
        cal=np.zeros(20)
        result=evaluate_selective(cal,[-1.,0.,0.,1.],[0,0,1,1])
        self.assertEqual([r['coverage'] for r in result['risk_coverage']],[0.,.5,1.])
        self.assertIsNone(result['risk_coverage'][0]['risk'])
        row=counts([0.,0.],[0,1],ReviewPolicy(0.,0.),[False,False])
        self.assertIsNone(row['risk']); self.assertEqual(row['automatic_tp'],0)

    def test_fixed_fpr_is_attainable_and_not_deploy_threshold(self):
        cal=np.linspace(0,1,20)
        scores=np.r_[np.zeros(20),np.ones(5)]
        labels=np.r_[np.zeros(20,dtype=int),np.ones(5,dtype=int)]
        result=evaluate_selective(cal,scores,labels)
        diagnostic=result['fixed_fpr_test_roc']
        self.assertEqual(diagnostic['attained_fpr'],0.)
        self.assertEqual(diagnostic['fnr'],0.)
        self.assertNotEqual(diagnostic['threshold_evaluation_only'],result['policy']['threshold'])


if __name__=='__main__': unittest.main()
