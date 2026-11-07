import unittest
from unittest.mock import patch
import cv2
import numpy as np
from methods.part1.ecc import align_reference


def scene():
    image = np.zeros((252, 420, 3), np.uint8)
    cv2.rectangle(image, (75, 40), (195, 145), (90, 130, 210), -1)
    cv2.circle(image, (300, 170), 35, (230, 150, 80), -1)
    cv2.line(image, (70, 210), (350, 35), (160, 200, 110), 7)
    return cv2.GaussianBlur(image, (5, 5), 1.)


class ECCTests(unittest.TestCase):
    def test_translation_direction_and_full_resolution_scaling(self):
        ref = scene()
        query = cv2.warpAffine(ref, np.float32([[1, 0, 13], [0, 1, -9]]), (420, 252))
        aligned, trace = align_reference(query, ref)
        self.assertEqual(trace['status'], 'aligned')
        self.assertAlmostEqual(trace['warp_query_to_reference'][0][2], -13, delta=1.2)
        self.assertAlmostEqual(trace['warp_query_to_reference'][1][2], 9, delta=1.2)
        self.assertLess(np.abs(query.astype(float)-aligned).mean(), np.abs(query.astype(float)-ref).mean()*.12)

    def test_rotation_and_border_fallback_do_not_copy_query(self):
        ref = scene()
        transform = cv2.getRotationMatrix2D((210, 126), 5, 1)
        query = cv2.warpAffine(ref, transform, (420, 252))
        aligned, trace = align_reference(query, ref)
        self.assertEqual(trace['status'], 'aligned')
        self.assertLess(np.abs(query.astype(float)-aligned).mean(), np.abs(query.astype(float)-ref).mean()*.2)
        warp = np.array(trace['warp_query_to_reference'], np.float32)
        valid = cv2.warpAffine(np.ones((252, 420),np.float32),warp,(420,252),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)>=.999
        self.assertTrue(np.array_equal(aligned[~valid],ref[~valid]))

    def test_nonconvergence_and_constant_fallback(self):
        ref = scene(); query = np.roll(ref, 3, axis=0)
        with patch('methods.part1.ecc.cv2.findTransformECC',side_effect=cv2.error('synthetic solver failure')):
            aligned, trace = align_reference(query, ref)
        self.assertEqual(trace['reason'],'ecc_nonconvergence')
        self.assertTrue(np.array_equal(aligned,ref))
        _, trace = align_reference(np.zeros_like(ref),ref)
        self.assertEqual(trace['reason'],'constant_image')

    def test_excessive_transform_rejected(self):
        ref = scene()
        with patch('methods.part1.ecc.cv2.findTransformECC',return_value=(.9,np.float32([[1,0,200],[0,1,0]]))):
            aligned, trace = align_reference(ref,ref)
        self.assertEqual(trace['reason'],'transform_out_of_bounds')
        self.assertTrue(np.array_equal(aligned,ref))


if __name__=='__main__':
    unittest.main()
