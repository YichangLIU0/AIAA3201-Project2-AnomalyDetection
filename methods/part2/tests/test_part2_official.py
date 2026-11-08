"""CPU-only checks for the Part 2 pipeline pieces (no torch, no data needed).
Run:  python -m unittest discover -s methods/part2/tests -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]  # methods/part2
sys.path.insert(0, str(HERE))
import make_temp_manifest as mtm          # noqa: E402
import run_dino_official as rdo           # noqa: E402


def fake_visa(root, category='candle', n_train=60, n_good=3, n_bad=3):
    for sub in ['train/good', 'test/good', 'test/bad', 'ground_truth/bad']:
        (root / category / sub).mkdir(parents=True, exist_ok=True)
    img = np.zeros((40, 60, 3), np.uint8)
    for i in range(n_train):
        cv2.imwrite(str(root / category / 'train/good' / f'{i:03d}.JPG'), img)
    for i in range(n_good):
        cv2.imwrite(str(root / category / 'test/good' / f'{i:03d}.JPG'), img)
    for i in range(n_bad):
        cv2.imwrite(str(root / category / 'test/bad' / f'{i:03d}.JPG'), img)
        m = np.zeros((40, 60), np.uint8); m[10:20, 10:30] = 255
        cv2.imwrite(str(root / category / 'ground_truth/bad' / f'{i:03d}.png'), m)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        fake_visa(self.root)
        self.m = mtm.build(self.root, 'candle')

    def tearDown(self):
        self.tmp.cleanup()

    def test_calibration_is_first_20_sorted(self):
        self.assertEqual(self.m['calibration'], [f'candle/train/good/{i:03d}.JPG' for i in range(20)])

    def test_official_reference_slicing_after_calibration(self):
        pool = [f'candle/train/good/{i:03d}.JPG' for i in range(20, 60)]
        for s in mtm.SEEDS:
            for k in mtm.SHOTS:
                self.assertEqual(self.m['references'][str(s)][str(k)], pool[s * k:(s + 1) * k])

    def test_validator_accepts_and_rejects(self):
        d = self.root / 'man'; d.mkdir()
        (d / 'candle.json').write_text(json.dumps(self.m))
        rdo.load_manifest(d, 'candle', self.root)
        bad = json.loads(json.dumps(self.m))
        bad['references']['0']['1'] = [bad['calibration'][0]]
        (d / 'candle.json').write_text(json.dumps(bad))
        with self.assertRaises(AssertionError):
            rdo.load_manifest(d, 'candle', self.root)

    def test_test_set_complete_and_gt(self):
        self.assertEqual(len(self.m['test']), 6)
        shape = rdo.metric_shape((40, 60), 20)
        self.assertEqual(shape, (20, 30))
        self.assertFalse(rdo.gt_mask(self.root, 'candle/test/good/000.JPG', shape).any())
        gt = rdo.gt_mask(self.root, 'candle/test/bad/000.JPG', shape)
        self.assertEqual(int(gt.sum()), 5 * 10)


class NNTests(unittest.TestCase):
    def test_matches_bruteforce_normalised_l2_over_2(self):
        rng = np.random.default_rng(0)
        bank, q = rng.normal(size=(500, 16)), rng.normal(size=(70, 16))
        d = rdo.NearestNeighbour(bank, backend='numpy').search(q, chunk=32)
        bn, qn = rdo.l2n(bank), rdo.l2n(q)
        ref = (((qn[:, None] - bn[None]) ** 2).sum(-1)).min(1) / 2
        np.testing.assert_allclose(d, ref, atol=1e-5)

    def test_patch_distances_zero_on_background(self):
        rng = np.random.default_rng(1)
        nn = rdo.NearestNeighbour(rng.normal(size=(50, 8)), backend='numpy')
        f = rng.normal(size=(12, 8)); m = np.zeros(12, bool); m[:5] = True
        d = rdo.patch_distances(nn, f, m, (3, 4))
        self.assertEqual(d.shape, (3, 4)); self.assertTrue((d.ravel()[5:] == 0).all())

    def test_threshold_and_pack(self):
        self.assertAlmostEqual(rdo.provisional_threshold(np.arange(21)), 19.0)
        self.assertEqual(rdo.pack([np.zeros((2, 3))] * 4).shape, (4, 2, 3))
        self.assertEqual(rdo.pack([np.zeros(2), np.zeros(3)]).dtype, object)


if __name__ == '__main__':
    unittest.main()
