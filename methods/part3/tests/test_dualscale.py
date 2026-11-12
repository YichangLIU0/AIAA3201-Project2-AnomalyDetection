"""CPU-only checks for the dual-scale helpers (no torch, no data needed).
Run:  python -m unittest methods/part3/tests/test_dualscale.py
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_dualscale as rd  # noqa: E402


class DualScaleTests(unittest.TestCase):
    def test_mask_projection_keeps_shape_and_content(self):
        grid4, grid6 = (32, 35), (48, 52)
        mask = np.zeros(grid4, dtype=bool)
        mask[8:24, 10:25] = True
        out = rd.project_mask(mask.reshape(-1), grid4, grid6)
        self.assertEqual(out.shape, (48 * 52,))
        self.assertTrue(out.reshape(grid6)[24, 26])
        self.assertFalse(out.reshape(grid6)[2, 2])
        self.assertAlmostEqual(out.mean(), mask.mean(), delta=0.02)
        full = rd.project_mask(np.ones(32 * 35, dtype=bool), grid4, grid6)
        self.assertTrue(full.all())

    def test_fusion_is_equal_weight_mean_on_the_672_grid(self):
        d4 = np.full((32, 35), 0.2, dtype=np.float32)
        d6 = np.full((48, 52), 0.4, dtype=np.float32)
        fused = rd.fuse_grids(d4, d6)
        self.assertEqual(fused.shape, (48, 52))
        self.assertEqual(fused.dtype, np.float32)
        np.testing.assert_allclose(fused, 0.3, atol=1e-7)
        self.assertEqual(tuple(rd.FUSE_WEIGHTS), (0.5, 0.5))

    def test_fusion_keeps_a_local_peak_and_background_zero(self):
        d4 = np.zeros((32, 35), dtype=np.float32)
        d6 = np.zeros((48, 52), dtype=np.float32)
        d6[30, 40] = 1.0
        fused = rd.fuse_grids(d4, d6)
        self.assertEqual(np.unravel_index(np.argmax(fused), fused.shape), (30, 40))
        self.assertAlmostEqual(float(fused[30, 40]), 0.5, places=6)
        self.assertEqual(float(fused[0, 0]), 0.0)

    def test_config_grid_routes_each_configuration(self):
        d4, d6 = np.ones((2, 3), np.float32), np.full((3, 4), 3, np.float32)
        self.assertIs(rd.config_grid('s448', d4, d6), d4)
        self.assertIs(rd.config_grid('s672', d4, d6), d6)
        self.assertEqual(rd.config_grid('fuse', d4, d6).shape, (3, 4))
        self.assertEqual(set(rd.CONFIGS), {'s448', 's672', 'fuse'})


if __name__ == '__main__':
    unittest.main()
