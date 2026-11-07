from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from methods.part1.template import TemplateDetector, preprocess, rotate_reference, image_score, process_difference


class TemplateTests(unittest.TestCase):
    def test_geometry_keeps_aspect_then_patch_multiple(self):
        result = preprocess(np.zeros((1000, 1500, 3), dtype=np.uint8))
        self.assertEqual(result.shape, (448, 672, 3))
        self.assertEqual(preprocess(np.zeros((1500, 1000, 3), dtype=np.uint8)).shape, (672, 448, 3))

    def test_identity_has_zero_difference_and_global_score_scale(self):
        image = np.full((28, 35, 3), 80, dtype=np.uint8)
        self.assertEqual(float(process_difference(image, image).max()), 0.)
        self.assertAlmostEqual(image_score(np.full((10, 10), .25)), .25)
        self.assertAlmostEqual(image_score(np.array([[0., 0., 0., .8]])), .8)

    def test_template_selection_and_synthetic_local_defect(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dark, bright = np.full((448, 448, 3), 30, dtype=np.uint8), np.full((448, 448, 3), 180, dtype=np.uint8)
            Image.fromarray(dark).save(root / 'dark.png')
            Image.fromarray(bright).save(root / 'bright.png')
            detector = TemplateDetector([root / 'dark.png', root / 'bright.png'], ['dark', 'bright'])
            normal = detector.predict_rgb(bright)
            self.assertEqual(normal.chosen_reference, 'bright')
            self.assertEqual(normal.score, 0.)
            damaged = bright.copy()
            damaged[200:230, 200:230] = 0
            prediction = detector.predict_rgb(damaged)
            self.assertEqual(prediction.chosen_reference, 'bright')
            self.assertGreater(prediction.score, 0.)
            self.assertGreater(prediction.native_map[215, 215], prediction.native_map[50, 50])
            self.assertEqual(prediction.native_map.shape, (448, 448))

    def test_rotation_preserves_canvas_and_zero_rotation(self):
        image = np.arange(30 * 40 * 3, dtype=np.uint8).reshape(30, 40, 3)
        self.assertTrue(np.array_equal(rotate_reference(image, 0), image))
        self.assertEqual(rotate_reference(image, 45).shape, image.shape)


if __name__ == '__main__':
    unittest.main()
