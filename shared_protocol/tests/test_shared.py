from pathlib import Path
import tempfile
import unittest
import ast
import contextlib
import copy
import io
import json
from bisect import bisect

import numpy as np
from PIL import Image
from scipy.integrate import trapezoid
from scipy.ndimage import label
from sklearn.metrics import roc_auc_score, average_precision_score

from shared_protocol.data import load_mask, load_rgb, load_manifest, reference_paths
from shared_protocol.metrics import normal_threshold, evaluate, pixel_statistics, aggregate_runs, pixel_statistics_stream
from shared_protocol.protocol import official_references, inside, save_json, PROTOCOL_ID, PACKAGE, OFFICIAL_SPLIT_SHA256, sha256
from shared_protocol.evaluate_predictions import evaluate_file
from shared_protocol.predictions import PredictionWriter
from shared_protocol.run_detector import run_detector


def brute_pro(maps, masks, limit=0.3):
    """Independent threshold-by-threshold reference; no weighted-sort shortcut."""
    components = []
    for i, mask in enumerate(masks):
        labels, count = label(mask, np.ones((3, 3), dtype=np.uint8))
        components.extend((i, labels == j) for j in range(1, count + 1))
    x, y = [0.], [0.]
    negative_count = sum(np.count_nonzero(~mask) for mask in masks)
    for threshold in sorted(np.unique(np.concatenate([m.ravel() for m in maps])), reverse=True):
        predicted = [m >= threshold for m in maps]
        x.append(sum(np.count_nonzero(p & ~m) for p, m in zip(predicted, masks)) / negative_count)
        y.append(float(np.mean([predicted[i][region].mean() for i, region in components])))
    k = np.searchsorted(x, limit, side='right')
    if x[k - 1] < limit:
        crossing = y[k - 1] + (y[k] - y[k - 1]) * (limit - x[k - 1]) / (x[k] - x[k - 1])
        return float(trapezoid(y[:k] + [crossing], x[:k] + [limit]) / limit)
    return float(trapezoid(y[:k], x[:k]) / limit)


class MetricsTests(unittest.TestCase):
    def test_required_threshold_is_linear_95th_percentile_of_exactly_20(self):
        self.assertAlmostEqual(normal_threshold(np.arange(20)), 18.05)
        for bad in [np.arange(19), np.arange(40), [np.nan] * 20, np.ones((20, 1))]:
            with self.assertRaises(ValueError):
                normal_threshold(bad)

    def test_perfect_tied_and_reversed_predictions(self):
        gt = np.zeros((2, 4, 4), dtype=bool)
        gt[1, 1:3, 1:3] = True
        perfect = evaluate(gt.astype(float), gt, [0., 1.], np.zeros(20))
        for key in ['image_auroc', 'image_ap', 'pixel_auroc', 'pixel_ap', 'aupro_03']:
            self.assertAlmostEqual(perfect[key], 1.)
        self.assertEqual(perfect['image_fpr'], 0.)  # Equality to threshold is OK.
        tied = evaluate(np.ones(gt.shape), gt, [1., 1.], np.ones(20))
        self.assertAlmostEqual(tied['image_auroc'], .5)
        self.assertAlmostEqual(tied['pixel_auroc'], .5)
        self.assertAlmostEqual(tied['pixel_ap'], 4 / 32)
        self.assertAlmostEqual(tied['aupro_03'], .15)
        self.assertEqual(tied['image_fnr'], 1.)
        reverse = evaluate(1. - gt.astype(float), gt, [1., 0.], np.zeros(20))
        self.assertEqual(reverse['image_auroc'], 0.)
        self.assertEqual(reverse['pixel_auroc'], 0.)

    def test_raw_pixel_metrics_match_sklearn_and_independent_pro(self):
        rng = np.random.default_rng(20261105)
        for ties in [False, True]:
            for _ in range(4):
                gt = rng.random((3, 7, 8)) < .15
                gt[0] = False  # All normal test pixels must be included.
                maps = rng.random(gt.shape)
                if ties:
                    maps = np.round(maps, 1)
                result = pixel_statistics(maps, gt)
                self.assertAlmostEqual(result['pixel_auroc'], roc_auc_score(gt.ravel(), maps.ravel()), places=12)
                self.assertAlmostEqual(result['pixel_ap'], average_precision_score(gt.ravel(), maps.ravel()), places=12)
                self.assertAlmostEqual(result['aupro_03'], brute_pro(maps, gt), places=7)

    def test_image_labels_survive_small_defect_mask_downsampling(self):
        masks = np.zeros((3, 2, 2), dtype=bool)
        masks[2, 0, 0] = True
        result = evaluate(masks.astype(float), masks, [0., .5, 1.], np.zeros(20), image_labels=[0, 1, 1])
        self.assertEqual(result['n_test_anomaly'], 2)

    def test_invalid_arrays_and_nonfinite_values_rejected(self):
        gt = np.zeros((2, 3, 3), dtype=bool)
        gt[1, 1, 1] = True
        with self.assertRaises(ValueError):
            pixel_statistics(np.full(gt.shape, np.nan), gt)
        with self.assertRaises(ValueError):
            pixel_statistics(gt.astype(float), gt.astype(np.uint8) * 255)
        with self.assertRaises(ValueError):
            pixel_statistics(np.ones((2, 3, 4)), gt)

    def test_unsigned_integer_maps_do_not_overflow_when_sorted(self):
        gt = np.zeros((2, 3, 3), dtype=bool)
        gt[1, 1, 1] = True
        result = pixel_statistics(gt.astype(np.uint8) * 255, gt)
        self.assertAlmostEqual(result['pixel_auroc'], 1.)
        self.assertAlmostEqual(result['aupro_03'], 1.)

    def test_category_macro_then_population_seed_std(self):
        rows = []
        for c in ['candle', 'capsules']:
            for s in [0, 1, 2]:
                rows.append({'category': c, 'shots': 1, 'seed': s, 'method': 'fixture',
                             'status': 'success', 'config_id': 'fixture', 'protocol_id': PROTOCOL_ID,
                             'image_auroc': (.2 if c == 'candle' else .8) + s * .05,
                             'pixel_auroc': .7})
        result = aggregate_runs(rows, ['candle', 'capsules'], shots=[1])[0]
        self.assertAlmostEqual(result['image_auroc']['mean'], .55)
        self.assertAlmostEqual(result['image_auroc']['std'], np.std([.5, .55, .6], ddof=0))
        with self.assertRaises(ValueError):
            aggregate_runs(rows[:-1], ['candle', 'capsules'], shots=[1])
        with self.assertRaises(ValueError):
            aggregate_runs(rows + rows[:1], ['candle', 'capsules'], shots=[1])


class DataTests(unittest.TestCase):
    def test_official_block_selection_not_random_nested_sampling(self):
        train = [f'candle/train/good/{i:04d}.JPG' for i in range(100)]
        cal, pool = train[:20], train[20:]
        self.assertEqual(official_references(pool, 1, 2), [train[22]])
        self.assertEqual(official_references(pool, 4, 1), train[24:28])
        self.assertEqual(official_references(pool, 8, 2), train[36:44])
        self.assertTrue(set(official_references(pool, 8, 0)).isdisjoint(cal))

    def test_zero_normal_mask_nearest_neighbor_and_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.array([[0, 255], [0, 0]], dtype=np.uint8)
            Image.fromarray(values).save(root / 'mask.png')
            Image.new('RGB', (6, 4), 'red').save(root / 'image.png')
            normal = {'label': 0, 'mask': None, 'height': 4, 'width': 6}
            anomaly = {'label': 1, 'mask': 'mask.png', 'height': 2, 'width': 2}
            self.assertEqual(load_mask(normal, root).shape, (4, 6))
            self.assertFalse(load_mask(normal, root).any())
            self.assertEqual(load_mask(anomaly, root, (4, 4)).sum(), 4)
            self.assertEqual(load_rgb(root / 'image.png').shape, (4, 6, 3))
            self.assertEqual(load_rgb(root / 'image.png', (8, 10)).shape, (8, 10, 3))
            with self.assertRaises(ValueError):
                inside(root, '../outside.txt')

    def test_prediction_file_end_to_end_and_coverage_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = create_fixture_manifest(root)
            writer = PredictionWriter(root / 'run', 'candle', 1, 0, 'TEST_FIXTURE', 'TEST_ONLY',
                                      {'synthetic_test_fixture': True}, manifest_root=root, integration_only=True)
            for path in manifest['calibration']:
                writer.write_calibration(path, 0.)
            for sample in manifest['test']:
                values = load_mask(sample, root).astype(np.float32)
                writer.write_test(sample['path'], values, float(sample['label']))
            run_path = writer.finish()
            result = evaluate_file(run_path, data_root=root, manifest_root=root)
            self.assertEqual(result['image_auroc'], 1.)
            self.assertEqual(result['pixel_auroc'], 1.)
            self.assertEqual(result['pixel_evaluation'], 'native_original_image_and_mask')
            metadata = json.loads(run_path.read_text(encoding='utf-8'))
            metadata['test'].reverse()
            save_json(run_path, metadata)
            with self.assertRaises(ValueError):
                evaluate_file(run_path, data_root=root, manifest_root=root)


def create_fixture_manifest(root):
    """Official candle membership with tiny varied-size masks, for integration tests only."""
    m = load_manifest('candle')
    lookup = {r['path']: r for r in m['inventory']}
    for index, sample in enumerate(m['test']):
        sample['height'], sample['width'] = 4 + index % 2, 6 + index % 3
        if sample['label']:
            values = np.zeros((sample['height'], sample['width']), dtype=np.uint8)
            values[1, 1] = 255
            p = root / sample['mask']
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(values).save(p)
            sample['mask_sha256'], sample['anomaly_pixels'] = sha256(p), 1
        lookup[sample['path']].update(sample)
    save_json(root / 'candle.json', m)
    for group in ['calibration', 'normal_training_pool']:
        p = root / group / 'candle.txt'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('\n'.join(m[group]) + '\n', encoding='utf-8')
    for k, per_seed in m['references'].items():
        for s, paths in per_seed.items():
            p = root / 'references' / f'candle_{k}shot_seed{s}.txt'
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('\n'.join(paths) + '\n', encoding='utf-8')
    save_json(root / 'manifest_lock.json', {'protocol_id': PROTOCOL_ID, 'official_split_sha256': OFFICIAL_SPLIT_SHA256,
              'protocol_sha256': sha256(PACKAGE / 'protocol.json'), 'manifests': {'candle': sha256(root / 'candle.json')}})
    return m


class FixRegressionTests(unittest.TestCase):
    def test_diagonal_connected_region_matches_official(self):
        gt = np.zeros((2, 4, 4), dtype=bool)
        gt[1, 0, 0] = gt[1, 1, 1] = gt[1, 1, 2] = True
        maps = np.full(gt.shape, .5)
        maps[gt] = 0.
        maps[1, 0, 0] = 1.
        self.assertAlmostEqual(pixel_statistics(maps, gt)['aupro_03'], 1 / 3, places=12)

    def test_official_pro_crosscheck_with_ties(self):
        path = PACKAGE / 'upstream/anomalydino_post_eval.py'
        parsed = ast.parse(path.read_text(encoding='utf-8'))
        functions = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in ['compute_pro', 'trapezoid']]
        env = {'np': np, 'label': label, 'bisect': bisect}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), env)
        rng = np.random.default_rng(98)
        for dtype in [np.float32, np.float64]:
            for tied in [False, True]:
                maps = rng.random((4, 11, 13)).astype(dtype)
                masks = rng.random(maps.shape) < .17
                masks[0] = False
                if tied:
                    maps = np.round(maps, 1)
                with contextlib.redirect_stdout(io.StringIO()):
                    x, y = env['compute_pro'](maps, masks)
                reference = env['trapezoid'](x, y, x_max=.3) / .3
                result = pixel_statistics(maps, masks)
                self.assertAlmostEqual(result['aupro_03'], reference, places=7)

    def test_variable_native_shapes_and_background_chunking_match_sklearn(self):
        rng = np.random.default_rng(612)
        maps = [rng.random((4, 7)), rng.random((6, 3)), rng.random((5, 8))]
        masks = [np.zeros((4, 7), dtype=bool), rng.random((6, 3)) < .2, rng.random((5, 8)) < .3]
        flat_scores, flat_gt = np.concatenate([x.ravel() for x in maps]), np.concatenate([x.ravel() for x in masks])
        expected_roc, expected_ap = roc_auc_score(flat_gt, flat_scores), average_precision_score(flat_gt, flat_scores)
        for chunk in [1, 15, 1000]:
            result = pixel_statistics_stream(lambda: zip(maps, masks), background_chunk_pixels=chunk)
            self.assertAlmostEqual(result['pixel_auroc'], expected_roc, places=12)
            self.assertAlmostEqual(result['pixel_ap'], expected_ap, places=12)
            self.assertAlmostEqual(result['aupro_03'], brute_pro(maps, masks), places=12)

    def test_changed_calibration_reference_is_rejected_even_in_memory(self):
        m = load_manifest('candle')
        m['references']['1']['0'] = m['calibration'][:1]
        with self.assertRaises(ValueError):
            reference_paths(m, 1, 0)

    def test_changed_txt_and_changed_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            m = create_fixture_manifest(root)
            txt = root / 'references/candle_1shot_seed0.txt'
            txt.write_text(m['calibration'][0] + '\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_manifest('candle', root)
            txt.write_text(m['references']['1']['0'][0] + '\n', encoding='utf-8')
            m['test'].pop()
            save_json(root / 'candle.json', m)
            with self.assertRaises(ValueError):
                load_manifest('candle', root)

    def test_missing_required_metrics_failed_runs_and_out_of_range_rejected(self):
        record = {'category': 'candle', 'shots': 1, 'seed': 0, 'method': 'fixture',
                  'status': 'success', 'config_id': 'fixture', 'protocol_id': PROTOCOL_ID}
        for extra in [{}, {'image_auroc': .5}, {'image_auroc': .5, 'pixel_auroc': 1.1},
                      {'integration_only': True, 'image_auroc': .5, 'pixel_auroc': .5},
                      {'status': 'failed', 'image_auroc': .5, 'pixel_auroc': .5}]:
            with self.assertRaises(ValueError):
                aggregate_runs([{**record, **extra}], ['candle'], shots=[1], seeds=[0])

    def test_writer_rejects_shrunken_native_map_and_incomplete_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            m = create_fixture_manifest(root)
            writer = PredictionWriter(root / 'run', 'candle', 1, 0, 'fixture', 'TEST_ONLY', {'test': True}, root, True)
            with self.assertRaises(ValueError):
                writer.write_test(m['test'][0]['path'], np.zeros((448, 448)), 0.)
            with self.assertRaises(ValueError):
                writer.finish()

    def test_common_runner_visits_20_calibration_and_every_official_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = create_fixture_manifest(root)
            samples = {s['path']: s for s in manifest['test']}
            visited, sync_calls = [], []

            def predict(path):
                relative = path.relative_to(root).as_posix()
                visited.append(relative)
                if relative in samples:
                    sample = samples[relative]
                    return float(sample['label']), load_mask(sample, root).astype(np.float32)
                return 0., np.zeros((4, 6), dtype=np.float32)

            result = run_detector(predict, root / 'run', 'candle', 1, 0, 'fixture', 'TEST_ONLY',
                                  {'fixture_only': True}, data_root=root, manifest_root=root,
                                  synchronize=lambda: sync_calls.append(True),
                                  resource_reader=lambda: {'device': 'mock_device'}, integration_only=True)
            self.assertEqual(visited, manifest['calibration'] + [s['path'] for s in manifest['test']])
            self.assertEqual(len(sync_calls), 2 * len(manifest['test']))
            self.assertEqual(result['n_calibration'], 20)
            self.assertEqual(result['n_test'], 200)
            self.assertTrue(result['integration_only'])


if __name__ == '__main__':
    unittest.main()
