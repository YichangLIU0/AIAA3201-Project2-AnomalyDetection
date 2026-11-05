"""Zero-shot metadata, end-to-end I/O and backward-compatible few-shot checks."""
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from shared_protocol.data import load_manifest, load_mask, reference_paths
from shared_protocol.evaluate_predictions import evaluate_file
from shared_protocol.experiments import experiment_spec
from shared_protocol.metrics import aggregate_runs
from shared_protocol.predictions import PredictionWriter
from shared_protocol.protocol import CATEGORIES, PROTOCOL_ID, save_json
from shared_protocol.run_detector import run_detector
from shared_protocol.tests.test_shared import create_fixture_manifest


def zero_record(category, seed=0):
    return {'category': category, 'shots': 0, 'seed': seed, 'method': 'anomalyclip',
            'config_id': 'TEST_ONLY', 'protocol_id': PROTOCOL_ID, 'status': 'success',
            'experiment_mode': 'zero_shot', 'reference_seed': None, 'run_seed': seed,
            'seed_role': 'runtime_only', 'reference_paths': [], 'image_auroc': .7,
            'pixel_auroc': .8}


class ZeroShotTests(unittest.TestCase):
    def test_reference_helper_returns_empty_list(self):
        self.assertEqual(reference_paths(load_manifest('candle'), 0, 0), [])
        spec = experiment_spec(0, 0, 'AnomalyCLIP')
        self.assertEqual(spec['experiment_mode'], 'zero_shot')
        self.assertIsNone(spec['reference_seed'])
        self.assertEqual(spec['run_seed'], 0)

    def test_reject_fake_anomalyclip_shots_and_contradictory_modes(self):
        for method in ['AnomalyCLIP', 'anomaly_clip', 'anomaly-clip']:
            with self.assertRaises(ValueError):
                experiment_spec(1, 0, method)
        for shots, seed, mode in [(0, 0, 'few_shot'), (1, 0, 'zero_shot'), (2, 3, None),
                                  (True, 0, None), (1.0, 0, None), (0, -1, None)]:
            with self.assertRaises(ValueError):
                experiment_spec(shots, seed, experiment_mode=mode)

    def test_zero_shot_runner_calibrates_and_evaluates_all_images_without_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = create_fixture_manifest(root)
            samples = {s['path']: s for s in manifest['test']}
            visited = []

            def fixture_predict(path):
                relative = path.relative_to(root).as_posix()
                visited.append(relative)
                if relative in samples:
                    sample = samples[relative]
                    return float(sample['label']), load_mask(sample, root).astype(np.float32)
                return 0., np.zeros((4, 6), dtype=np.float32)

            result = run_detector(fixture_predict, root / 'run', 'candle', 0, 0, 'anomalyclip',
                                  'TEST_FIXTURE_NOT_MODEL_RESULTS', {'synthetic_test_fixture': True},
                                  data_root=root, manifest_root=root, integration_only=True,
                                  experiment_mode='zero_shot')
            self.assertEqual(visited, manifest['calibration'] + [s['path'] for s in manifest['test']])
            self.assertEqual(len(visited), 220)
            self.assertEqual(result['shots'], 0)
            self.assertEqual(result['reference_paths'], [])
            self.assertEqual(result['experiment_mode'], 'zero_shot')
            self.assertIsNone(result['reference_seed'])
            self.assertEqual(result['run_seed'], 0)
            self.assertEqual(result['n_calibration'], 20)
            self.assertEqual(result['n_test'], 200)
            run_path = root / 'run/run.json'
            metadata = json.loads(run_path.read_text(encoding='utf-8'))
            metadata['reference_paths'] = manifest['calibration'][:1]
            save_json(run_path, metadata)
            with self.assertRaises(ValueError):
                evaluate_file(run_path, data_root=root, manifest_root=root)
            metadata['reference_paths'] = []
            metadata['reference_seed'] = 0
            save_json(run_path, metadata)
            with self.assertRaises(ValueError):
                evaluate_file(run_path, data_root=root, manifest_root=root)

    def test_zero_shot_still_requires_exactly_20_calibration_and_full_test_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = create_fixture_manifest(root)
            writer = PredictionWriter(root / 'run', 'candle', 0, 0, 'anomalyclip', 'TEST_ONLY',
                                      {'fixture': True}, manifest_root=root)
            for path in manifest['calibration'][:19]:
                writer.write_calibration(path, 0.)
            with self.assertRaises(ValueError):
                writer.finish()
            writer.write_calibration(manifest['calibration'][19], 0.)
            with self.assertRaises(ValueError):
                writer.finish()

    def test_six_category_zero_shot_macro_has_no_fabricated_reference_seed_stability(self):
        rows = [zero_record(c) for c in CATEGORIES]
        rows[0]['image_auroc'] = .4
        summary = aggregate_runs(rows, CATEGORIES)
        self.assertEqual(len(summary), 1)
        item = summary[0]
        self.assertEqual(item['shots'], 0)
        self.assertEqual(item['n_categories'], 6)
        self.assertEqual(item['n_runs_per_category'], 1)
        self.assertEqual(item['seed_role'], 'runtime_only')
        self.assertFalse(item['seed_stability_measured'])
        self.assertIsNone(item['image_auroc']['std'])
        self.assertAlmostEqual(item['image_auroc']['mean'], .65)
        self.assertIsNone(item['image_auroc']['per_category']['candle']['std'])
        for bad in [rows[:-1], rows + rows[:1]]:
            with self.assertRaises(ValueError):
                aggregate_runs(bad, CATEGORIES)

    def test_explicit_repeated_runtime_seeds_are_distinct_from_reference_seeds(self):
        rows = [zero_record(c, seed) for c in CATEGORIES for seed in [0, 1]]
        summary = aggregate_runs(rows, CATEGORIES, seeds=[0, 1])[0]
        self.assertTrue(summary['seed_stability_measured'])
        self.assertEqual(summary['seed_role'], 'runtime_only')
        self.assertEqual(summary['image_auroc']['std'], 0.)

    def test_reject_nonempty_references_and_mixed_modes_in_aggregation(self):
        row = zero_record('candle')
        row['reference_paths'] = ['candle/train/good/0023.JPG']
        with self.assertRaises(ValueError):
            aggregate_runs([row], ['candle'])
        few = {**zero_record('candle'), 'method': 'combined_fixture', 'shots': 1,
               **experiment_spec(1, 0), 'reference_paths': ['candle/train/good/0023.JPG']}
        zero = {**zero_record('candle'), 'method': 'combined_fixture'}
        with self.assertRaises(ValueError):
            aggregate_runs([zero, few], ['candle'], shots=[0, 1], seeds=[0])

    def test_default_few_shot_grid_still_requires_72_runs(self):
        rows = []
        for c in CATEGORIES:
            for k in [1, 2, 4, 8]:
                for seed in [0, 1, 2]:
                    rows.append({'category': c, 'shots': k, 'seed': seed, 'method': 'official_dino',
                                 'config_id': 'TEST_ONLY', 'protocol_id': PROTOCOL_ID, 'status': 'success',
                                 'image_auroc': .7, 'pixel_auroc': .8})
        summary = aggregate_runs(rows, CATEGORIES)
        self.assertEqual(len(rows), 72)
        self.assertEqual([s['shots'] for s in summary], [1, 2, 4, 8])
        self.assertTrue(all(s['n_runs_per_category'] == 3 and s['experiment_mode'] == 'few_shot' for s in summary))
        with self.assertRaises(ValueError):
            aggregate_runs(rows[:-1], CATEGORIES)


if __name__ == '__main__':
    unittest.main()
