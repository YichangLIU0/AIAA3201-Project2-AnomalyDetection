"""Portable, ordered native-map prediction files for either teammate's detector."""
from pathlib import Path
import hashlib
import json

import numpy as np

from .data import load_manifest
from .protocol import PACKAGE, MANIFEST_ROOT, PROTOCOL_ID, save_json, sha256
from .experiments import experiment_spec, expected_references


def source_version():
    digest = hashlib.sha256()
    for path in sorted(PACKAGE.glob('*.py')):
        digest.update(path.name.encode('utf-8'))
        digest.update(path.read_bytes())
    return digest.hexdigest()


class PredictionWriter:
    """Save raw scores and one original-size anomaly map per test image.

    Maps must already be aligned to the original image by the detector, e.g.
    upstream dists2map(patch_distances, original_rgb.shape). This writer does
    not resize, smooth, normalize or change the image-score definition.
    """
    def __init__(self, output, category, shots, seed, method, config_id, run_config,
                 manifest_root=MANIFEST_ROOT, integration_only=False, experiment_mode=None):
        self.output = Path(output)
        self.manifest = load_manifest(category, manifest_root)
        spec = experiment_spec(shots, seed, method, experiment_mode)
        self.refs = expected_references(self.manifest, shots, seed, method, experiment_mode)
        if not method or not config_id or not isinstance(run_config, dict) or not run_config:
            raise ValueError('Record method, fixed config ID and actual run configuration')
        if (self.output / 'run.json').exists() or (self.output / 'maps').exists():
            raise ValueError('Prediction destination already exists; use a new run directory')
        self.output.mkdir(parents=True, exist_ok=True)
        self.records, self.calibration = [], []
        self.metadata = {'format': 'native_maps_v2', 'protocol_id': PROTOCOL_ID, 'status': 'incomplete',
                         'category': category, 'shots': shots, 'seed': seed, 'method': method, 'config_id': config_id,
                         'run_config': run_config, 'integration_only': integration_only,
                         'manifest_sha256': sha256(Path(manifest_root) / f'{category}.json'),
                         'source_sha256': source_version(), 'reference_paths': self.refs,
                         'calibration_paths': self.manifest['calibration'], 'map_space': 'original_image_pixels',
                         'raw_scores_without_per_image_normalization': True}
        self.metadata.update(spec)
        save_json(self.output / 'incomplete.json', self.metadata)

    def write_calibration(self, path, score):
        index = len(self.calibration)
        if index >= 20 or path != self.manifest['calibration'][index] or not np.isfinite(score):
            raise ValueError('Calibration path/order/score differs from the fixed sorted 20')
        self.calibration.append(float(score))

    def write_test(self, path, anomaly_map, score, inference_seconds=None):
        index = len(self.records)
        if index >= len(self.manifest['test']) or path != self.manifest['test'][index]['path']:
            raise ValueError('Write every official test image once, in manifest order')
        sample = self.manifest['test'][index]
        values = np.asarray(anomaly_map)
        if values.shape != (sample['height'], sample['width']):
            raise ValueError('Prediction map must match ORIGINAL image dimensions; do not shrink GT')
        if values.dtype.kind not in 'biuf' or not np.isfinite(values).all() or not np.isfinite(score):
            raise ValueError('Map and image score must be finite real numbers')
        if values.dtype.kind != 'f':
            values = values.astype(np.float64)
        if inference_seconds is not None and (not np.isfinite(inference_seconds) or inference_seconds < 0):
            raise ValueError('Invalid inference duration')
        relative = f'maps/{index:04d}.npy'
        target = self.output / relative
        target.parent.mkdir(exist_ok=True)
        np.save(target, values, allow_pickle=False)
        self.records.append({'path': path, 'map': relative, 'map_sha256': sha256(target), 'score': float(score),
                             'inference_seconds': None if inference_seconds is None else float(inference_seconds)})

    def finish(self, resources=None):
        if len(self.calibration) != 20 or len(self.records) != len(self.manifest['test']):
            raise ValueError('Cannot finalize an incomplete calibration/test run')
        metadata = {**self.metadata, 'status': 'complete', 'calibration_scores': self.calibration,
                    'test': self.records, 'resources': resources or {}}
        save_json(self.output / 'run.json', metadata)
        return self.output / 'run.json'
