"""Truthful normal-training records, deliberately separate from few/zero-shot schema.

Uses the frozen shared manifests and metric functions without changing their release.
NPZ compression preserves every float32 native pixel exactly.
"""
from pathlib import Path
import json
import os
import shutil
import numpy as np
from shared_protocol.protocol import sha256, save_json, inside, PROTOCOL_ID
from shared_protocol.data import load_mask
from shared_protocol.metrics import image_statistics, pixel_statistics_stream


def validate_training(manifest, names):
    if len(names) != len(set(names)) or set(names) != set(manifest['normal_training_pool']):
        raise ValueError('Training must use the exact normal pool, without duplicates')
    forbidden = set(manifest['calibration']) | {r['path'] for r in manifest['test']}
    if set(names) & forbidden:
        raise ValueError('Calibration/test leakage into normal-only training')


def save_map(path, values):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to replace a saved map: {path}')
    if values.dtype != np.float32 or values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Expected finite native float32 2D map')
    if shutil.disk_usage(path.parent).free < 8 * 1024**3:
        raise OSError('Less than8GiB free; refusing to fill disk')
    temporary = path.with_suffix('.partial')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, anomaly_map=values)
    with np.load(temporary, allow_pickle=False) as archive:
        restored = archive['anomaly_map']
    if not np.array_equal(values.view(np.uint32), restored.view(np.uint32)):
        raise ValueError('Lossless storage verification failed')
    os.replace(temporary, path)
    return sha256(path)


def evaluate_run(folder, manifest, data_root, optional_metrics=True):
    folder = Path(folder)
    raw = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
    if raw['protocol_id'] != PROTOCOL_ID or raw['experiment_mode'] != 'normal_training':
        raise ValueError('Wrong normal-training protocol identity')
    validate_training(manifest, raw['training_paths'])
    if raw['shots'] is not None or raw['reference_paths'] or raw['integration_only']:
        raise ValueError('Normal training must not pretend to be few/zero-shot')
    if [r['path'] for r in raw['calibration']] != manifest['calibration']:
        raise ValueError('Calibration list mismatch')
    if [r['path'] for r in raw['test']] != [r['path'] for r in manifest['test']]:
        raise ValueError('Incomplete or reordered tests')
    for record, sample in zip(raw['test'], manifest['test']):
        path = inside(folder, record['map'])
        if sha256(path) != record['map_sha256']:
            raise ValueError('Stored prediction checksum mismatch')
        with np.load(path, allow_pickle=False) as archive:
            a = archive['anomaly_map']
        if a.shape != (sample['height'], sample['width']) or a.dtype != np.float32:
            raise ValueError('Map is not original-size float32')

    def pairs():
        for prediction, sample in zip(raw['test'], manifest['test']):
            with np.load(inside(folder, prediction['map']), allow_pickle=False) as archive:
                values = archive['anomaly_map']
            yield values, load_mask(sample, data_root)

    metrics = image_statistics([r['score'] for r in raw['test']],
                               [r['score'] for r in raw['calibration']],
                               [r['label'] for r in manifest['test']], optional_metrics)
    metrics.update(pixel_statistics_stream(pairs, optional_metrics))
    for key in ['method', 'category', 'seed', 'shots', 'experiment_mode', 'protocol_id',
                'config_id', 'run_config', 'resources', 'training_count', 'integration_only']:
        metrics[key] = raw[key]
    metrics['run_manifest_sha256'] = sha256(folder / 'run.json')
    save_json(folder / 'metrics.json', metrics)
    return metrics
