"""Evaluate complete native-map runs without resizing ground-truth annotations."""
from pathlib import Path
import argparse
import json

import numpy as np

from .data import load_manifest, load_mask
from .metrics import image_statistics, pixel_statistics_stream
from .predictions import source_version
from .protocol import DATA_ROOT, MANIFEST_ROOT, PROTOCOL_ID, save_json, sha256, inside
from .experiments import expected_references, validate_experiment_metadata


def evaluate_file(predictions, category=None, shots=None, seed=None, method=None, config_id=None,
                  data_root=DATA_ROOT, manifest_root=MANIFEST_ROOT, optional_metrics=True):
    path = Path(predictions)
    run_path = path / 'run.json' if path.is_dir() else path
    if run_path.suffix != '.json':
        raise ValueError('Use PredictionWriter native-map run.json; old fixed-grid NPZ runs are not v2 results')
    metadata = json.loads(run_path.read_text(encoding='utf-8'))
    run_root = run_path.parent
    if metadata.get('protocol_id') != PROTOCOL_ID or metadata.get('format') != 'native_maps_v2' or metadata.get('status') != 'complete':
        raise ValueError('Wrong protocol/format or incomplete prediction run')
    for key, expected in [('category', category), ('shots', shots), ('seed', seed), ('method', method), ('config_id', config_id)]:
        if expected is not None and metadata.get(key) != expected:
            raise ValueError(f'Run {key} differs from requested evaluation')
    category, shots, seed = metadata['category'], metadata['shots'], metadata['seed']
    spec = validate_experiment_metadata(metadata)
    manifest = load_manifest(category, manifest_root)
    manifest_hash = sha256(Path(manifest_root) / f'{category}.json')
    if metadata.get('manifest_sha256') != manifest_hash:
        raise ValueError('Predictions were made with a different manifest version')
    if metadata.get('reference_paths') != expected_references(manifest, shots, seed, metadata['method'], spec['experiment_mode']):
        raise ValueError('Wrong reference selection')
    if metadata.get('calibration_paths') != manifest['calibration']:
        raise ValueError('Wrong calibration selection/order')
    if metadata.get('map_space') != 'original_image_pixels' or metadata.get('raw_scores_without_per_image_normalization') is not True:
        raise ValueError('Expected image-aligned native maps and original raw scores')
    records = metadata.get('test', [])
    if [r['path'] for r in records] != [r['path'] for r in manifest['test']]:
        raise ValueError('Predictions do not cover the COMPLETE ordered official test set')
    if len({r['map'] for r in records}) != len(records):
        raise ValueError('Each test image must have its own anomaly map')
    for record, sample in zip(records, manifest['test']):
        if sha256(inside(run_root, record['map'])) != record['map_sha256']:
            raise ValueError(f'Prediction map content changed: {record["path"]}')
        if sample['mask'] and sha256(inside(data_root, sample['mask'])) != sample['mask_sha256']:
            raise ValueError(f'GT mask content changed: {sample["mask"]}')

    def pairs():
        for record, sample in zip(records, manifest['test']):
            values = np.load(inside(run_root, record['map']), mmap_mode='r', allow_pickle=False)
            expected_shape = (sample['height'], sample['width'])
            if values.shape != expected_shape:
                raise ValueError(f'Map must match native image dimensions: {record["path"]}')
            mask = load_mask(sample, data_root)
            if mask.shape != expected_shape or (sample['label'] == 1 and not mask.any()):
                raise ValueError('GT has wrong native dimensions or empty anomaly annotation')
            yield values, mask

    result = image_statistics([r['score'] for r in records], metadata['calibration_scores'],
                              [sample['label'] for sample in manifest['test']], optional_metrics)
    result.update(pixel_statistics_stream(pairs, optional_metrics))
    result.update(category=category, shots=shots, seed=seed, method=metadata['method'],
                  config_id=metadata['config_id'], protocol_id=PROTOCOL_ID,
                  integration_only=metadata.get('integration_only', False),
                  pixel_evaluation='native_original_image_and_mask', aupro_connectivity=8,
                  run_config=metadata['run_config'], resources=metadata.get('resources', {}),
                  manifest_sha256=manifest_hash, prediction_manifest_sha256=sha256(run_path),
                  producer_source_sha256=metadata.get('source_sha256'), evaluator_source_sha256=source_version())
    result.update(spec, reference_paths=metadata['reference_paths'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', '--predictions', dest='predictions', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--manifest-root', type=Path, default=MANIFEST_ROOT)
    parser.add_argument('--required-only', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_file(args.predictions, data_root=args.data_root, manifest_root=args.manifest_root,
                           optional_metrics=not args.required_only)
    save_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
