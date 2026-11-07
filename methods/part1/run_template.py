"""Formal Template grid. Freeze config, use all tests, save native maps and metrics."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import psutil

from shared_protocol.data import load_manifest, reference_paths, calibration_paths
from shared_protocol.evaluate_predictions import evaluate_file
from shared_protocol.predictions import PredictionWriter
from shared_protocol.protocol import ROOT, DATA_ROOT, CATEGORIES, SHOTS, SEEDS, inside, save_json, sha256
from shared_protocol.metrics import aggregate_runs
from .template import CONFIG, TemplateDetector

METHOD = 'template_rgb_rot8'
HERE = Path(__file__).resolve().parent


def source_hashes():
    return {p.name: sha256(p) for p in [HERE / 'template.py', HERE / 'run_template.py']}


def frozen_config():
    encoded = json.dumps(CONFIG, sort_keys=True).encode('utf-8')
    return {**CONFIG, 'config_id': 'template_rgb_rot8_' + hashlib.sha256(encoded).hexdigest()[:12],
            'algorithm_source_sha256': source_hashes()}


def environment():
    versions = {}
    for package in ['numpy', 'scipy', 'scikit-learn', 'Pillow', 'matplotlib', 'psutil']:
        versions[package] = importlib.metadata.version(package)
    proc = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return {'python': sys.version, 'platform': platform.platform(), 'cpu': platform.processor(),
            'opencv': cv2.__version__, 'dependencies': versions,
            'git_base_commit': proc.stdout.strip() if proc.returncode == 0 else None,
            'cpu_count_logical': psutil.cpu_count(), 'gpu_used': False,
            'hardware_comparison_note': 'CPU classical baseline; compare latency only with explicitly disclosed CPU/GPU device settings.'}


def freeze(output_root, result_root, config):
    output_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    path = output_root / 'frozen_config.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != config:
        raise ValueError('Configuration/source changed: use a new output directory, never mix runs')
    save_json(path, config)
    save_json(result_root / 'frozen_config.json', config)


def run_one(category, shots, seed, data_root, output_root, result_root, config):
    folder = output_root / category / f'{shots}shot_seed{seed}'
    compact = result_root / 'runs' / category / f'{shots}shot_seed{seed}'
    metric_file = folder / 'metrics.json'
    if metric_file.exists():
        previous = json.loads(metric_file.read_text(encoding='utf-8'))
        if previous['run_config'] != config or previous['integration_only'] or previous['status'] != 'success':
            raise ValueError(f'Existing run cannot be reused: {folder}')
        print('SKIP VERIFIED RUN', category, shots, seed, flush=True)
        return previous
    manifest = load_manifest(category)
    if not (folder / 'run.json').exists():
        names = manifest['references'][str(shots)][str(seed)]
        started = time.perf_counter()
        detector = TemplateDetector(reference_paths(manifest, shots, seed, data_root), names)
        bank_seconds = time.perf_counter() - started
        # All warmup inputs are fixed calibration normals; none change the model.
        for path in calibration_paths(manifest, data_root)[:3]:
            detector.predict_path(path)
        writer = PredictionWriter(folder, category, shots, seed, METHOD, config['config_id'], config)
        choices = []
        for relative, path in zip(manifest['calibration'], calibration_paths(manifest, data_root)):
            prediction = detector.predict_path(path)
            writer.write_calibration(relative, prediction.score)
            choices.append({'path': relative, 'split': 'calibration', 'reference': prediction.chosen_reference,
                            'angle': prediction.chosen_angle, 'descriptor_mse': prediction.descriptor_mse,
                            'input_shape': list(prediction.input_shape)})
        durations, peak_rss = [], psutil.Process().memory_info().rss
        for index, sample in enumerate(manifest['test']):
            started = time.perf_counter()
            prediction = detector.predict_path(inside(data_root, sample['path']))
            elapsed = time.perf_counter() - started
            writer.write_test(sample['path'], prediction.native_map, prediction.score, elapsed)
            durations.append(elapsed)
            peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
            choices.append({'path': sample['path'], 'split': 'test', 'reference': prediction.chosen_reference,
                            'angle': prediction.chosen_angle, 'descriptor_mse': prediction.descriptor_mse,
                            'input_shape': list(prediction.input_shape)})
            if (index + 1) % 50 == 0 or index == len(manifest['test']) - 1:
                print('PREDICT', category, shots, seed, index + 1, '/', len(manifest['test']), flush=True)
        writer.finish({'device': 'CPU', 'gpu_used': False, 'peak_gpu_allocated_mb': 0.,
                       'cpu_threads_opencv': CONFIG['opencv_threads'], 'warmup_images': 3,
                       'inference_seconds_mean': float(np.mean(durations)),
                       'inference_seconds_std': float(np.std(durations, ddof=0)),
                       'reference_build_seconds': bank_seconds,
                       'peak_process_rss_during_inference_mb': peak_rss / 1024**2,
                       'timing_includes': 'JPG reading, RGB conversion, resize/crop, nearest-template selection, raw difference, smoothing, morphology, native-map interpolation',
                       'timing_excludes': 'reference setup, warmup, calibration, map serialization, metric evaluation',
                       'environment': environment()})
        save_json(folder / 'template_choices.json', choices)
        del detector
    print('EVALUATE NATIVE ALL-TEST PIXELS', category, shots, seed, flush=True)
    started = time.perf_counter()
    result = evaluate_file(folder, data_root=data_root, optional_metrics=True)
    result['metric_evaluation_seconds'] = time.perf_counter() - started
    result['experiment_scope'] = 'formal_part1_template'
    save_json(metric_file, result)
    save_json(compact / 'metrics.json', result)
    raw = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
    choices = {r['path']: r for r in json.loads((folder / 'template_choices.json').read_text(encoding='utf-8'))}
    compact.mkdir(parents=True, exist_ok=True)
    with (compact / 'image_predictions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ['path', 'label', 'image_score', 'threshold', 'predicted_ng', 'reference', 'angle', 'inference_seconds']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sample, prediction in zip(manifest['test'], raw['test']):
            writer.writerow({'path': sample['path'], 'label': sample['label'], 'image_score': prediction['score'],
                             'threshold': result['image_threshold'], 'predicted_ng': int(prediction['score'] > result['image_threshold']),
                             'reference': choices[sample['path']]['reference'], 'angle': choices[sample['path']]['angle'],
                             'inference_seconds': prediction['inference_seconds']})
    print('RESULT', category, shots, seed, 'I-AUROC', round(result['image_auroc'], 5),
          'P-AUROC', round(result['pixel_auroc'], 5), flush=True)
    return result


def update_progress(output_root, result_root, config):
    records = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(output_root.glob('*/*/metrics.json'))]
    for row in records:
        if row['run_config'] != config:
            raise ValueError('Mixed configuration in output root')
    progress = {'completed': len(records), 'required': 72, 'updated_utc': datetime.now(timezone.utc).isoformat(),
                'completed_runs': [{'category': r['category'], 'shots': r['shots'], 'seed': r['seed']} for r in records]}
    save_json(output_root / 'progress.json', progress)
    save_json(result_root / 'progress.json', progress)
    if len(records) == 72:
        summaries = aggregate_runs(records, CATEGORIES)
        save_json(result_root / 'summary.json', summaries)
        table = []
        for summary in summaries:
            for category in CATEGORIES:
                selected = [r for r in records if r['category'] == category and r['shots'] == summary['shots']]
                row = {'category': category, 'shots': summary['shots']}
                for key in ['image_auroc', 'pixel_auroc', 'image_ap', 'pixel_ap', 'aupro_03', 'image_fpr', 'image_fnr']:
                    row[key + '_mean'] = summary[key]['per_category'][category]['mean']
                    row[key + '_std'] = summary[key]['per_category'][category]['std']
                row['inference_ms_mean'] = float(np.mean([r['resources']['inference_seconds_mean'] * 1000 for r in selected]))
                row['peak_gpu_mb_max'] = max(r['resources']['peak_gpu_allocated_mb'] for r in selected)
                table.append(row)
            macro = {'category': 'six_category_macro', 'shots': summary['shots']}
            for key in ['image_auroc', 'pixel_auroc', 'image_ap', 'pixel_ap', 'aupro_03', 'image_fpr', 'image_fnr']:
                macro[key + '_mean'], macro[key + '_std'] = summary[key]['mean'], summary[key]['std']
            macro['inference_ms_mean'] = float(np.mean([r['resources']['inference_seconds_mean'] * 1000 for r in records if r['shots'] == summary['shots']]))
            macro['peak_gpu_mb_max'] = 0.
            table.append(macro)
        with (result_root / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        print('ALL 72 FORMAL RUNS COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--categories', nargs='+', choices=CATEGORIES, default=list(CATEGORIES))
    parser.add_argument('--shots', nargs='+', type=int, choices=SHOTS, default=list(SHOTS))
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/part1_template_v1')
    parser.add_argument('--result-root', type=Path, default=ROOT / 'results/part1_template')
    args = parser.parse_args()
    cv2.setNumThreads(CONFIG['opencv_threads'])
    config = frozen_config()
    freeze(args.output_root, args.result_root, config)
    # Conservative raw-map disk estimate; no lossy compression or GT downsampling.
    pending_bytes = sum(sum(s['height'] * s['width'] * 4 for s in load_manifest(c)['test'])
                        for c in args.categories for k in args.shots for seed in args.seeds
                        if not (args.output_root / c / f'{k}shot_seed{seed}' / 'run.json').exists())
    if shutil.disk_usage(args.output_root).free < pending_bytes + 5 * 1024**3:
        raise ValueError('Insufficient disk for all requested native maps plus a 5 GiB reserve')
    for category in args.categories:
        for shots in args.shots:
            for seed in args.seeds:
                run_one(category, shots, seed, args.data_root, args.output_root, args.result_root, config)
                update_progress(args.output_root, args.result_root, config)


if __name__ == '__main__':
    main()
