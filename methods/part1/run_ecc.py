"""Predeclared 8-shot / 6-class / 3-seed ECC paired experiment (18 cells)."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse
import csv
import hashlib
import json
import shutil
import time

import cv2
import numpy as np
import psutil

from shared_protocol.data import load_manifest, reference_paths
from shared_protocol.predictions import PredictionWriter
from shared_protocol.evaluate_predictions import evaluate_file
from shared_protocol.protocol import ROOT, CATEGORIES, inside, save_json, sha256
from shared_protocol.metrics import aggregate_runs
from .ecc import CONFIG, ECCDetector
from .run_template import environment, freeze

HERE = Path(__file__).resolve().parent


def configuration():
    return {**CONFIG, 'experiment_shots': [8], 'experiment_seeds': [0, 1, 2],
            'config_id': 'template_ecc_' + hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:12],
            'algorithm_source_sha256': {name: sha256(HERE / name) for name in
                ['template.py', 'run_template.py', 'ecc.py', 'run_ecc.py']}}


def export_compact(folder, compact, manifest, result):
    raw = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
    traces = json.loads((folder / 'alignment_traces.json').read_text(encoding='utf-8'))
    choices = {r['path']: r for r in traces}
    save_json(compact / 'metrics.json', result)
    save_json(compact / 'prediction_manifest.json', raw)
    save_json(compact / 'alignment_traces.json', traces)
    with (compact / 'calibration_scores.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['path', 'score'])
        writer.writerows(zip(raw['calibration_paths'], raw['calibration_scores']))
    with (compact / 'image_predictions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ['path', 'label', 'score', 'threshold', 'predicted_ng', 'reference', 'angle', 'alignment_status', 'fallback_reason', 'inference_seconds']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for sample, pred in zip(manifest['test'], raw['test']):
            choice = choices[sample['path']]
            writer.writerow(dict(path=sample['path'], label=sample['label'], score=pred['score'],
                                 threshold=result['image_threshold'], predicted_ng=int(pred['score'] > result['image_threshold']),
                                 reference=choice['reference'], angle=choice['angle'],
                                 alignment_status=choice['alignment']['status'], fallback_reason=choice['alignment']['reason'],
                                 inference_seconds=pred['inference_seconds']))


def run_cell(category, seed, args, config):
    folder = args.output_root / category / f'8shot_seed{seed}'
    compact = args.result_root / 'runs' / category / folder.name
    manifest = load_manifest(category)
    metric_path = folder / 'metrics.json'
    if metric_path.exists():
        result = json.loads(metric_path.read_text(encoding='utf-8'))
        if result['run_config'] != config or result['integration_only'] or result['status'] != 'success':
            raise ValueError('Refuse incompatible cached ECC result')
        export_compact(folder, compact, manifest, result)
        print('SKIP COMPLETE', category, seed, flush=True)
        return
    baseline_folder = args.baseline_root / category / folder.name
    baseline = json.loads((baseline_folder / 'run.json').read_text(encoding='utf-8'))
    expected_choices = {r['path']: r for r in json.loads((baseline_folder / 'template_choices.json').read_text(encoding='utf-8'))}
    assert baseline['reference_paths'] == manifest['references']['8'][str(seed)]
    assert not baseline['integration_only'] and baseline['status'] == 'complete'
    if not (folder / 'run.json').exists():
        started = time.perf_counter()
        model = ECCDetector(reference_paths(manifest, 8, seed, args.data_root), manifest['references']['8'][str(seed)])
        setup_seconds = time.perf_counter() - started
        for relative in manifest['calibration'][:3]:
            model.predict_path(inside(args.data_root, relative))
        writer = PredictionWriter(folder, category, 8, seed, CONFIG['method'], config['config_id'], config)
        traces, durations = [], []
        peak_rss = psutil.Process().memory_info().rss
        for split, samples in [('calibration', [{'path': p} for p in manifest['calibration']]), ('test', manifest['test'])]:
            for index, sample in enumerate(samples):
                started = time.perf_counter()
                prediction = model.predict_path(inside(args.data_root, sample['path']))
                elapsed = time.perf_counter() - started
                expected = expected_choices[sample['path']]
                assert (prediction.chosen_reference, prediction.chosen_angle) == (expected['reference'], expected['angle'])
                trace = dict(path=sample['path'], split=split, reference=prediction.chosen_reference,
                             angle=prediction.chosen_angle, alignment=prediction.alignment_trace)
                traces.append(trace)
                if split == 'calibration':
                    writer.write_calibration(sample['path'], prediction.score)
                else:
                    writer.write_test(sample['path'], prediction.native_map, prediction.score, elapsed)
                    durations.append(elapsed)
                    peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
                    if (index+1) % 50 == 0 or index == len(samples)-1:
                        print('ECC PREDICT', category, seed, index+1, '/', len(samples), flush=True)
        save_json(folder / 'alignment_traces.json', traces)
        writer.finish(dict(device='CPU', gpu_used=False, peak_gpu_allocated_mb=0.,
                           cpu_threads_opencv=CONFIG['opencv_threads'], warmup_images=3,
                           inference_seconds_mean=float(np.mean(durations)), inference_seconds_std=float(np.std(durations)),
                           reference_build_seconds=setup_seconds,
                           peak_process_rss_during_inference_mb=peak_rss/1024**2,
                           timing_includes='JPG decode, preprocess, identical nearest-template selection, ECC solve and acceptance checks, difference and postprocess, native-map interpolation',
                           timing_excludes='reference setup, warmup, calibration, disk serialization, metric evaluation',
                           environment=environment()))
        del model
    traces = json.loads((folder / 'alignment_traces.json').read_text(encoding='utf-8'))
    print('ECC EVALUATE ALL NATIVE PIXELS', category, seed, flush=True)
    started = time.perf_counter()
    result = evaluate_file(folder, data_root=args.data_root, optional_metrics=True)
    result['metric_evaluation_seconds'] = time.perf_counter()-started
    result['experiment_scope'] = 'formal_8shot_ecc_paired_18cells_not_full_shot_grid'
    result['baseline_prediction_manifest_sha256'] = sha256(baseline_folder / 'run.json')
    for split in ['calibration', 'test']:
        subset = [t['alignment'] for t in traces if t['split'] == split]
        result['alignment_counts_'+split] = dict(Counter(t['status'] for t in subset))
        result['fallback_reasons_'+split] = dict(Counter(t['reason'] for t in subset if t['status'] == 'fallback'))
    save_json(metric_path, result)
    export_compact(folder, compact, manifest, result)
    print('ECC RESULT', category, seed, 'Image', round(result['image_auroc'],5), 'Pixel', round(result['pixel_auroc'],5),
          'alignment',result['alignment_counts_test'], flush=True)


def progress(args, config):
    records = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(args.output_root.glob('*/*/metrics.json'))]
    assert all(r['run_config'] == config for r in records)
    record = dict(completed=len(records), required=18, shots=[8], seeds=[0,1,2],
                  updated_utc=datetime.now(timezone.utc).isoformat(),
                  completed_runs=[dict(category=r['category'],seed=r['seed']) for r in records])
    save_json(args.output_root/'progress.json',record); save_json(args.result_root/'progress.json',record)
    if len(records)==18:
        summary=aggregate_runs(records,CATEGORIES,shots=(8,),seeds=(0,1,2))
        save_json(args.result_root/'summary.json',summary)
        print('ALL 18 FORMAL ECC CELLS COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--categories',nargs='+',choices=CATEGORIES,default=list(CATEGORIES))
    parser.add_argument('--seeds',nargs='+',type=int,choices=[0,1,2],default=[0,1,2])
    parser.add_argument('--baseline-root',type=Path,default=ROOT/'outputs/part1_template_v1')
    parser.add_argument('--output-root',type=Path,default=ROOT/'outputs/part1_ecc_8shot_v1')
    parser.add_argument('--result-root',type=Path,default=ROOT/'results/part1_ecc_8shot')
    args=parser.parse_args();cv2.setNumThreads(CONFIG['opencv_threads'])
    config=configuration();freeze(args.output_root,args.result_root,config)
    estimate=sum(sum(s['height']*s['width']*4 for s in load_manifest(c)['test'])
                 for c in args.categories for seed in args.seeds
                 if not (args.output_root/c/f'8shot_seed{seed}'/'run.json').exists())
    if shutil.disk_usage(args.output_root).free < estimate+5*1024**3:
        raise ValueError('Insufficient native-map storage plus 5 GiB reserve')
    for category in args.categories:
        for seed in args.seeds:
            run_cell(category,seed,args,config);progress(args,config)


if __name__=='__main__':
    main()
