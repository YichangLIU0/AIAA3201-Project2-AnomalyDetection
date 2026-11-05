"""Shared runner: use a fixed normal bank, calibrate on 20, evaluate every test image."""
import time

from .data import load_manifest
from .predictions import PredictionWriter
from .evaluate_predictions import evaluate_file
from .protocol import DATA_ROOT, MANIFEST_ROOT, inside, save_json


def run_detector(predict_one, output, category, shots, seed, method, config_id, run_config,
                 data_root=DATA_ROOT, manifest_root=MANIFEST_ROOT, synchronize=None,
                 resource_reader=None, optional_metrics=True, integration_only=False, experiment_mode=None):
    """predict_one(Path) -> (raw_image_score, native_anomaly_map).

    For few-shot, build the normal bank before this call using reference_paths().
    For zero-shot, pass shots=0, seed=0 and do not build any reference bank.
    Both modes use 20 normals only for threshold calibration, never training.
    No GT or test label is passed into predict_one().
    For CUDA timing, pass synchronize=torch.cuda.synchronize. Reset GPU peak
    counters and warm up your model before calling; report bank setup separately.
    """
    manifest = load_manifest(category, manifest_root)
    writer = PredictionWriter(output, category, shots, seed, method, config_id, run_config,
                              manifest_root, integration_only, experiment_mode)
    for relative in manifest['calibration']:
        score, _ = predict_one(inside(data_root, relative))
        writer.write_calibration(relative, score)
    durations = []
    for sample in manifest['test']:
        if synchronize:
            synchronize()
        started = time.perf_counter()
        score, anomaly_map = predict_one(inside(data_root, sample['path']))
        if synchronize:
            synchronize()
        duration = time.perf_counter() - started
        durations.append(duration)
        writer.write_test(sample['path'], anomaly_map, score, duration)
    resources = resource_reader() if resource_reader else {}
    resources.update(inference_seconds_mean=sum(durations) / len(durations),
                     timing_includes='predict_one call and optional device synchronization',
                     timing_excludes='bank construction, calibration, map file writing, metric evaluation',
                     cuda_synchronization_provided=synchronize is not None)
    writer.finish(resources)
    result = evaluate_file(output, data_root=data_root, manifest_root=manifest_root, optional_metrics=optional_metrics)
    save_json(writer.output / 'metrics.json', result)
    return result
