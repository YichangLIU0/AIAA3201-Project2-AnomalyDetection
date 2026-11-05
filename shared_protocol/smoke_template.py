"""Real-data Template integration check; not the final Part 1 experiment grid."""
from pathlib import Path
import argparse
import json
import time

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, grey_closing

from .data import load_manifest, load_rgb, reference_paths, calibration_paths
from .evaluate_predictions import evaluate_file
from .predictions import PredictionWriter
from .protocol import ROOT, DATA_ROOT, CATEGORIES, inside, save_json


def input_rgb(rgb):
    """Official DINO geometric convention: short edge 448, then crop to 14 multiples.

    Template comparison uses RGB values; DINO's own normalization/features remain
    B's responsibility. No labels are consulted to choose this fixed smoke config.
    """
    height, width = rgb.shape[:2]
    if height <= width:
        new_height, new_width = 448, int(448 * width / height)
    else:
        new_height, new_width = int(448 * height / width), 448
    resized = np.asarray(Image.fromarray(rgb).resize((new_width, new_height), Image.Resampling.BICUBIC))
    return resized[:new_height - new_height % 14, :new_width - new_width % 14].copy()


def predict(path, reference_images):
    started = time.perf_counter()
    original = load_rgb(path)
    query = input_rgb(original)
    references = [r if r.shape == query.shape else np.asarray(Image.fromarray(r).resize(
                    (query.shape[1], query.shape[0]), Image.Resampling.BICUBIC)) for r in reference_images]
    descriptor = np.asarray(Image.fromarray(query).resize((28, 28)), dtype=np.float32)
    descriptors = [np.asarray(Image.fromarray(r).resize((28, 28)), dtype=np.float32) for r in references]
    nearest = int(np.argmin([np.mean((descriptor - r) ** 2) for r in descriptors]))
    difference = np.abs(query.astype(np.float32) - references[nearest].astype(np.float32)).mean(axis=2) / 255.
    difference = grey_closing(gaussian_filter(difference, sigma=2), size=(3, 3))
    flat = difference.ravel()
    k = max(1, int(len(flat) * .01))
    score = float(np.partition(flat, len(flat) - k)[-k:].mean())
    # Only the prediction is aligned to the original image; GT is never resized.
    native = np.asarray(Image.fromarray(difference).resize((original.shape[1], original.shape[0]),
                                                         Image.Resampling.BILINEAR), dtype=np.float32)
    return native, score, time.perf_counter() - started


def run(category='candle', shots=1, seed=0, data_root=DATA_ROOT, output=None):
    output = Path(output or ROOT / f'outputs/integration_only/template_{category}_{shots}shot_seed{seed}')
    manifest = load_manifest(category)
    config = {'purpose': 'public-interface integration check, not final Part 1',
              'input_resolution': 448, 'input_geometry': 'short_edge_448_then_top_left_crop_to_patch14_multiple',
              'representation': 'RGB, fixed global division by 255',
              'reference_selection': 'shared official list', 'template_choice': 'nearest 28x28 RGB descriptor',
              'alignment': 'none', 'reference_augmentation': 'none', 'gaussian_sigma': 2,
              'morphology': '3x3 grayscale closing', 'image_score': 'mean top 1 percent of processed difference map',
              'prediction_alignment': 'Pillow bilinear to native original image',
              'device': 'CPU', 'weights': 'none'}
    writer = PredictionWriter(output, category, shots, seed, 'template_integration', 'fixed_template_smoke_v2',
                              config, integration_only=True)
    reference_images = [input_rgb(load_rgb(path)) for path in reference_paths(manifest, shots, seed, data_root)]
    for relative, path in zip(manifest['calibration'], calibration_paths(manifest, data_root)):
        _, score, _ = predict(path, reference_images)
        writer.write_calibration(relative, score)
    print('CALIBRATION COMPLETE', 20, flush=True)
    durations = []
    for index, sample in enumerate(manifest['test']):
        native, score, duration = predict(inside(data_root, sample['path']), reference_images)
        writer.write_test(sample['path'], native, score, duration)
        durations.append(duration)
        if (index + 1) % 40 == 0:
            print('REAL TEMPLATE', index + 1, '/', len(manifest['test']), flush=True)
    writer.finish({'device': 'CPU', 'inference_seconds_mean': float(np.mean(durations)),
                   'peak_gpu_allocated_mb': 0., 'gpu_used': False,
                   'timing_includes': 'image reading, preprocessing, template selection, difference, refinement, native map resize',
                   'timing_excludes': 'reference setup, prediction file writing and metric evaluation'})
    print('EVALUATING NATIVE MAPS', flush=True)
    result = evaluate_file(output, data_root=data_root)
    save_json(output / 'metrics.json', result)
    save_json(ROOT / 'shared_protocol/reports/template_integration.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--category', choices=CATEGORIES, default='candle')
    parser.add_argument('--shots', type=int, choices=[1, 2, 4, 8], default=1)
    parser.add_argument('--seed', type=int, choices=[0, 1, 2], default=0)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run(args.category, args.shots, args.seed, args.data_root, args.output)
