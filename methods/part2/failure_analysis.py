"""Part 2.5 (member B): post-hoc failure analysis of the formal DINO and AnomalyCLIP runs. CPU only.

Test labels and masks are used here ONLY to explain errors after evaluation. Nothing from this script may
feed back into a threshold, prompt, setting or method choice.

Writes to <out>:
  panels/<category>_<kind>_<image>.png   input | GT | DINO 1-shot | DINO 8-shot | AnomalyCLIP
                                         kinds: fn (lowest-scored anomaly, 1-shot seed 0),
                                         fp (highest-scored normal), small (smallest defect)
  defect_size.csv    image detection rate and map-peak-in-GT rate by defect size in DINO patches
  normal_shift.csv   calibration vs normal-test scores: why FPR can exceed the nominal 5%
  samples.csv        every selected panel with its scores and decisions
Colour scales are shared within one panel row and are for display only.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared_protocol.data import load_manifest, load_mask  # noqa: E402
from shared_protocol.protocol import CATEGORIES, DATA_ROOT, SEEDS  # noqa: E402

PATCH, SHORT_SIDE = 14, 448
SIZE_BINS = [(0, 1), (1, 4), (4, 16), (16, 64), (64, 256), (256, float('inf'))]


def load_run(folder):
    folder = Path(folder)
    run = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
    met = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
    return {'folder': folder, 'threshold': met['image_threshold'], 'calibration': np.array(run['calibration_scores']),
            'items': {t['path']: t for t in run['test']}}


def load_map(run, path):
    return np.load(run['folder'] / run['items'][path]['map'])


def defect_patches(mask):
    """Defect area in DINO patch units: 14 px at short side 448, mapped back to native pixels."""
    native_patch = PATCH * min(mask.shape) / SHORT_SIDE
    return float(mask.sum()) / native_patch ** 2


def panel(plt, Image, out, category, sample, title, runs, mask):
    path = sample['path']
    image = np.asarray(Image.open(DATA_ROOT / path).convert('RGB'))
    overlay = image.copy()
    overlay[mask] = (0.4 * overlay[mask] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
    fig, axes = plt.subplots(1, 2 + len(runs), figsize=(3.3 * (2 + len(runs)), 3.6))
    axes[0].imshow(image); axes[0].set_title('input', fontsize=9)
    axes[1].imshow(overlay); axes[1].set_title('ground truth (red)' if mask.any() else 'normal (no defect)', fontsize=9)
    for axis, (name, run) in zip(axes[2:], runs.items()):
        amap = load_map(run, path)
        im = axis.imshow(amap, cmap='magma', vmin=np.percentile(amap, 1), vmax=np.percentile(amap, 99.9))
        score, thr = run['items'][path]['score'], run['threshold']
        axis.set_title(f"{name}\nscore {score:.3f} thr {thr:.3f}: {'NG' if score > thr else 'OK'}", fontsize=9)
    for axis in axes:
        axis.axis('off')
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dino', default=str(ROOT / 'outputs/dino'))
    ap.add_argument('--clip', default=str(ROOT / 'outputs/anomalyclip'))
    ap.add_argument('--out', default=str(ROOT / 'outputs/failure_analysis'))
    args = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    out = Path(args.out)
    (out / 'panels').mkdir(parents=True, exist_ok=True)
    dino, clip = Path(args.dino), Path(args.clip)
    size_rows, shift_rows, sample_rows = [], [], []
    tallies = {}

    for category in CATEGORIES:
        manifest = load_manifest(category)
        tests = manifest['test']
        masks = {s['path']: load_mask(s, DATA_ROOT) for s in tests}
        patches = {s['path']: defect_patches(masks[s['path']]) for s in tests if s['label'] == 1}

        configs = {f'dino_{k}shot_seed{s}': dino / category / f'{k}shot_seed{s}' for k in (1, 8) for s in SEEDS}
        configs['clip'] = clip / category / 'zero_shot_seed0'
        for name, folder in configs.items():
            run = load_run(folder)
            family = 'clip' if name == 'clip' else name.rsplit('_seed', 1)[0]
            for s in tests:
                if s['label'] != 1:
                    continue
                path = s['path']
                amap = load_map(run, path)
                row, col = np.unravel_index(np.argmax(amap), amap.shape)
                detected = run['items'][path]['score'] > run['threshold']
                in_gt = bool(masks[path][row, col])
                size = patches[path]
                index = next(i for i, (lo, hi) in enumerate(SIZE_BINS) if lo <= size < hi)
                t = tallies.setdefault((family, index), [0, 0, 0, 0, 0])
                t[0] += 1; t[1] += int(detected); t[2] += int(in_gt)
                t[3] += int(in_gt and not detected); t[4] += int(detected and not in_gt)
            normal = np.array([run['items'][s['path']]['score'] for s in tests if s['label'] == 0])
            cal = run['calibration']
            shift_rows.append({'category': category, 'config': name, 'threshold': run['threshold'],
                               'cal_median': float(np.median(cal)), 'cal_max': float(cal.max()),
                               'normal_test_median': float(np.median(normal)),
                               'normal_test_p95': float(np.percentile(normal, 95)),
                               'median_shift_test_minus_cal': float(np.median(normal) - np.median(cal)),
                               'calibration_fpr': float((cal > run['threshold']).mean()),
                               'test_fpr': float((normal > run['threshold']).mean())})

        runs = {'DINO 1-shot seed0': load_run(dino / category / '1shot_seed0'),
                'DINO 8-shot seed0': load_run(dino / category / '8shot_seed0'),
                'AnomalyCLIP': load_run(clip / category / 'zero_shot_seed0')}
        base = runs['DINO 1-shot seed0']
        anomalies = [s for s in tests if s['label'] == 1]
        normals = [s for s in tests if s['label'] == 0]
        picks = {'fn': min(anomalies, key=lambda s: base['items'][s['path']]['score']),
                 'fp': max(normals, key=lambda s: base['items'][s['path']]['score']),
                 'small': min(anomalies, key=lambda s: patches[s['path']])}
        for kind, sample in picks.items():
            path = sample['path']
            stem = Path(path).stem
            size_note = f', defect {patches[path]:.1f} patches' if sample['label'] == 1 else ''
            panel(plt, Image, out / 'panels' / f'{category}_{kind}_{stem}.png', category, sample,
                  f'{category} {kind}: {path}{size_note}', runs, masks[path])
            row = {'category': category, 'kind': kind, 'path': path, 'label': sample['label'],
                   'defect_patches': patches.get(path)}
            for name, run in runs.items():
                row[name + ' score'] = run['items'][path]['score']
                row[name + ' NG'] = run['items'][path]['score'] > run['threshold']
            sample_rows.append(row)
        print('done', category, flush=True)

    for (family, index), (n, detected, in_gt, missed_localised, detected_elsewhere) in sorted(tallies.items()):
        lo, hi = SIZE_BINS[index]
        size_rows.append({'config': family, 'defect_patches': f'{lo}-{hi}', 'n': n,
                          'image_detection_rate': detected / n, 'map_peak_in_gt_rate': in_gt / n,
                          'missed_but_peak_in_gt': missed_localised, 'detected_but_peak_outside_gt': detected_elsewhere})
    for name, rows in (('defect_size', size_rows), ('normal_shift', shift_rows), ('samples', sample_rows)):
        with (out / f'{name}.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print('FAILURE_ANALYSIS_DONE', flush=True)


if __name__ == '__main__':
    main()
