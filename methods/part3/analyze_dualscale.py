"""Post-hoc analysis of the dual-scale grid (member B, Part 3 direction A). CPU only, never reruns the model.

Test labels and masks are used ONLY here, after the design was frozen and all runs finished.

Writes to <out>/analysis/:
  aggregate_<config>.json      shared_protocol.metrics.aggregate_runs for s448 (recomputed), s672, fuse
  aggregate_formal448.json     the formal Part 2 baseline, aggregated the same way
  per_run.csv                  every run and configuration
  paired.json                  per-shot paired differences (fuse-448, 672-448, fuse-672), wins/losses, sign test
  defect_size.csv              detection rate and map-peak-in-GT rate by defect size (448 patch geometry), 1/8-shot
  flips.json                   errors fixed / introduced by fuse and s672 relative to s448 on the same images
  panels/                      input | GT | 448 | 672 | fuse for selected cases (8-shot, seed 0)

  python methods/part3/analyze_dualscale.py
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'shared_protocol').is_dir())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from shared_protocol.data import load_manifest, load_mask  # noqa: E402
from shared_protocol.metrics import aggregate_runs  # noqa: E402
from shared_protocol.protocol import CATEGORIES, SHOTS, SEEDS, DATA_ROOT, save_json  # noqa: E402
import run_dualscale as rd  # noqa: E402

NAMES = ('s448', 's672', 'fuse')
KEYS = ('image_auroc', 'pixel_auroc', 'image_ap', 'pixel_ap', 'aupro_03', 'image_fpr', 'image_fnr')
SIZE_BINS = [(0, 1), (1, 4), (4, 16), (16, 64), (64, 256), (256, float('inf'))]
PAIRS = (('fuse', 's448'), ('s672', 's448'), ('fuse', 's672'))


def sign_test(wins, losses):
    from scipy.stats import binomtest
    n = wins + losses
    return None if n == 0 else float(binomtest(wins, n, 0.5).pvalue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(ROOT / 'outputs' / 'part3_dualscale'))
    ap.add_argument('--formal-dir', default=str(ROOT / 'outputs' / 'dino'))
    ap.add_argument('--no-panels', action='store_true')
    args = ap.parse_args()
    out, formal = Path(args.out_dir), Path(args.formal_dir)
    ana = out / 'analysis'
    (ana / 'panels').mkdir(parents=True, exist_ok=True)
    _, dists2map, _ = rd.rdo.import_upstream()

    metrics = {n: [] for n in NAMES}
    formal_rows, table = [], []
    index = {}
    for c in CATEGORIES:
        for k in SHOTS:
            for s in SEEDS:
                d = out / 'runs' / c / f'{k}shot_seed{s}'
                for n in NAMES:
                    m = json.loads((d / f'metrics_{n}.json').read_text(encoding='utf-8'))
                    metrics[n].append(m)
                    index[(n, c, k, s)] = m
                fm = json.loads((formal / c / f'{k}shot_seed{s}' / 'metrics.json').read_text(encoding='utf-8'))
                formal_rows.append(fm)
                for n in NAMES:
                    m = index[(n, c, k, s)]
                    table.append({'category': c, 'shots': k, 'seed': s, 'config': n,
                                  **{key: m.get(key) for key in KEYS}, 'threshold': m['image_threshold']})
    for n in NAMES:
        save_json(ana / f'aggregate_{n}.json', aggregate_runs(metrics[n], CATEGORIES))
    save_json(ana / 'aggregate_formal448.json', aggregate_runs(formal_rows, CATEGORIES))
    with (ana / 'per_run.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    paired = {}
    for a, b in PAIRS:
        for k in list(SHOTS) + ['all']:
            ks = SHOTS if k == 'all' else (k,)
            entry = {}
            for key in KEYS:
                diffs = [index[(a, c, kk, s)][key] - index[(b, c, kk, s)][key] for c in CATEGORIES for kk in ks for s in SEEDS]
                better = (lambda x: x < 0) if key in ('image_fpr', 'image_fnr') else (lambda x: x > 0)
                wins = sum(better(x) for x in diffs)
                losses = sum(x != 0 and not better(x) for x in diffs)
                macro = [float(np.mean([index[(a, c, kk, s)][key] - index[(b, c, kk, s)][key] for c in CATEGORIES]))
                         for kk in ks for s in SEEDS]
                entry[key] = {'mean_diff': float(np.mean(diffs)), 'per_seed_macro_diff': macro,
                              'n_runs': len(diffs), 'wins': wins, 'losses': losses, 'ties': len(diffs) - wins - losses,
                              'sign_test_p': sign_test(wins, losses)}
            paired[f'{a}_minus_{b}_shots{k}'] = entry
    save_json(ana / 'paired.json', {'note': 'wins = runs where the first configuration is better (higher AUROC/AP, '
                                            'lower FPR/FNR). Runs share categories and references, so they are not '
                                            'independent samples; the sign test is a rough guide only.', **paired})

    # image-level and localisation analysis for 1- and 8-shot, all seeds, using maps regenerated from the grids
    tallies, flips, cases = {}, {}, []
    for c in CATEGORIES:
        manifest = load_manifest(c)
        tests = manifest['test']
        masks = {i: load_mask(sample, DATA_ROOT) for i, sample in enumerate(tests) if sample['label'] == 1}
        size = {i: float(m.sum()) / (14 * min(m.shape) / 448) ** 2 for i, m in masks.items()}
        for k in (1, 8):
            for s in SEEDS:
                d = out / 'runs' / c / f'{k}shot_seed{s}'
                run = json.loads((d / 'run.json').read_text(encoding='utf-8'))
                grids = np.load(d / run['grids_file'])
                thr = {n: index[(n, c, k, s)]['image_threshold'] for n in NAMES}
                score = {n: run['configs'][n]['test_scores'] for n in NAMES}
                decide = {n: [sc > thr[n] for sc in score[n]] for n in NAMES}
                for i, sample in enumerate(tests):
                    if sample['label'] == 1:
                        d4 = grids[f'test448_{i:04d}'].astype(np.float32)
                        d6 = grids[f'test672_{i:04d}'].astype(np.float32)
                        b = next(j for j, (lo, hi) in enumerate(SIZE_BINS) if lo <= size[i] < hi)
                        for n in NAMES:
                            amap = dists2map(rd.config_grid(n, d4, d6), (sample['height'], sample['width']))
                            r, col = np.unravel_index(np.argmax(amap), amap.shape)
                            in_gt = bool(masks[i][r, col])
                            t = tallies.setdefault((n, k, b), [0, 0, 0, 0, 0])
                            t[0] += 1; t[1] += decide[n][i]; t[2] += in_gt
                            t[3] += int(in_gt and not decide[n][i]); t[4] += int(decide[n][i] and not in_gt)
                    for a in ('fuse', 's672'):
                        key = f'{a}_vs_s448_shots{k}'
                        f = flips.setdefault(key, {'fn_fixed': 0, 'fn_introduced': 0, 'fp_fixed': 0, 'fp_introduced': 0,
                                                   'fn_fixed_by_size': [0] * len(SIZE_BINS),
                                                   'fn_introduced_by_size': [0] * len(SIZE_BINS)})
                        base, new = decide['s448'][i], decide[a][i]
                        if sample['label'] == 1:
                            b = next(j for j, (lo, hi) in enumerate(SIZE_BINS) if lo <= size[i] < hi)
                            if not base and new:
                                f['fn_fixed'] += 1; f['fn_fixed_by_size'][b] += 1
                            elif base and not new:
                                f['fn_introduced'] += 1; f['fn_introduced_by_size'][b] += 1
                        else:
                            if base and not new:
                                f['fp_fixed'] += 1
                            elif not base and new:
                                f['fp_introduced'] += 1
                    if k == 8 and s == 0:
                        margin = {n: score[n][i] - thr[n] for n in NAMES}
                        cases.append({'category': c, 'index': i, 'path': sample['path'], 'label': sample['label'],
                                      'defect_patches': size.get(i), **{f'margin_{n}': margin[n] for n in NAMES},
                                      **{f'ng_{n}': decide[n][i] for n in NAMES}})
    rows = []
    for (n, k, b), (cnt, det, ing, miss_in, det_out) in sorted(tallies.items()):
        lo, hi = SIZE_BINS[b]
        rows.append({'config': n, 'shots': k, 'defect_patches_448': f'{lo}-{hi}', 'n': cnt,
                     'image_detection_rate': det / cnt, 'map_peak_in_gt_rate': ing / cnt,
                     'missed_but_peak_in_gt': miss_in, 'detected_but_peak_outside_gt': det_out})
    with (ana / 'defect_size.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    save_json(ana / 'flips.json', {'size_bins_448_patches': [f'{lo}-{hi}' for lo, hi in SIZE_BINS],
                                   'note': 'counted over 3 seeds; the same image appears once per seed', **flips})

    if not args.no_panels:
        make_panels(cases, out, dists2map, index)
    print(json.dumps({k: {m: round(v['mean_diff'], 4) for m, v in e.items()} for k, e in paired.items()
                      if k.endswith('shotsall')}, indent=1))
    print(json.dumps(flips, indent=1))
    print('ANALYSIS_DONE')


def make_panels(cases, out, dists2map, index):
    """Per category, 8-shot seed 0: the anomaly whose decision fuse fixed with the largest margin gain, the case it broke
    (anomaly missed or normal falsely flagged) with the largest loss, and the smallest defect."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    chosen = []
    for c in CATEGORIES:
        rows = [r for r in cases if r['category'] == c]
        fixed = [r for r in rows if r['label'] == 1 and r['ng_fuse'] and not r['ng_s448']]
        broke = [r for r in rows if (r['label'] == 1 and r['ng_s448'] and not r['ng_fuse'])
                 or (r['label'] == 0 and r['ng_fuse'] and not r['ng_s448'])]
        small = [r for r in rows if r['label'] == 1]
        if fixed:
            chosen.append(('fixed', max(fixed, key=lambda r: r['margin_fuse'] - r['margin_s448'])))
        if broke:
            chosen.append(('broke', min(broke, key=lambda r: abs(r['margin_fuse']) - abs(r['margin_s448']))))
        if small:
            chosen.append(('small', min(small, key=lambda r: r['defect_patches'])))
    listing = []
    for kind, r in chosen:
        d = out / 'runs' / r['category'] / '8shot_seed0'
        run = json.loads((d / 'run.json').read_text(encoding='utf-8'))
        grids = np.load(d / run['grids_file'])
        i = r['index']
        image = np.asarray(Image.open(DATA_ROOT / r['path']).convert('RGB'))
        sample = load_manifest(r['category'])['test'][i]
        mask = load_mask(sample, DATA_ROOT)
        overlay = image.copy()
        overlay[mask] = (0.4 * overlay[mask] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
        d4, d6 = grids[f'test448_{i:04d}'].astype(np.float32), grids[f'test672_{i:04d}'].astype(np.float32)
        maps = {n: dists2map(rd.config_grid(n, d4, d6), image.shape[:2]) for n in NAMES}
        vmax = max(np.percentile(m, 99.9) for m in maps.values())
        fig, axes = plt.subplots(1, 5, figsize=(17, 3.6))
        axes[0].imshow(image); axes[0].set_title('input', fontsize=9)
        axes[1].imshow(overlay); axes[1].set_title('ground truth (red)' if mask.any() else 'normal (no defect)', fontsize=9)
        for ax, n in zip(axes[2:], NAMES):
            ax.imshow(maps[n], cmap='magma', vmin=0, vmax=vmax)
            thr = index[(n, r['category'], 8, 0)]['image_threshold']
            sc = run['configs'][n]['test_scores'][i]
            ax.set_title(f"{n}\nscore {sc:.3f} thr {thr:.3f}: {'NG' if sc > thr else 'OK'}", fontsize=9)
        for ax in axes:
            ax.axis('off')
        size_note = f", defect {r['defect_patches']:.1f} patches (448)" if r['label'] == 1 else ''
        fig.suptitle(f"{r['category']} {kind}: {r['path']}{size_note}  [shared colour scale 0..{vmax:.3f}, display only]",
                     fontsize=10)
        fig.tight_layout()
        name = f"{r['category']}_{kind}_{Path(r['path']).stem}.png"
        fig.savefig(out / 'analysis' / 'panels' / name, dpi=100)
        plt.close(fig)
        listing.append({'panel': name, 'kind': kind, **r})
    save_json(out / 'analysis' / 'panels' / 'panels.json', listing)


if __name__ == '__main__':
    main()
