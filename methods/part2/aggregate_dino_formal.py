"""Post-batch for the formal DINO grid (member B). CPU only, never reruns the model.

1. For each of the 72 runs: check run.json/metrics.json exist and re-evaluate with
   python -m shared_protocol.evaluate_predictions into metrics_reeval.json, then
   require it to equal metrics.json (ignoring nothing).
2. Aggregate the 72 metrics.json with A's shared_protocol.metrics.aggregate_runs.
3. Compare image AUROC (and raw image scores) with the PROVISIONAL runs.

Writes outputs/dino/summary/{aggregate.json, per_run.csv, check.json}.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared_protocol.metrics import aggregate_runs  # noqa: E402
from shared_protocol.protocol import CATEGORIES, SHOTS, SEEDS, save_json  # noqa: E402

PROV = ROOT / 'outputs/part2_dino448/official_dinov2_vits14_448_agnostic_cal20_PROVISIONAL'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(ROOT / 'outputs/dino'))
    ap.add_argument('--skip-reeval', action='store_true')
    args = ap.parse_args()
    out = Path(args.out_dir)
    missing, failed, reeval_mismatch, rows, table = [], [], [], [], []
    for c in CATEGORIES:
        for k in SHOTS:
            for s in SEEDS:
                d = out / c / f'{k}shot_seed{s}'
                tag = f'{c} {k}-shot seed{s}'
                if not (d / 'run.json').exists() or not (d / 'metrics.json').exists():
                    missing.append(tag)
                    continue
                met = json.loads((d / 'metrics.json').read_text(encoding='utf-8'))
                if met.get('status') != 'success':
                    failed.append(tag)
                    continue
                if not args.skip_reeval:
                    target = d / 'metrics_reeval.json'
                    r = subprocess.run([sys.executable, '-m', 'shared_protocol.evaluate_predictions',
                                        '--run', str(d), '--output', str(target)],
                                       cwd=ROOT, capture_output=True, text=True)
                    if r.returncode != 0:
                        failed.append(f'{tag} (re-eval error: {r.stderr.strip().splitlines()[-1:]})')
                        continue
                    if json.loads(target.read_text(encoding='utf-8')) != met:
                        reeval_mismatch.append(tag)
                rows.append(met)
                run = json.loads((d / 'run.json').read_text(encoding='utf-8'))
                p = PROV / c / f'{k}shot_seed{s}'
                prov_auroc = prov_dscore = None
                if (p / 'metrics.json').exists():
                    pm = json.loads((p / 'metrics.json').read_text(encoding='utf-8'))
                    prov_auroc = pm['image_auroc']
                    pp = np.load(p / 'predictions.npz', allow_pickle=True)
                    pmap = dict(zip(map(str, pp['paths']), pp['scores'].astype(float)))
                    prov_dscore = float(max(abs(r['score'] - pmap[r['path']]) for r in run['test']))
                table.append({'category': c, 'shots': k, 'seed': s,
                              'image_auroc': met['image_auroc'], 'prov_image_auroc': prov_auroc,
                              'auroc_abs_diff': None if prov_auroc is None else abs(met['image_auroc'] - prov_auroc),
                              'max_abs_score_diff_vs_prov': prov_dscore,
                              'pixel_auroc': met['pixel_auroc'], 'image_ap': met.get('image_ap'),
                              'pixel_ap': met.get('pixel_ap'), 'aupro_03': met.get('aupro_03'),
                              'image_threshold': met['image_threshold'], 'image_fpr': met['image_fpr'],
                              'image_fnr': met['image_fnr'],
                              'inference_ms': 1000 * met['resources']['inference_seconds_mean'],
                              'bank_build_s': met['resources']['bank_build_seconds'],
                              'peak_alloc_mb': met['resources']['peak_gpu_allocated_mb'],
                              'peak_reserved_mb': met['resources']['peak_gpu_reserved_mb']})
    summary = out / 'summary'
    summary.mkdir(parents=True, exist_ok=True)
    with (summary / 'per_run.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    agg, agg_error = None, None
    try:
        agg = aggregate_runs(rows, CATEGORIES)
        save_json(summary / 'aggregate.json', agg)
    except ValueError as e:
        agg_error = str(e)
    diffs = [t['auroc_abs_diff'] for t in table if t['auroc_abs_diff'] is not None]
    sdiffs = [t['max_abs_score_diff_vs_prov'] for t in table if t['max_abs_score_diff_vs_prov'] is not None]
    check = {'n_complete': len(rows), 'missing': missing, 'failed': failed, 'reeval_mismatch': reeval_mismatch,
             'aggregate_error': agg_error, 'n_compared_with_provisional': len(diffs),
             'max_image_auroc_abs_diff': max(diffs) if diffs else None,
             'n_image_auroc_exact_equal': sum(d == 0 for d in diffs),
             'max_abs_image_score_diff': max(sdiffs) if sdiffs else None}
    save_json(summary / 'check.json', check)
    print(json.dumps(check, indent=1))
    if agg:
        for a in agg:
            print(f"{a['shots']}-shot  I-AUROC {a['image_auroc']['mean']:.4f}±{a['image_auroc']['std']:.4f}  "
                  f"P-AUROC {a['pixel_auroc']['mean']:.4f}±{a['pixel_auroc']['std']:.4f}  "
                  f"AUPRO {a['aupro_03']['mean']:.4f}  FPR {a['image_fpr']['mean']:.3f}  FNR {a['image_fnr']['mean']:.3f}")


if __name__ == '__main__':
    main()
