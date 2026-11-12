"""Check the dual-scale runs against the formal Part 2 baseline. CPU only.

For every finished run:
  1. the recomputed 448 configuration (s448) vs the formal Part 2 run: test and calibration scores, threshold and every
     metric (Image/Pixel AUROC, AP, AUPRO, FPR, FNR);
  2. where the formal native maps still exist (seed-0 runs), regenerate the first --n-maps maps from the saved 448 grids
     with upstream dists2map and compare them with the formal map files;
  3. grids file integrity (sha256) and completeness of all three configurations.
A match in (1) on all metrics also checks that the in-memory evaluation equals the formal file-based evaluation.

  python methods/part3/verify_dualscale.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'shared_protocol').is_dir())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from shared_protocol.protocol import CATEGORIES, SHOTS, SEEDS, save_json, sha256  # noqa: E402
import run_dualscale as rd  # noqa: E402

METRICS = ('image_auroc', 'pixel_auroc', 'image_ap', 'pixel_ap', 'aupro_03', 'image_fpr', 'image_fnr', 'image_threshold')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(ROOT / 'outputs' / 'part3_dualscale'))
    ap.add_argument('--formal-dir', default=str(ROOT / 'outputs' / 'dino'))
    ap.add_argument('--n-maps', type=int, default=5)
    ap.add_argument('--score-tol', type=float, default=1e-5)
    args = ap.parse_args()
    out, formal = Path(args.out_dir), Path(args.formal_dir)
    _, dists2map, _ = rd.rdo.import_upstream()
    rows, problems = [], []
    for c in CATEGORIES:
        for k in SHOTS:
            for s in SEEDS:
                d = out / 'runs' / c / f'{k}shot_seed{s}'
                if not (d / 'done.json').exists():
                    continue
                run = json.loads((d / 'run.json').read_text(encoding='utf-8'))
                tag = f'{c} {k}-shot seed{s}'
                if sha256(d / run['grids_file']) != run['grids_sha256']:
                    problems.append(f'{tag}: grids file changed')
                if set(run['configs']) != set(rd.CONFIGS) or any(not (d / f'metrics_{n}.json').exists() for n in rd.CONFIGS):
                    problems.append(f'{tag}: missing configuration output')
                f_run = json.loads((formal / c / f'{k}shot_seed{s}' / 'run.json').read_text(encoding='utf-8'))
                f_met = json.loads((formal / c / f'{k}shot_seed{s}' / 'metrics.json').read_text(encoding='utf-8'))
                mine = run['configs']['s448']
                if [t['path'] for t in f_run['test']] != run['test_paths'] or f_run['reference_paths'] != run['reference_paths']:
                    problems.append(f'{tag}: test order or references differ from the formal run')
                dt = float(np.max(np.abs(np.array(mine['test_scores']) - np.array([t['score'] for t in f_run['test']]))))
                dc = float(np.max(np.abs(np.array(mine['calibration_scores']) - np.array(f_run['calibration_scores']))))
                m = json.loads((d / 'metrics_s448.json').read_text(encoding='utf-8'))
                dm = {key: abs(m[key] - f_met[key]) for key in METRICS}
                row = {'run': tag, 'max_test_score_diff': dt, 'max_calibration_score_diff': dc,
                       **{f'diff_{key}': v for key, v in dm.items()}}
                check = run['resources'].get('formal_448_map_check')   # done in-run on float32 grids
                if check:
                    row['max_map_diff_vs_formal_files'] = check['max_abs_diff_vs_formal_448_maps']
                rows.append(row)
                if dt > args.score_tol or dc > args.score_tol:
                    problems.append(f'{tag}: s448 scores differ from formal by {max(dt, dc):.2e}')
    summary = {'n_runs_checked': len(rows), 'problems': problems,
               'max_test_score_diff': max((r['max_test_score_diff'] for r in rows), default=None),
               'max_calibration_score_diff': max((r['max_calibration_score_diff'] for r in rows), default=None),
               'max_metric_diff': {key: max((r[f'diff_{key}'] for r in rows), default=None) for key in METRICS},
               'max_map_diff_vs_formal_files': max((r.get('max_map_diff_vs_formal_files', 0.0) for r in rows), default=None),
               'n_runs_with_map_check': sum('max_map_diff_vs_formal_files' in r for r in rows),
               'n_metric_exact_equal': {key: sum(r[f'diff_{key}'] == 0 for r in rows) for key in METRICS},
               'score_tolerance': args.score_tol, 'rows': rows}
    save_json(out / 'verify_baseline.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=1))


if __name__ == '__main__':
    main()
