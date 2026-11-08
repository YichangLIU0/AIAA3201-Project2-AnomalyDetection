"""Parity check: run_dino_official.py pipeline vs unmodified upstream, on the SAME references.

Upstream picks references as sorted(train/good)[seed*k:(seed+1)*k] WITHOUT excluding the 20
calibration images, so for seed 0 / 1-shot it uses train/good/0000.JPG (a calibration image in our
protocol). This script feeds that same reference through our runner's functions (extract,
NearestNeighbour, patch_distances, mean_top1p) and compares per-image scores and image AUROC with
upstream's measurements CSV. It is a code-parity check only, not a protocol result.

  python methods/part2/compare_with_upstream.py --upstream-dir outputs/part2_upstream_reference/results_VisA/dinov2_vits14_448/1-shot_preprocess=agnostic
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_dino_official as rdo  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--upstream-dir', required=True)
    ap.add_argument('--category', default='candle')
    ap.add_argument('--shots', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--data-root', default=str(rdo.ROOT / 'data' / 'VisA_pytorch' / '1cls'))
    ap.add_argument('--nn', default='torch', choices=['torch', 'faiss'])
    args = ap.parse_args()
    up_dir, data_root = Path(args.upstream_dir), Path(args.data_root)

    up = {}
    with open(up_dir / f'measurements_seed={args.seed}.csv') as f:
        for r in csv.DictReader(f):
            if r['Object'] == args.category:
                up[r['Sample']] = float(r['Anomaly_Score'])
    train = sorted((data_root / args.category / 'train' / 'good').iterdir())
    refs = train[args.seed * args.shots:(args.seed + 1) * args.shots]
    print('upstream references:', [p.name for p in refs])

    augment_image, dists2map, mean_top1p = rdo.import_upstream()
    model = rdo.load_model('cuda')
    bank = np.concatenate([rdo.extract(model, a, masking=False)[0] for p in refs for a in augment_image(rdo.read_rgb(p))])
    nn = rdo.NearestNeighbour(bank, backend=args.nn)
    keys, ours, theirs = [], [], []
    for key in sorted(up):
        img = rdo.read_rgb(data_root / args.category / 'test' / key)
        f, m, g = rdo.extract(model, img, masking=True)
        ours.append(float(mean_top1p(rdo.patch_distances(nn, f, m, g).flatten())))
        theirs.append(up[key]); keys.append(key)
    ours, theirs = np.array(ours), np.array(theirs)
    y = np.array([k.startswith('bad/') for k in keys])
    diff = np.abs(ours - theirs)
    res = dict(category=args.category, shots=args.shots, seed=args.seed, references=[p.name for p in refs],
               n=len(keys), max_abs_score_diff=float(diff.max()), mean_abs_score_diff=float(diff.mean()),
               n_diff_gt_1e3=int((diff > 1e-3).sum()), upstream_csv_precision='5 decimals',
               image_auroc_ours=float(roc_auc_score(y, ours)), image_auroc_upstream=float(roc_auc_score(y, theirs)),
               spearman_like_rank_agreement=float(np.corrcoef(np.argsort(np.argsort(ours)), np.argsort(np.argsort(theirs)))[0, 1]),
               nn_backend=args.nn, worst=[(keys[i], float(ours[i]), float(theirs[i])) for i in np.argsort(-diff)[:5]])
    print(json.dumps(res, indent=1))
    out = up_dir.parents[2] / f'parity_{args.category}_{args.shots}shot_seed{args.seed}.json'
    out.write_text(json.dumps(res, indent=1)); print('saved', out)


if __name__ == '__main__':
    main()
