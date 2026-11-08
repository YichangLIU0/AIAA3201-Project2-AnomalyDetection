"""Build a TEMPORARY reference/calibration manifest so Part 2 can be developed
before member A delivers the shared one.

Rule implemented (from Project2_双人协作计划_v2, section 2.1/2.2):
  * calibration = first 20 normal training images, sorted by file name
  * pool        = remaining normal training images, sorted by file name
  * references  = official AnomalyDINO selection applied to the pool:
                  sorted(pool)[seed*k:(seed+1)*k]  for k in 1/2/4/8, seed in 0/1/2
  * test        = every official test image (good + bad)

The file is marked "provisional": true. run_dino_official.py writes results from a
provisional manifest into a separate *_PROVISIONAL output folder, so these numbers
can never be mixed with formal results. When A's manifest arrives, point
--manifest-dir at it instead; nothing else changes.

Usage:
  python methods/part2/make_temp_manifest.py --data-root data/VisA_pytorch/1cls
"""
import argparse
import json
from pathlib import Path

CATEGORIES = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'pipe_fryum']
SHOTS = [1, 2, 4, 8]
SEEDS = [0, 1, 2]
N_CAL = 20
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}


def images(folder):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def build(data_root, category):
    root = data_root / category
    train = images(root / 'train' / 'good')
    if len(train) < N_CAL + max(SEEDS + [0]) * max(SHOTS) + max(SHOTS):
        raise SystemExit(f'{category}: not enough training images ({len(train)})')
    cal, pool = train[:N_CAL], train[N_CAL:]
    refs = {str(s): {str(k): pool[s * k:(s + 1) * k] for k in SHOTS} for s in SEEDS}
    test = [p for sub in sorted((root / 'test').iterdir()) if sub.is_dir() for p in images(sub)]
    rel = lambda p: p.relative_to(data_root).as_posix()
    manifest = {
        'category': category,
        'provisional': True,
        'created_by': 'member B, make_temp_manifest.py (temporary, replace with member A manifest)',
        'rule': 'cal=sorted(train)[:20]; pool=sorted(train)[20:]; ref[seed][k]=pool[seed*k:(seed+1)*k]',
        'paths_relative_to': 'data root (e.g. data/VisA_pytorch/1cls)',
        'calibration': [rel(p) for p in cal],
        'references': {s: {k: [rel(p) for p in v] for k, v in d.items()} for s, d in refs.items()},
        'test': [rel(p) for p in test],
    }
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='data/VisA_pytorch/1cls')
    ap.add_argument('--out-dir', default='methods/part2/manifests_provisional')
    ap.add_argument('--categories', nargs='+', default=CATEGORIES)
    args = ap.parse_args()
    data_root, out = Path(args.data_root), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for c in args.categories:
        m = build(data_root, c)
        (out / f'{c}.json').write_text(json.dumps(m, indent=1))
        n_bad = sum('/bad/' in p for p in m['test'])
        print(f'{c}: calibration {len(m["calibration"])}, test {len(m["test"])} ({n_bad} bad)')
    print('Wrote provisional manifests to', out)


if __name__ == '__main__':
    main()
