"""Part 3, direction A (member B): dual-scale AnomalyDINO, frozen backbone.

Three configurations are produced from ONE pass per run, so they share references, calibration images,
test images and evaluation code exactly:
  s448  official Part 2 baseline recomputed (dinov2_vits14, smaller edge 448, agnostic): used to check that
        this code reproduces the formal Part 2 run, and as the per-image baseline for analysis
  s672  resolution control: same pipeline at smaller edge 672
  fuse  proposed: 0.5 * (448 patch distances, bilinearly resampled to the 672 patch grid) + 0.5 * (672 distances)

Fixed design (decided before any test metric was computed; see FROZEN_DESIGN in the code):
  - references: shared manifest, unmasked, upstream 8 rotations, a separate memory bank per scale
  - query foreground mask: upstream PCA mask computed at 448 only, projected to the 672 patch grid with
    nearest-neighbour resizing (so a new scale-dependent mask is not an extra variable)
  - distance: exact 1-NN on L2-normalised features, ||a-b||^2/2 (as Part 2)
  - fusion on raw patch distances, then ONE upstream dists2map (bilinear to native size + gaussian sigma 4)
  - image score: upstream mean_top1p on the grid of each configuration (448 grid for s448, 672 grid otherwise)
  - threshold: shared 95th percentile of the 20 calibration scores of each configuration
  - no per-image normalisation, no learned weights, nothing chosen with test labels

Disk: native maps are NOT written. Raw patch grids are saved (small); metrics are computed in memory with the
shared protocol's own image_statistics and pixel_statistics_stream, regenerating each native map from its grid
with upstream dists2map. verify_dualscale.py checks this path against the formal file-based Part 2 evaluation.

Modes:
  python methods/part3/run_dualscale.py --mode grid                 # all 72 runs (skips finished ones)
  python methods/part3/run_dualscale.py --mode grid --categories candle --shots 1 --seeds 0
  python methods/part3/run_dualscale.py --mode heldout --categories candle   # normal-only implementation check
  python methods/part3/run_dualscale.py --mode timing --categories candle pipe_fryum --n-images 40
"""
import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'shared_protocol').is_dir())
sys.path.insert(0, str(ROOT))
PART2 = ROOT / 'part2_official' if (ROOT / 'part2_official' / 'run_dino_official.py').exists() else ROOT / 'methods' / 'part2'
sys.path.insert(0, str(PART2))

from shared_protocol.data import load_manifest, reference_paths, calibration_paths, load_mask  # noqa: E402
from shared_protocol.protocol import CATEGORIES, SHOTS, SEEDS, DATA_ROOT, PROTOCOL_ID, ANOMALYDINO_REVISION, inside, save_json, sha256  # noqa: E402
from shared_protocol.metrics import image_statistics, pixel_statistics_stream, normal_threshold  # noqa: E402
from shared_protocol.experiments import experiment_spec  # noqa: E402
from shared_protocol.predictions import source_version  # noqa: E402
import run_dino_official as rdo  # noqa: E402

SCALES = (448, 672)
FUSE_WEIGHTS = (0.5, 0.5)
CONFIGS = {
    's448': {'method': 'dino_scale448_rerun', 'config_id': 'vits14_448_agnostic_v1_rerun'},
    's672': {'method': 'dino_scale672', 'config_id': 'vits14_672_agnostic_mask448_v1'},
    'fuse': {'method': 'dino_dualscale', 'config_id': 'vits14_448x672_mean05_mask448_v1'},
}
FROZEN_DESIGN = {
    'backbone': 'dinov2_vits14 (frozen), weights checkpoints/dinov2_vits14_pretrain.pth',
    'scales_smaller_edge': list(SCALES), 'fuse_weights_448_672': list(FUSE_WEIGHTS),
    'preprocess': 'agnostic: query PCA mask (upstream, threshold 10) computed at 448; references unmasked; 8 rotations',
    'mask_672': '448 query mask projected to the 672 patch grid with cv2 INTER_NEAREST',
    'distance': '1-NN, L2-normalised features, ||a-b||^2/2, exact (torch)',
    'fusion': '448 distance grid resampled to the 672 grid with cv2 INTER_LINEAR, weighted mean with the 672 grid',
    'map': 'upstream dists2map(grid, native shape): INTER_LINEAR to native size + gaussian sigma 4; float32',
    'image_score': 'upstream mean_top1p over the configuration grid (zeros on masked background included, as upstream)',
    'threshold': 'shared normal_threshold: linear 95th percentile of the 20 calibration scores, per configuration',
    'test_labels': 'used only by evaluation and post-hoc analysis',
}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


RUNNER_SHA = file_sha(__file__)


def project_mask(mask, grid_from, grid_to):
    """Nearest-neighbour projection of a flat patch mask between patch grids (both span the whole image)."""
    m = np.asarray(mask, dtype=np.uint8).reshape(grid_from)
    return cv2.resize(m, (grid_to[1], grid_to[0]), interpolation=cv2.INTER_NEAREST).astype(bool).reshape(-1)


def fuse_grids(d448, d672, weights=FUSE_WEIGHTS):
    up = cv2.resize(np.asarray(d448, dtype=np.float32), (d672.shape[1], d672.shape[0]), interpolation=cv2.INTER_LINEAR)
    return (weights[0] * up + weights[1] * np.asarray(d672, dtype=np.float32)).astype(np.float32)


def config_grid(name, d448, d672):
    if name == 's448':
        return d448
    if name == 's672':
        return d672
    return fuse_grids(d448, d672)


def load_models():
    rdo.RESOLUTION = 448
    model448 = rdo.load_model('cuda')
    rdo.RESOLUTION = 672
    model672 = rdo.load_model('cuda')
    rdo.RESOLUTION = 448
    assert model448.smaller_edge_size == 448 and model672.smaller_edge_size == 672
    return model448, model672


class Branches:
    """Memory banks for one category/shots/seed at both scales, and the per-image raw distance grids."""

    def __init__(self, models, tools, references):
        import torch
        self.torch = torch
        self.m448, self.m672 = models
        self.augment, self.dists2map, self.mean_top1p = tools
        t0 = time.perf_counter()
        b448, b672 = [], []
        for p in references:
            for a in self.augment(rdo.read_rgb(p)):
                b448.append(rdo.extract(self.m448, a, masking=False)[0])
                b672.append(rdo.extract(self.m672, a, masking=False)[0])
        self.bank448, self.bank672 = np.concatenate(b448), np.concatenate(b672)
        self.nn448 = rdo.NearestNeighbour(self.bank448, backend='torch')
        self.nn672 = rdo.NearestNeighbour(self.bank672, backend='torch')
        torch.cuda.synchronize()
        self.bank_seconds = time.perf_counter() - t0

    def grids(self, path, scales=('s448', 's672')):
        image = rdo.read_rgb(path)
        f4, m4, g4 = rdo.extract(self.m448, image, masking=True)
        d4 = rdo.patch_distances(self.nn448, f4, m4, g4) if 's448' in scales else None
        d6 = None
        if 's672' in scales:
            tensor, g6 = self.m672.prepare_image(image)
            f6 = self.m672.extract_features(tensor).astype(np.float32)
            d6 = rdo.patch_distances(self.nn672, f6, project_mask(m4, g4, g6), tuple(g6))
        return image.shape[:2], d4, d6

    def score(self, grid):
        return float(self.mean_top1p(grid.flatten()))


def evaluate_config(grids, shapes, cal_scores, test_scores, manifest, dists2map, optional=True):
    """Same functions as shared_protocol.evaluate_predictions, with native maps regenerated in memory."""
    labels = [s['label'] for s in manifest['test']]

    def pairs():
        for grid, sample, shape in zip(grids, manifest['test'], shapes):
            assert tuple(shape) == (sample['height'], sample['width'])
            yield dists2map(grid, shape), load_mask(sample, DATA_ROOT)

    result = image_statistics(test_scores, cal_scores, labels, optional)
    result.update(pixel_statistics_stream(pairs, optional))
    return result


def run_one(models, tools, category, shots, seed, out_root, args, static):
    import torch
    out = out_root / 'runs' / category / f'{shots}shot_seed{seed}'
    if (out / 'done.json').exists():
        done = json.loads((out / 'done.json').read_text(encoding='utf-8'))
        if done.get('runner_sha256') != RUNNER_SHA:
            raise SystemExit(f'{out} was made by a different runner version; inspect it before continuing')
        print(f'skip {out}', flush=True)
        return
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} holds a partial run; inspect and remove it manually')
    free = shutil.disk_usage(out_root).free
    if free < args.min_free_gb * 1e9:
        raise SystemExit(f'DISK_GUARD_STOP: {free/1e9:.2f} GB free < {args.min_free_gb} GB')
    manifest = load_manifest(category)
    refs = reference_paths(manifest, shots, seed)
    warm = calibration_paths(manifest)[0]
    for _ in range(args.warmup):
        models[0].extract_features(models[0].prepare_image(rdo.read_rgb(warm))[0])
        models[1].extract_features(models[1].prepare_image(rdo.read_rgb(warm))[0])
    torch.cuda.synchronize()
    br = Branches(models, tools, refs)
    torch.cuda.reset_peak_memory_stats()
    cal = [br.grids(p) for p in calibration_paths(manifest)]
    test, seconds = [], []
    for sample in manifest['test']:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        test.append(br.grids(inside(DATA_ROOT, sample['path'])))
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - t0)
    resources = {'peak_gpu_allocated_mb': torch.cuda.max_memory_allocated() / 2**20,
                 'peak_gpu_reserved_mb': torch.cuda.max_memory_reserved() / 2**20,
                 'peak_scope': 'calibration + test grid computation, both banks resident',
                 'bank_build_seconds_both_scales': br.bank_seconds,
                 'bank_tokens_448': int(len(br.bank448)), 'bank_tokens_672': int(len(br.bank672)),
                 'bank_mb_448': br.bank448.nbytes / 2**20, 'bank_mb_672': br.bank672.nbytes / 2**20,
                 'grid_seconds_mean_both_scales': float(np.mean(seconds)),
                 'timing_note': 'grid pass only (read, 448 and 672 forward, PCA mask, 1-NN at both scales); '
                                'maps, scores and fusion are excluded here. Comparable per-configuration end-to-end '
                                'timing comes from --mode timing.'}
    # Check against the formal Part 2 map files where they still exist (before anything is rounded for storage).
    map_check = None
    formal_maps = ROOT / 'outputs' / 'dino' / category / f'{shots}shot_seed{seed}' / 'maps'
    if formal_maps.exists():
        worst = 0.0
        for i in range(min(args.map_check, len(test))):
            regen = br.dists2map(test[i][1], tuple(test[i][0]))
            worst = max(worst, float(np.max(np.abs(regen - np.load(formal_maps / f'{i:04d}.npy')))))
        map_check = {'n_maps': min(args.map_check, len(test)), 'max_abs_diff_vs_formal_448_maps': worst}
    resources['formal_448_map_check'] = map_check
    out.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for prefix, items in (('cal', cal), ('test', test)):
        for i, (_, d4, d6) in enumerate(items):
            arrays[f'{prefix}448_{i:04d}'] = d4.astype(np.float16)
            arrays[f'{prefix}672_{i:04d}'] = d6.astype(np.float16)
    np.savez_compressed(out / 'grids.npz', **arrays)
    record = {'protocol_id': PROTOCOL_ID, 'category': category, 'shots': shots, 'seed': seed,
              'reference_paths': [str(Path(p).relative_to(DATA_ROOT)).replace('\\', '/') for p in refs],
              'calibration_paths': manifest['calibration'], 'test_paths': [s['path'] for s in manifest['test']],
              'test_shapes': [list(t[0]) for t in test], 'grids_file': 'grids.npz', 'grids_sha256': sha256(out / 'grids.npz'),
              'grids_dtype': 'float16, for post-hoc analysis and panels only; every score and metric in this run was '
                             'computed from the float32 grids before rounding (the shared disk was full)',
              'runner_sha256': RUNNER_SHA, 'static_config': static, 'resources': resources, 'configs': {}}
    spec = experiment_spec(shots, seed)
    for name, ident in CONFIGS.items():
        cg = [config_grid(name, d4, d6) for _, d4, d6 in cal]
        tg = [config_grid(name, d4, d6) for _, d4, d6 in test]
        cal_scores, test_scores = [br.score(g) for g in cg], [br.score(g) for g in tg]
        record['configs'][name] = {**ident, 'calibration_scores': cal_scores, 'test_scores': test_scores}
        result = evaluate_config(tg, [t[0] for t in test], cal_scores, test_scores, manifest, br.dists2map,
                                 optional=not args.required_only)
        result.update(category=category, shots=shots, seed=seed, protocol_id=PROTOCOL_ID, integration_only=False,
                      pixel_evaluation='native_original_image_and_mask (maps regenerated in memory from saved grids)',
                      aupro_connectivity=8, run_config={**static, 'configuration': name},
                      resources=resources, manifest_sha256=sha256(ROOT / 'shared_protocol' / 'manifests' / f'{category}.json'),
                      evaluator='shared_protocol.metrics image_statistics + pixel_statistics_stream (in memory)',
                      evaluator_source_sha256=source_version(), runner_sha256=RUNNER_SHA, **ident)
        result.update(spec, reference_paths=record['reference_paths'])
        save_json(out / f'metrics_{name}.json', result)
    save_json(out / 'run.json', record)
    save_json(out / 'done.json', {'runner_sha256': RUNNER_SHA, 'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    m = {n: json.loads((out / f'metrics_{n}.json').read_text(encoding='utf-8')) for n in CONFIGS}
    print(f'[{category} {shots}-shot seed{seed}] ' + '  '.join(
        f'{n}: I {m[n]["image_auroc"]:.4f} P {m[n]["pixel_auroc"]:.4f} FPR {m[n]["image_fpr"]:.3f} FNR {m[n]["image_fnr"]:.3f}'
        for n in CONFIGS) + f'  grid {np.mean(seconds)*1000:.0f} ms/img', flush=True)


def heldout(models, tools, category, out_root, n_images):
    """Normal-only implementation check: no test image is read. Held-out normals = training pool images that are
    in no reference list of any shot/seed and not in calibration."""
    manifest = load_manifest(category)
    used = {p for k in SHOTS for s in SEEDS for p in manifest['references'][str(k)][str(s)]} | set(manifest['calibration'])
    pool = [p for p in manifest['normal_training_pool'] if p not in used]
    held = sorted(pool)[-n_images:]
    report = {'category': category, 'n_heldout': len(held), 'heldout_paths': held, 'runs': {}, 'runner_sha256': RUNNER_SHA}
    formal_root = ROOT / 'outputs' / 'dino' / category
    for shots, seed in ((1, 0), (8, 0)):
        br = Branches(models, tools, reference_paths(manifest, shots, seed))
        cal = [br.grids(p) for p in calibration_paths(manifest)]
        hel = [br.grids(inside(DATA_ROOT, p)) for p in held]
        entry = {}
        for name in CONFIGS:
            cs = [br.score(config_grid(name, d4, d6)) for _, d4, d6 in cal]
            hs = [br.score(config_grid(name, d4, d6)) for _, d4, d6 in hel]
            thr = normal_threshold(cs)
            entry[name] = {'threshold': thr, 'heldout_fraction_above_threshold': float(np.mean(np.array(hs) > thr)),
                           'heldout_score_median': float(np.median(hs)), 'heldout_score_max': float(np.max(hs)),
                           'calibration_score_median': float(np.median(cs)),
                           'all_finite': bool(np.isfinite(cs).all() and np.isfinite(hs).all()),
                           'grid_shape_example': list(config_grid(name, *cal[0][1:]).shape)}
        formal = formal_root / f'{shots}shot_seed{seed}' / 'run.json'
        if formal.exists():
            fcal = json.loads(formal.read_text(encoding='utf-8'))['calibration_scores']
            mine = [br.score(d4) for _, d4, _ in cal]
            entry['s448_vs_formal_calibration_max_abs_diff'] = float(np.max(np.abs(np.array(mine) - np.array(fcal))))
        report['runs'][f'{shots}shot_seed{seed}'] = entry
        print(category, shots, seed, json.dumps(entry)[:600], flush=True)
    save_json(out_root / 'heldout' / f'{category}.json', report)


def timing(models, tools, categories, out_root, n_images, warmup):
    """End-to-end per-image cost of each configuration on its own, measured one configuration at a time."""
    import torch
    rows = []
    for category in categories:
        manifest = load_manifest(category)
        paths = [inside(DATA_ROOT, s['path']) for s in manifest['test'][:n_images]]  # first N by manifest order, no labels
        for shots in (1, 8):
            br = Branches(models, tools, reference_paths(manifest, shots, 0))
            for name in CONFIGS:
                scales = {'s448': ('s448',), 's672': ('s672',), 'fuse': ('s448', 's672')}[name]

                def predict(path):
                    shape, d4, d6 = br.grids(path, scales)
                    grid = d4 if name == 's448' else d6 if name == 's672' else fuse_grids(d4, d6)
                    return br.score(grid), br.dists2map(grid, shape)

                for _ in range(warmup):
                    predict(paths[0])
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                ts = []
                for p in paths:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    predict(p)
                    torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
                rows.append({'category': category, 'shots': shots, 'config': name, 'n_images': len(paths),
                             'ms_mean': 1000 * float(np.mean(ts)), 'ms_median': 1000 * float(np.median(ts)),
                             'peak_gpu_allocated_mb': torch.cuda.max_memory_allocated() / 2**20,
                             'peak_gpu_reserved_mb': torch.cuda.max_memory_reserved() / 2**20,
                             'bank_mb': (br.bank448.nbytes if 's448' in scales else 0) / 2**20
                                        + (br.bank672.nbytes if 's672' in scales else 0) / 2**20})
                print(json.dumps(rows[-1]), flush=True)
            del br
            torch.cuda.empty_cache()
    scope = ('per image: read + 448 forward + PCA mask (needed by every configuration, the 672 control uses the 448 mask) '
             '+ 1-NN at the configuration scale(s) + fusion + mean_top1p + dists2map to native size; cuda synchronize; '
             'banks resident (memory includes only the banks of the configuration timed, but the other bank stays '
             'allocated, so allocated memory is an upper bound); excludes bank build and metric evaluation')
    save_json(out_root / 'timing' / 'timing.json', {'rows': rows, 'scope': scope, 'runner_sha256': RUNNER_SHA,
                                                    'gpu': torch.cuda.get_device_name(0), 'host': platform.node()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['grid', 'heldout', 'timing'], default='grid')
    ap.add_argument('--categories', nargs='+', default=list(CATEGORIES), choices=CATEGORIES)
    ap.add_argument('--shots', nargs='+', type=int, default=list(SHOTS), choices=SHOTS)
    ap.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS), choices=SEEDS)
    ap.add_argument('--out-dir', default=str(ROOT / 'outputs' / 'part3_dualscale'))
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--n-images', type=int, default=30)
    ap.add_argument('--min-free-gb', type=float, default=0.1)
    ap.add_argument('--map-check', type=int, default=5, help='formal 448 maps compared per run, where they exist')
    ap.add_argument('--required-only', action='store_true')
    args = ap.parse_args()
    import torch
    if rdo.git_rev(rdo.UPSTREAM) != ANOMALYDINO_REVISION:
        raise SystemExit('AnomalyDINO checkout differs from the protocol revision')
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    weights = ROOT / 'checkpoints' / 'dinov2_vits14_pretrain.pth'
    static = {**FROZEN_DESIGN, 'upstream_commit': rdo.git_rev(rdo.UPSTREAM), 'dinov2_repo_commit': rdo.git_rev(ROOT / 'third_party' / 'dinov2'),
              'weights_sha256': sha256(weights), 'runner': 'methods/part3/run_dualscale.py', 'runner_sha256': RUNNER_SHA,
              'part2_building_blocks': str(PART2.relative_to(ROOT)).replace('\\', '/') + '/run_dino_official.py',
              'part2_building_blocks_sha256': file_sha(PART2 / 'run_dino_official.py'),
              'host': platform.node(), 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'python': platform.python_version(), 'run_by': 'B'}
    models = load_models()
    tools = rdo.import_upstream()
    if args.mode == 'grid':
        for c in args.categories:
            for k in args.shots:
                for s in args.seeds:
                    run_one(models, tools, c, k, s, out_root, args, static)
        print('DUALSCALE_GRID_COMPLETE', flush=True)
    elif args.mode == 'heldout':
        for c in args.categories:
            heldout(models, tools, c, out_root, args.n_images)
        print('HELDOUT_COMPLETE', flush=True)
    else:
        timing(models, tools, args.categories, out_root, args.n_images, args.warmup)
        print('TIMING_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
