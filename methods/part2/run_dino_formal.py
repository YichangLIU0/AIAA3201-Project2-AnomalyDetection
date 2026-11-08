"""Part 2 (member B): FORMAL official AnomalyDINO baseline on the shared protocol.

Protocol: shared_protocol (visa6_sorted20_official_refs_native_v2, member A).
References / calibration / test come ONLY from shared_protocol.data; results are
saved with shared_protocol.run_detector (PredictionWriter native maps + shared
evaluation). run_dino_official.py stays as the verified PROVISIONAL version.

Upstream logic is reused unchanged from run_dino_official.py (which was
parity-checked against third_party/AnomalyDINO @ b9d1c26):
  dinov2_vits14, smaller edge 448, preprocess "agnostic" (VisA: test masking
  ON, references unmasked, 8 rotations), exact 1-NN on L2-normalised features
  (torch matmul; distance = ||a-b||^2/2 = 1-cos), image score mean_top1p,
  native map = upstream dists2map(patch_grid, original_rgb.shape) (bilinear to
  original size + gaussian sigma 4). No downsampling, no min-max.

Per run: warm up -> build bank from the run's references (timed separately,
no cross-run feature cache) -> reset CUDA peak counters -> run_detector, whose
predict_one extracts the test/calibration image features itself, so the
per-image time is honest.

Usage:
  python methods/part2/run_dino_formal.py --categories candle --shots 1 --seeds 0
  python methods/part2/run_dino_formal.py            # all 72 runs (skips finished ones)
"""
import argparse
import hashlib
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared_protocol.data import load_manifest, reference_paths, calibration_paths  # noqa: E402
from shared_protocol.protocol import CATEGORIES, SHOTS, SEEDS, ANOMALYDINO_REVISION  # noqa: E402
from shared_protocol.run_detector import run_detector  # noqa: E402
from run_dino_official import (UPSTREAM, MODEL_NAME, RESOLUTION, import_upstream, load_model,  # noqa: E402
                               git_rev, read_rgb, extract, NearestNeighbour, patch_distances)

METHOD = 'official_dino'
CONFIG_ID = 'vits14_448_agnostic_v1'
WEIGHTS = ROOT / 'checkpoints' / 'dinov2_vits14_pretrain.pth'


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def run_one(model, tools, category, shots, seed, args, static_config):
    import torch
    augment_image, dists2map, mean_top1p = tools
    output = Path(args.out_dir) / category / f'{shots}shot_seed{seed}'
    if (output / 'metrics.json').exists():
        print(f'skip {output} (metrics.json exists)', flush=True)
        return None
    if (output / 'run.json').exists() or (output / 'maps').exists():
        raise SystemExit(f'{output} has a partial/unevaluated run; inspect and remove it manually')

    manifest = load_manifest(category)
    need = sum(128 + s['height'] * s['width'] * 4 for s in manifest['test']) + (1 << 20)   # float32 maps + json
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out_root).free   # the disk holding the outputs, not the path's root (differs on Linux mounts)
    if free - need < args.min_free_gb * 1e9:
        raise SystemExit(f'DISK_GUARD_STOP: free {free/1e9:.2f} GB, run needs {need/1e9:.2f} GB, '
                         f'floor {args.min_free_gb} GB; not starting {category} {shots}-shot seed{seed}')
    refs = reference_paths(manifest, shots, seed)
    warm = read_rgb(calibration_paths(manifest)[0])   # normal calibration image, no labels involved

    # 1. warm-up
    for _ in range(args.warmup):
        model.extract_features(model.prepare_image(warm)[0])
    torch.cuda.synchronize()

    # 2. bank from this run's references only (refs unmasked, 8 rotations), timed separately
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    bank = np.concatenate([extract(model, a, masking=False)[0]
                           for p in refs for a in augment_image(read_rgb(p))])
    nn = NearestNeighbour(bank, backend=args.nn)
    torch.cuda.synchronize()
    bank_seconds = time.perf_counter() - t0
    bank_peak_alloc = torch.cuda.max_memory_allocated() / 2**20
    bank_peak_reserved = torch.cuda.max_memory_reserved() / 2**20

    # 3. reset peak counters before the timed calibration/test pass
    torch.cuda.reset_peak_memory_stats()

    def predict_one(path):
        img = read_rgb(path)
        feats, mask, grid = extract(model, img, masking=True)   # agnostic VisA: test masking ON
        d = patch_distances(nn, feats, mask, grid)
        score = float(mean_top1p(d.flatten()))
        return score, dists2map(d, img.shape)                    # native map, float32

    run_config = {**static_config,
                  'bank_tokens': int(len(bank)), 'bank_mb': float(bank.nbytes / 2**20),
                  'bank_build_seconds': bank_seconds,
                  'bank_build_scope': 'read + 8 rotations + DINOv2 forward for every reference + 1-NN index upload; '
                                      'after warm-up, no cached features'}

    def resources():
        return {'device': f'cuda:{torch.cuda.current_device()} {torch.cuda.get_device_name(0)}',
                'peak_gpu_allocated_mb': torch.cuda.max_memory_allocated() / 2**20,
                'peak_gpu_reserved_mb': torch.cuda.max_memory_reserved() / 2**20,
                'peak_scope': 'calibration + test predict_one calls; bank stays resident and is included',
                'bank_build_seconds': bank_seconds,
                'bank_build_peak_gpu_allocated_mb': bank_peak_alloc,
                'bank_build_peak_gpu_reserved_mb': bank_peak_reserved}

    print(f'[{category} {shots}-shot seed{seed}] bank {len(bank)} tokens in {bank_seconds:.2f}s', flush=True)
    result = run_detector(predict_one, output=output, category=category, shots=shots, seed=seed,
                          method=METHOD, config_id=CONFIG_ID, run_config=run_config,
                          synchronize=torch.cuda.synchronize, resource_reader=resources,
                          optional_metrics=not args.required_only)
    print(f'  I-AUROC {result["image_auroc"]:.4f} P-AUROC {result["pixel_auroc"]:.4f} '
          f'thr {result["image_threshold"]:.5f} FPR {result["image_fpr"]:.3f} FNR {result["image_fnr"]:.3f} '
          f't/img {result["resources"]["inference_seconds_mean"]*1000:.0f} ms', flush=True)
    del nn, bank
    torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--categories', nargs='+', default=list(CATEGORIES), choices=CATEGORIES)
    ap.add_argument('--shots', nargs='+', type=int, default=list(SHOTS), choices=SHOTS)
    ap.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS), choices=SEEDS)
    ap.add_argument('--out-dir', default=str(ROOT / 'outputs' / 'dino'))
    ap.add_argument('--nn', choices=['torch', 'faiss'], default='torch')
    ap.add_argument('--warmup', type=int, default=25)
    ap.add_argument('--min-free-gb', type=float, default=20.0, help='stop before a run that would leave less free disk')
    ap.add_argument('--required-only', action='store_true', help='skip optional AP/AUPRO')
    args = ap.parse_args()

    import torch
    upstream_commit = git_rev(UPSTREAM)
    if upstream_commit != ANOMALYDINO_REVISION:
        raise SystemExit(f'AnomalyDINO checkout {upstream_commit} != protocol {ANOMALYDINO_REVISION}')
    static_config = {
        'model': MODEL_NAME, 'input_resolution': RESOLUTION, 'preprocess': 'agnostic',
        'test_masking': True, 'reference_masking': False, 'reference_rotations': 8,
        'knn': '1-NN, L2-normalised features, distance ||a-b||^2/2 (upstream L2_normalized)',
        'nn_backend': args.nn, 'image_score': 'upstream mean_top1p over patch distances (raw)',
        'anomaly_map': 'upstream dists2map(patch_grid, original_rgb.shape): cv2 bilinear to native size + '
                       'gaussian sigma 4; float32; no downsampling, no min-max',
        'image_reader': 'cv2.imread BGR->RGB (as upstream)',
        'upstream_repo': 'third_party/AnomalyDINO', 'upstream_commit': upstream_commit,
        'dinov2_repo_commit': git_rev(ROOT / 'third_party' / 'dinov2'),
        'weights': str(WEIGHTS.relative_to(ROOT)).replace('\\', '/'), 'weights_sha256': file_sha256(WEIGHTS),
        'model_loading': 'torch.hub local dinov2 repo, pretrained=False, strict state_dict load',
        'warmup_forward_passes': args.warmup,
        'timing_scope': 'per image predict_one: imread + upstream prepare_image + DINOv2 forward + PCA mask + '
                        '1-NN + mean_top1p + dists2map to native size, with cuda synchronize; excludes bank build '
                        '(bank_build_seconds), map file writing and metric evaluation',
        'run_by': 'B', 'runner': 'methods/part2/run_dino_formal.py',
        'host': platform.node(), 'gpu': torch.cuda.get_device_name(0),
        'torch': torch.__version__, 'python': platform.python_version(),
    }
    tools = import_upstream()
    model = load_model('cuda')
    for c in args.categories:
        for k in args.shots:
            for s in args.seeds:
                run_one(model, tools, c, k, s, args, static_config)
    print('DINO_FORMAL_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
