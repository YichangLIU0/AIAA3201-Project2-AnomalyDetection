"""Part 2 (member B): official AnomalyDINO baseline on VisA.

Required configuration (Project2_双人协作计划_v2, section 2.2 / Part 2):
  backbone dinov2_vits14, smaller-edge resolution 448, preprocess "agnostic".
  For VisA, upstream "agnostic" = masking ON for test images, rotation ON
  (8 rotations) for reference images; reference images are NOT masked
  (upstream default --mask_ref_images False).

What is taken verbatim from upstream (third_party/AnomalyDINO @ b9d1c26):
  DINOv2Wrapper.prepare_image / extract_features / compute_background_mask,
  augment_image (rotations), mean_top1p (image score), dists2map (map
  upsampling to the ORIGINAL image size + gaussian sigma=4).

Deliberate, documented differences from running run_anomalydino.py:
  1. References and calibration images come from a manifest file (member A's
     shared one, or a provisional one from make_temp_manifest.py) instead of
     upstream's own slicing of train/good, so the 20 calibration images are
     excluded and every method uses identical references.
  2. Exact 1-NN search is done with a torch matmul on L2-normalised features
     instead of faiss.IndexFlatL2 (avoids the Windows OpenMP/FAISS crash noted
     in followup_20261031). Distances are identical: ||a-b||^2/2 = 1-cos.
     Use --nn faiss to run upstream's FAISS search instead.
  3. Pixel metrics: the upstream full-resolution map is area-downsampled to
     --metric-short-side (default 448) to keep memory bounded. Record this.
  4. The model is loaded from the local dinov2 repo + checkpoint (same
     transform as upstream), because torch.hub download may be unavailable.
  5. Test-image features are extracted once per category and reused across the
     12 shot/seed settings; per-image timing = feature time + kNN time.

Outputs (one folder per category/shot/seed):
  metrics.json      metrics, threshold, config, timing, peak GPU memory
  predictions.npz   patch-distance grids, image scores, test paths
  calibration.npz   same for the 20 calibration images
Threshold: PROVISIONAL 95th percentile of calibration scores. Member A owns the
shared threshold/metric code; calibration scores are saved so it can be
recomputed with A's implementation without rerunning the model.

Usage (one class smoke run, then all):
  python methods/part2/run_dino_official.py --categories candle
  python methods/part2/run_dino_official.py
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / 'third_party' / 'AnomalyDINO'
CATEGORIES = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'pipe_fryum']
SHOTS = [1, 2, 4, 8]
SEEDS = [0, 1, 2]
RESOLUTION = 448
MODEL_NAME = 'dinov2_vits14'
PROTOCOL = 'official_dinov2_vits14_448_agnostic_cal20'


# ---------------------------------------------------------------- upstream
def import_upstream():
    sys.path.insert(0, str(UPSTREAM))
    from src.utils import augment_image, dists2map          # noqa: E402
    from src.post_eval import mean_top1p                    # noqa: E402
    return augment_image, dists2map, mean_top1p


def load_model(device='cuda'):
    import torch
    from torchvision import transforms
    sys.path.insert(0, str(UPSTREAM))
    from src.backbones import DINOv2Wrapper

    class LocalDINOv2(DINOv2Wrapper):
        def load_model(self):
            m = torch.hub.load(str(ROOT / 'third_party' / 'dinov2'), self.model_name,
                               source='local', pretrained=False)
            sd = torch.load(ROOT / 'checkpoints' / 'dinov2_vits14_pretrain.pth',
                            map_location='cpu', weights_only=True)
            m.load_state_dict(sd, strict=True)
            # identical to upstream DINOv2Wrapper.load_model transform
            self.transform = transforms.Compose([
                transforms.Resize(size=self.smaller_edge_size,
                                  interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            return m.eval().to(self.device)

    return LocalDINOv2(MODEL_NAME, device, RESOLUTION)


def git_rev(path):
    try:
        return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return 'unknown'


# ------------------------------------------------------------ core pieces
def read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def l2n(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class NearestNeighbour:
    """Exact 1-NN with upstream's L2_normalized metric: returns ||q-b||^2 / 2."""

    def __init__(self, bank, backend='torch', device='cuda'):
        self.backend = backend
        bank = l2n(bank)
        self.n = len(bank)
        if backend == 'torch':
            import torch
            self.torch = torch
            self.bank = torch.from_numpy(bank).to(device)
            self.device = device
        elif backend == 'faiss':
            import faiss
            self.index = faiss.IndexFlatL2(bank.shape[1])
            self.index.add(bank)
        elif backend == 'numpy':
            self.bank = bank
        else:
            raise ValueError(backend)

    def search(self, q, chunk=1024):
        q = l2n(q)
        if self.backend == 'faiss':
            d, _ = self.index.search(q, 1)
            return np.maximum(d[:, 0], 0) / 2
        out = []
        for i in range(0, len(q), chunk):
            if self.backend == 'torch':
                t = self.torch
                qt = t.from_numpy(q[i:i + chunk]).to(self.device)
                best = (qt @ self.bank.T).max(dim=1).values
                out.append(t.clamp(1 - best, min=0).cpu().numpy())
            else:
                best = (q[i:i + chunk] @ self.bank.T).max(axis=1)
                out.append(np.maximum(1 - best, 0))
        return np.concatenate(out).astype(np.float32)


def extract(model, image, masking):
    """Upstream per-image step: prepare, extract tokens, (optionally) PCA mask."""
    tensor, grid = model.prepare_image(image)
    feats = model.extract_features(tensor)
    if masking:
        mask = model.compute_background_mask(feats, grid, threshold=10, masking_type=True)
    else:
        mask = np.ones(feats.shape[0], dtype=bool)
    return feats.astype(np.float32), np.asarray(mask, dtype=bool).reshape(-1), tuple(grid)


def patch_distances(nn, feats, mask, grid):
    d = np.zeros(mask.shape, dtype=np.float32)
    if mask.any():
        d[mask] = nn.search(feats[mask])
    return d.reshape(grid)


def metric_shape(orig_hw, short_side):
    h, w = orig_hw
    s = short_side / min(h, w)
    return int(round(h * s)), int(round(w * s))


def to_metric_map(dists2map, d_grid, orig_hw, shape):
    full = dists2map(d_grid, orig_hw)                       # upstream: original size + sigma 4
    return cv2.resize(full.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def gt_mask(data_root, rel, shape):
    p = Path(rel)
    if p.parent.name == 'good':
        return np.zeros(shape, dtype=bool)
    m = data_root / p.parts[0] / 'ground_truth' / 'bad' / f'{p.stem}.png'
    m = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(m)
    return cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def pack(arrays):
    """Stack equal-shaped arrays; fall back to an object array (load with allow_pickle=True)."""
    if len({a.shape for a in arrays}) == 1:
        return np.stack(arrays)
    out = np.empty(len(arrays), dtype=object)
    out[:] = list(arrays)
    return out


def provisional_threshold(cal_scores):
    """PROVISIONAL: 95th percentile. Replace with member A's shared implementation."""
    return float(np.percentile(np.asarray(cal_scores, dtype=np.float64), 95))


def load_manifest(manifest_dir, category, data_root):
    m = json.loads((manifest_dir / f'{category}.json').read_text(encoding='utf-8'))
    cal = set(m['calibration'])
    assert len(m['calibration']) == 20, 'expected 20 calibration images'
    for s in map(str, SEEDS):
        for k in SHOTS:
            refs = m['references'][s][str(k)]
            assert len(refs) == k and len(set(refs)) == k, (category, s, k)
            assert cal.isdisjoint(refs), f'{category} seed {s} {k}-shot: reference overlaps calibration'
    assert cal.isdisjoint(m['test'])
    missing = [p for p in m['calibration'] + m['test'] if not (data_root / p).exists()]
    assert not missing, f'missing files, e.g. {missing[:3]}'
    return m


# --------------------------------------------------------------- driver
def run_category(category, model, args, tools, out_root):
    augment_image, dists2map, mean_top1p = tools
    import torch
    data_root = Path(args.data_root)
    m = load_manifest(Path(args.manifest_dir), category, data_root)

    # warm-up as upstream (relevant for timing)
    first = read_rgb(data_root / m['calibration'][0])
    for _ in range(args.warmup):
        model.extract_features(model.prepare_image(first)[0])
    torch.cuda.synchronize()

    def query_pass(paths):
        torch.cuda.reset_peak_memory_stats()
        items = []
        for i, rel in enumerate(paths):
            t0 = time.perf_counter()
            img = read_rgb(data_root / rel)
            f, mk, g = extract(model, img, masking=True)        # agnostic VisA: test masking ON
            torch.cuda.synchronize()
            items.append(dict(f=f, m=mk, g=g, hw=img.shape[:2], t=time.perf_counter() - t0))
            if i % 50 == 0:
                print(f'  features {category} {i}/{len(paths)}', flush=True)
        return items, torch.cuda.max_memory_allocated() / 2**20

    print(f'[{category}] extracting calibration + test features', flush=True)
    cal_items, _ = query_pass(m['calibration'])
    test_items, peak_feat_mb = query_pass(m['test'])

    shape = metric_shape(test_items[0]['hw'], args.metric_short_side)
    masks = np.stack([gt_mask(data_root, r, shape) for r in m['test']])

    ref_cache = {}

    def ref_feats(rel):
        if rel not in ref_cache:
            img = read_rgb(data_root / rel)
            ref_cache[rel] = [extract(model, a, masking=False)[0] for a in augment_image(img)]
        return ref_cache[rel]

    if args.save_features:
        np.savez_compressed(out_root / category / 'test_features_fp16.npz',
                            features=pack([x['f'].astype(np.float16) for x in test_items]),
                            fg_masks=pack([x['m'] for x in test_items]),
                            grids=np.array([x['g'] for x in test_items]), paths=np.array(m['test']))

    sys.path.insert(0, str(ROOT / 'coursework'))
    from metrics import evaluate     # shared metric code (member A owns it)

    for seed in SEEDS:
        for k in SHOTS:
            folder = out_root / category / f'{k}shot_seed{seed}'
            if (folder / 'metrics.json').exists() and not args.overwrite:
                print('  skip', folder.name)
                continue
            refs = m['references'][str(seed)][str(k)]
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            bank = np.concatenate([f for r in refs for f in ref_feats(r)])  # refs unmasked, 8 rotations
            nn = NearestNeighbour(bank, backend=args.nn)
            torch.cuda.synchronize()
            bank_s = time.perf_counter() - t0

            def score_items(items):
                grids, scores, times = [], [], []
                for it in items:
                    t1 = time.perf_counter()
                    d = patch_distances(nn, it['f'], it['m'], it['g'])
                    s = float(mean_top1p(d.flatten()))
                    torch.cuda.synchronize()
                    grids.append(d); scores.append(s); times.append(it['t'] + time.perf_counter() - t1)
                return grids, np.array(scores), np.array(times)

            cal_grids, cal_scores, _ = score_items(cal_items)
            grids, scores, times = score_items(test_items)
            peak_knn_mb = torch.cuda.max_memory_allocated() / 2**20
            thr = provisional_threshold(cal_scores)
            maps = np.stack([to_metric_map(dists2map, g, it['hw'], shape) for g, it in zip(grids, test_items)])
            res = evaluate(maps, masks, scores, thr)
            res.update(
                method='anomalydino_official', protocol=PROTOCOL, category=category, shots=k, seed=seed,
                provisional_manifest=bool(m.get('provisional')), manifest_rule=m.get('rule'),
                reference_paths=refs, calibration_paths=m['calibration'],
                threshold_rule='PROVISIONAL np.percentile(cal_scores,95); replace with shared implementation',
                calibration_fpr=float((cal_scores > thr).mean()),
                model=MODEL_NAME, resolution_smaller_edge=RESOLUTION, preprocess='agnostic',
                test_masking=True, reference_masking=False, rotations=8, knn='1-NN L2_normalized (dist/2)',
                nn_backend=args.nn, image_score='mean_top1p', map='dists2map(original size, sigma=4) -> INTER_AREA',
                metric_shape=list(shape), bank_tokens=int(len(bank)), bank_mb=float(bank.nbytes / 2**20),
                bank_build_seconds=bank_s,
                mean_inference_seconds=float(times.mean()),
                timing_scope='per test image: imread + preprocess + DINOv2 forward + PCA mask + 1-NN + score; '
                             'excludes memory-bank build and map upsampling',
                peak_gpu_mb_feature_phase=peak_feat_mb, peak_gpu_mb_knn_phase=peak_knn_mb,
                gpu=torch.cuda.get_device_name(0), host=platform.node(),
                upstream_commit=git_rev(UPSTREAM), dinov2_commit=git_rev(ROOT / 'third_party' / 'dinov2'),
                run_by='member B',
            )
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / 'predictions.npz', scores=scores, paths=np.array(m['test']),
                                patch_dists=pack(grids), times=times)
            np.savez_compressed(folder / 'calibration.npz', scores=cal_scores, paths=np.array(m['calibration']),
                                patch_dists=pack(cal_grids))
            (folder / 'metrics.json').write_text(json.dumps(res, indent=1))
            print(f'  {category} {k}-shot seed{seed}: I-AUROC {res["image_auroc"]:.4f} '
                  f'P-AUROC {res["pixel_auroc"]:.4f} FNR {res["image_fnr"]:.3f} '
                  f't/img {res["mean_inference_seconds"]*1000:.0f} ms', flush=True)
            del nn, maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--categories', nargs='+', default=CATEGORIES)
    ap.add_argument('--data-root', default=str(ROOT / 'data' / 'VisA_pytorch' / '1cls'))
    ap.add_argument('--manifest-dir', default=str(ROOT / 'methods' / 'part2' / 'manifests_provisional'))
    ap.add_argument('--out-dir', default=str(ROOT / 'outputs' / 'part2_dino448'))
    ap.add_argument('--nn', choices=['torch', 'faiss'], default='torch')
    ap.add_argument('--metric-short-side', type=int, default=448)
    ap.add_argument('--warmup', type=int, default=25)
    ap.add_argument('--save-features', action='store_true', help='save fp16 test features for member A (C/F)')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    probe = json.loads((Path(args.manifest_dir) / f'{args.categories[0]}.json').read_text(encoding='utf-8'))
    tag = PROTOCOL + ('_PROVISIONAL' if probe.get('provisional') else '')
    out_root = Path(args.out_dir) / tag
    for c in args.categories:
        (out_root / c).mkdir(parents=True, exist_ok=True)
    if probe.get('provisional'):
        print('!! Provisional manifest: results go to', out_root, '- NOT formal results', flush=True)

    tools = import_upstream()
    model = load_model('cuda')
    for c in args.categories:
        run_category(c, model, args, tools, out_root)
    print('PART2_DINO448_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
