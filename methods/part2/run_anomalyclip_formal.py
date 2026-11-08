"""Part 2.6 (member B): FORMAL official AnomalyCLIP baseline on the shared protocol, zero-shot.

Protocol: shared_protocol section 8 (shots=0, experiment_mode=zero_shot, no references, seed 0 is a
runtime index only). The 20 calibration images only set the displayed OK/NG threshold; nothing is fitted
on VisA and no prompt or setting is chosen with test labels. run_anomalyclip_official.py stays as the
verified PROVISIONAL version; model loading comes from it unchanged.

Checkpoint: third_party/AnomalyCLIP/checkpoints/9_12_4_multiscale_visa/epoch_15.pth. Despite the folder
name it is trained on MVTec AD: upstream train.sh trains this folder with --dataset mvtec and test.sh
tests it with --dataset visa.

Upstream inference (@ 3911738, as test.py --dataset visa): ViT-L/14@336px, image_size 518, features_list
[24], DPAM layer 20, text_probs[:,0,1] as image score, (sim[...,1] + 1 - sim[...,0]) / 2 as map, gaussian
sigma 4 on the 518x518 map. Protocol step: that map is resized with cv2 INTER_LINEAR to the original image
size, because evaluation uses native masks (test.py evaluates squashed 518x518 masks instead).

Usage:
  python methods/part2/run_anomalyclip_formal.py --categories candle
  python methods/part2/run_anomalyclip_formal.py            # all six (skips finished ones)
"""
import argparse
import platform
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared_protocol.data import load_manifest, calibration_paths  # noqa: E402
from shared_protocol.protocol import CATEGORIES, DATA_ROOT, inside  # noqa: E402
from shared_protocol.run_detector import run_detector  # noqa: E402

METHOD = 'anomalyclip'
CONFIG_ID = 'released_visa_checkpoint_v1'
UPSTREAM_SETUP_SEED = 111   # test.py default; inference draws no random numbers after loading


def native_map(map_518, height, width):
    """Resize the upstream square map to the original image size. No normalisation."""
    return cv2.resize(np.asarray(map_518, dtype=np.float32), (int(width), int(height)),
                      interpolation=cv2.INTER_LINEAR)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--categories', nargs='+', default=list(CATEGORIES), choices=CATEGORIES)
    ap.add_argument('--out-dir', default=str(ROOT / 'outputs' / 'anomalyclip'))
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--min-free-gb', type=float, default=20.0, help='stop before a run that would leave less free disk')
    ap.add_argument('--required-only', action='store_true', help='skip optional AP/AUPRO')
    args = ap.parse_args()

    import torch
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    import run_anomalyclip_official as official

    torch.manual_seed(UPSTREAM_SETUP_SEED); torch.cuda.manual_seed_all(UPSTREAM_SETUP_SEED)
    np.random.seed(UPSTREAM_SETUP_SEED); random.seed(UPSTREAM_SETUP_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    lib, model, text, preprocess = official.load_clip()
    size, features, dpam, sigma = official.IMAGE_SIZE, official.FEATURES, official.DPAM, official.SIGMA

    def predict_one(path):
        img = Image.open(path).convert('RGB')       # as upstream dataset.py
        x = preprocess(img).unsqueeze(0).cuda()
        with torch.no_grad():
            img_f, patches = model.encode_image(x, features, DPAM_layer=dpam)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            prob = ((img_f @ text.permute(0, 2, 1)) / 0.07).softmax(-1)[:, 0, 1]
            maps = []
            for pf in patches:
                pf = pf / pf.norm(dim=-1, keepdim=True)
                sim, _ = lib.compute_similarity(pf, text[0])
                sm = lib.get_similarity_map(sim[:, 1:, :], size)
                maps.append((sm[..., 1] + 1 - sm[..., 0]) / 2.0)
            amap = torch.stack(maps).sum(0)[0].cpu().numpy()
        amap = gaussian_filter(amap, sigma=sigma).astype(np.float32)
        return float(prob.item()), native_map(amap, img.size[1], img.size[0])

    static_config = {
        'model': 'ViT-L/14@336px', 'input_resolution': size,
        'input_preprocessing': 'upstream utils.get_transform: bicubic squash to 518x518 + center crop + CLIP mean/std',
        'design': official.DESIGN, 'features_list': features, 'dpam_layer': dpam, 'sigma': sigma,
        'image_score': 'upstream text_probs[:,0,1] (raw)',
        'anomaly_map': 'upstream (sim[...,1]+1-sim[...,0])/2 at 518x518 + gaussian sigma 4, then cv2 INTER_LINEAR '
                       'to native size; float32; no min-max',
        'image_reader': 'PIL convert RGB (as upstream dataset.py)',
        'checkpoint': str(official.PROMPT_CKPT.relative_to(ROOT)).replace('\\', '/'),
        'checkpoint_sha256': official.sha256(official.PROMPT_CKPT),
        'checkpoint_source_training_data': 'MVTec AD, trained by the authors (upstream train.sh trains this folder '
                                           'with --dataset mvtec); nothing fitted on VisA',
        'backbone_sha256': official.BACKBONE_SHA,
        'target_reference_images': 0,
        'calibration_usage': '20 normal images for the OK/NG threshold only',
        'upstream_repo': 'third_party/AnomalyCLIP', 'upstream_commit': official.rdo.git_rev(official.CLIP_REPO),
        'upstream_setup_seed': UPSTREAM_SETUP_SEED, 'precision': 'fp32', 'batch_size': 1,
        'cudnn_deterministic': True, 'cudnn_benchmark': False, 'warmup_forward_passes': args.warmup,
        'timing_scope': 'per image predict_one: read + preprocess + CLIP forward + score + map + gaussian + resize '
                        'to native, with cuda synchronize; excludes model loading, map file writing, metric evaluation',
        'run_by': 'B', 'runner': 'methods/part2/run_anomalyclip_formal.py',
        'host': platform.node(), 'gpu': torch.cuda.get_device_name(0),
        'torch': torch.__version__, 'python': platform.python_version(),
    }

    def resources():
        return {'device': f'cuda:{torch.cuda.current_device()} {torch.cuda.get_device_name(0)}',
                'peak_gpu_allocated_mb': torch.cuda.max_memory_allocated() / 2**20,
                'peak_gpu_reserved_mb': torch.cuda.max_memory_reserved() / 2**20,
                'peak_scope': 'calibration + test predict_one calls, after warm-up; model stays resident and is included'}

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    for category in args.categories:
        output = out_root / category / 'zero_shot_seed0'
        if (output / 'metrics.json').exists():
            print(f'skip {output} (metrics.json exists)', flush=True)
            continue
        if (output / 'run.json').exists() or (output / 'maps').exists():
            raise SystemExit(f'{output} has a partial/unevaluated run; inspect and remove it manually')
        manifest = load_manifest(category)
        need = sum(128 + s['height'] * s['width'] * 4 for s in manifest['test']) + (1 << 20)
        free = shutil.disk_usage(out_root).free
        if free - need < args.min_free_gb * 1e9:
            raise SystemExit(f'DISK_GUARD_STOP: free {free/1e9:.2f} GB, run needs {need/1e9:.2f} GB, '
                             f'floor {args.min_free_gb} GB; not starting {category}')
        warm = inside(DATA_ROOT, calibration_paths(manifest)[0])   # normal calibration image, no labels involved
        for _ in range(args.warmup):
            predict_one(warm)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        result = run_detector(predict_one, output=output, category=category, shots=0, seed=0,
                              method=METHOD, config_id=CONFIG_ID, run_config=static_config,
                              synchronize=torch.cuda.synchronize, resource_reader=resources,
                              optional_metrics=not args.required_only, experiment_mode='zero_shot')
        print(f'{category}: I-AUROC {result["image_auroc"]:.4f} P-AUROC {result["pixel_auroc"]:.4f} '
              f'thr {result["image_threshold"]:.5f} FPR {result["image_fpr"]:.3f} FNR {result["image_fnr"]:.3f} '
              f't/img {result["resources"]["inference_seconds_mean"]*1000:.0f} ms '
              f'peak {result["resources"]["peak_gpu_allocated_mb"]:.0f} MB', flush=True)
    print('ANOMALYCLIP_FORMAL_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
