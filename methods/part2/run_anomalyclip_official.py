"""Part 2.6 (member B): official AnomalyCLIP checkpoint on VisA, same manifest/metrics as the DINO runner.

Zero-shot w.r.t. VisA: checkpoint third_party/AnomalyCLIP/checkpoints/9_12_4_multiscale_visa/epoch_15.pth
(prompt learner trained by the authors on MVTec AD = auxiliary data; README: "We test all datasets by
training once on MVTec AD"). Nothing is fitted on VisA; no prompt is chosen with test labels.

Taken from upstream (third_party/AnomalyCLIP @ 3911738), mirroring test.py / test.sh for --dataset visa:
  AnomalyCLIP_lib.load('ViT-L/14@336px', design_details={12, 9, 4}), AnomalyCLIP_PromptLearner,
  DAPM_replace(DPAM_layer=20), features_list=[24], image_size=518, utils.get_transform (bicubic squash to
  518x518 + center crop), text_probs[:, 0, 1] as image score, (sim[...,1] + 1 - sim[...,0]) / 2 summed over
  layers as map, gaussian_filter sigma=4.

Differences from running test.py (documented):
  1. Images/calibration from the shared manifest (same files as the DINO baseline); the 20 calibration
     images only set the threshold.
  5b. fp32 everywhere, batch size 1, seed 111 + cudnn deterministic as test.py; maps saved as float32 518x518.
  2. Threshold: same PROVISIONAL 95th percentile as run_dino_official.py (replace with A's implementation).
  3. Metrics with coursework/metrics.py (A's) at the DINO metric shape (short side 448, original aspect):
     the 518x518 map is resized back to the original aspect with cv2.INTER_LINEAR. test.py evaluates the
     squashed 518x518 map against a squashed GT instead.
  4. CLIP backbone loaded from checkpoints/ViT-L-14-336px.pt (sha256 checked) instead of downloading.
  5. Timing covers imread + preprocess + forward + map/score (same scope as the DINO runner, without kNN).

  python methods/part2/run_anomalyclip_official.py --categories candle
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_dino_official as rdo  # noqa: E402  (manifest, GT, metric shape, threshold helpers)

ROOT = rdo.ROOT
CLIP_REPO = ROOT / 'third_party' / 'AnomalyCLIP'
BACKBONE = ROOT / 'checkpoints' / 'ViT-L-14-336px.pt'
BACKBONE_SHA = '3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02'
PROMPT_CKPT = CLIP_REPO / 'checkpoints' / '9_12_4_multiscale_visa' / 'epoch_15.pth'
IMAGE_SIZE, FEATURES, DPAM, SIGMA = 518, [24], 20, 4
DESIGN = {'Prompt_length': 12, 'learnabel_text_embedding_depth': 9, 'learnabel_text_embedding_length': 4}
PROTOCOL = 'official_anomalyclip_518_mvtec_prompts_cal20'
SEED = 111  # official test.py default


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_clip():
    import torch
    sys.path.insert(0, str(CLIP_REPO))
    import AnomalyCLIP_lib
    from prompt_ensemble import AnomalyCLIP_PromptLearner
    from utils import get_transform
    assert sha256(BACKBONE) == BACKBONE_SHA, 'CLIP backbone checksum mismatch'
    model, _ = AnomalyCLIP_lib.load('ViT-L/14@336px', device='cpu', design_details=DESIGN,
                                    download_root=str(BACKBONE.parent))
    model.eval()
    learner = AnomalyCLIP_PromptLearner(model.to('cpu'), DESIGN)
    learner.load_state_dict(torch.load(PROMPT_CKPT, map_location='cpu', weights_only=True)['prompt_learner'])
    learner.cuda().eval(); model.cuda().eval()
    model.visual.DAPM_replace(DPAM_layer=DPAM)
    dtypes = {p.dtype for p in list(model.parameters()) + list(learner.parameters())}
    assert dtypes == {torch.float32}, f'expected fp32 weights, got {dtypes}'
    with torch.no_grad():
        prompts, tokens, compound = learner(cls_id=None)
        text = model.encode_text_learn(prompts, tokens, compound).float()
        text = torch.stack(torch.chunk(text, dim=0, chunks=2), dim=1)
        text = text / text.norm(dim=-1, keepdim=True)
    preprocess, _ = get_transform(SimpleNamespace(image_size=IMAGE_SIZE))
    return AnomalyCLIP_lib, model, text, preprocess


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--categories', nargs='+', default=rdo.CATEGORIES)
    ap.add_argument('--data-root', default=str(ROOT / 'data' / 'VisA_pytorch' / '1cls'))
    ap.add_argument('--manifest-dir', default=str(HERE / 'manifests_provisional'))
    ap.add_argument('--out-dir', default=str(ROOT / 'outputs' / 'part2_anomalyclip'))
    ap.add_argument('--metric-short-side', type=int, default=448)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    data_root = Path(args.data_root)
    probe = json.loads((Path(args.manifest_dir) / f'{args.categories[0]}.json').read_text(encoding='utf-8'))
    out_root = Path(args.out_dir) / (PROTOCOL + ('_PROVISIONAL' if probe.get('provisional') else ''))
    if probe.get('provisional'):
        print('!! Provisional manifest: results go to', out_root, '- NOT formal results', flush=True)

    # as official test.py: setup_seed(111) -> seeds + cudnn.deterministic=True, cudnn.benchmark=False
    import random
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); np.random.seed(SEED); random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    lib, model, text, preprocess = load_clip()
    sys.path.insert(0, str(ROOT / 'coursework'))
    from metrics import evaluate

    def infer(rel):
        t0 = time.perf_counter()
        img = Image.open(data_root / rel).convert('RGB')          # as upstream dataset.py
        x = preprocess(img).unsqueeze(0).cuda()
        with torch.no_grad():
            img_f, patches = model.encode_image(x, FEATURES, DPAM_layer=DPAM)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            prob = ((img_f @ text.permute(0, 2, 1)) / 0.07).softmax(-1)[:, 0, 1]
            maps = []
            for pf in patches:
                pf = pf / pf.norm(dim=-1, keepdim=True)
                sim, _ = lib.compute_similarity(pf, text[0])
                sm = lib.get_similarity_map(sim[:, 1:, :], IMAGE_SIZE)
                maps.append((sm[..., 1] + 1 - sm[..., 0]) / 2.0)
            amap = torch.stack(maps).sum(0)[0].cpu().numpy()
        from scipy.ndimage import gaussian_filter
        amap = gaussian_filter(amap, sigma=SIGMA).astype(np.float32)
        torch.cuda.synchronize()
        return float(prob.item()), amap, img.size[::-1], time.perf_counter() - t0

    for c in args.categories:
        folder = out_root / c / 'zero_shot'
        if (folder / 'metrics.json').exists() and not args.overwrite:
            print('skip', c); continue
        m = rdo.load_manifest(Path(args.manifest_dir), c, data_root)
        for _ in range(args.warmup):
            infer(m['calibration'][0])
        torch.cuda.reset_peak_memory_stats()
        cal = [infer(r) for r in m['calibration']]
        test = []
        for i, r in enumerate(m['test']):
            test.append(infer(r))
            if i % 50 == 0:
                print(f'  {c} {i}/{len(m["test"])}', flush=True)
        peak = torch.cuda.max_memory_allocated() / 2**20
        peak_reserved = torch.cuda.max_memory_reserved() / 2**20
        shape = rdo.metric_shape(test[0][2], args.metric_short_side)
        maps = np.stack([cv2.resize(t[1], (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR) for t in test])
        masks = np.stack([rdo.gt_mask(data_root, r, shape) for r in m['test']])
        cal_scores, scores = np.array([t[0] for t in cal]), np.array([t[0] for t in test])
        thr = rdo.provisional_threshold(cal_scores)
        res = evaluate(maps, masks, scores, thr)
        res.update(method='anomalyclip_official', protocol=PROTOCOL, category=c, shots=0,
                   provisional_manifest=bool(m.get('provisional')), manifest_rule=m.get('rule'),
                   calibration_paths=m['calibration'], calibration_fpr=float((cal_scores > thr).mean()),
                   threshold_rule='PROVISIONAL np.percentile(cal_scores,95); replace with shared implementation',
                   backbone='ViT-L/14@336px', backbone_sha256=BACKBONE_SHA,
                   prompt_checkpoint=str(PROMPT_CKPT.relative_to(ROOT)), prompt_checkpoint_sha256=sha256(PROMPT_CKPT),
                   auxiliary_training_data='MVTec AD (authors), no VisA fitting',
                   design=DESIGN, image_size=IMAGE_SIZE, features_list=FEATURES, dpam_layer=DPAM, sigma=SIGMA,
                   image_score='text_probs[:,0,1]', map='518x518 summed layer map, sigma 4 -> INTER_LINEAR to metric shape',
                   metric_shape=list(shape), mean_inference_seconds=float(np.mean([t[3] for t in test])),
                   timing_scope='per test image: imread + preprocess + CLIP forward + map + score',
                   peak_gpu_mb=peak, peak_gpu_allocated_mb=peak, peak_gpu_reserved_mb=peak_reserved,
                   memory_scope='torch.cuda.max_memory_{allocated,reserved} over calibration+test pass, after warm-up; '
                                'excludes CUDA context',
                   precision='fp32 (weights asserted float32; upstream rounds deep text-prompt tokens via .half() '
                             'in AnomalyCLIP.py:246, as in test.py)', batch_size=1, seed=SEED,
                   cudnn_deterministic=True, cudnn_benchmark=False,
                   tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32), tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
                   torch_version=torch.__version__, gpu=torch.cuda.get_device_name(0), host=platform.node(),
                   upstream_commit=rdo.git_rev(CLIP_REPO), run_by='member B')
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / 'predictions.npz', scores=scores, paths=np.array(m['test']),
                            maps_518=np.stack([t[1] for t in test]).astype(np.float32))
        np.savez_compressed(folder / 'calibration.npz', scores=cal_scores, paths=np.array(m['calibration']))
        (folder / 'metrics.json').write_text(json.dumps(res, indent=1))
        print(f'{c}: I-AUROC {res["image_auroc"]:.4f} P-AUROC {res["pixel_auroc"]:.4f} AUPRO {res["aupro_03"]:.4f} '
              f'FNR {res["image_fnr"]:.3f} t/img {res["mean_inference_seconds"]*1000:.0f} ms peak alloc {peak:.0f} MB reserved {peak_reserved:.0f} MB', flush=True)
    print('PART2_ANOMALYCLIP_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
