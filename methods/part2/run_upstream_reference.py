"""Run UNMODIFIED upstream run_anomalydino.py on a subset of VisA objects (reference check).

Upstream (third_party/AnomalyDINO @ b9d1c26) has no per-category option: --dataset VisA always
iterates over all 12 objects. This wrapper runs upstream's own __main__ via runpy and changes
only two things, both by patching module attributes at runtime (no upstream file is edited):

  1. Object list restricted to --objects (default: candle) in get_dataset_info (used by the
     runner and by create_sample_plots) and get_objects_from_dataset (used by eval_finished_run).
  2. DINOv2Wrapper.load_model loads the pinned local dinov2 repo (7764ea0) and the verified local
     checkpoint (checkpoints/dinov2_vits14_pretrain.pth, sha256 b938bf1b...) instead of
     torch.hub.load('facebookresearch/dinov2'), which would download the unpinned main branch.
     The transform is copied verbatim from upstream.

Everything else (reference selection sorted(train)[seed*k:(seed+1)*k] WITHOUT excluding the 20
calibration images, PCA masking, 8 rotations, FAISS IndexFlatL2, mean_top1p, upstream eval) is
upstream code. Results are written under --out-dir (upstream writes results_VisA/... relative
to the working directory, so we chdir there).

Usage:
  python methods/part2/run_upstream_reference.py -- --dataset VisA --resolution 448 \
      --preprocess agnostic --shots 1 --faiss_on_cpu --data_root <abs path to VisA_pytorch/1cls>
"""
import argparse
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / 'third_party' / 'AnomalyDINO'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--objects', nargs='+', default=['candle'])
    ap.add_argument('--out-dir', default=str(ROOT / 'outputs' / 'part2_upstream_reference'))
    ap.add_argument('upstream_args', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    up_args = [a for a in args.upstream_args if a != '--']

    sys.path.insert(0, str(UPSTREAM))
    import torch
    from torchvision import transforms
    import src.utils as U
    import src.post_eval as PE
    import src.visualize as V
    import src.backbones as B

    keep = list(args.objects)
    orig_info = U.get_dataset_info

    def info_subset(*a, **k):
        objects, anomalies, masking, rotation = orig_info(*a, **k)
        assert all(o in objects for o in keep), keep
        return (keep, {o: anomalies[o] for o in keep},
                {o: masking[o] for o in keep}, {o: rotation[o] for o in keep})

    orig_objs = PE.get_objects_from_dataset
    PE.get_objects_from_dataset = lambda d: [o for o in orig_objs(d) if o in keep]
    U.get_dataset_info = info_subset
    V.get_dataset_info = info_subset

    def local_load_model(self):
        m = torch.hub.load(str(ROOT / 'third_party' / 'dinov2'), self.model_name, source='local', pretrained=False)
        sd = torch.load(ROOT / 'checkpoints' / f'{self.model_name}_pretrain.pth', map_location='cpu', weights_only=True)
        m.load_state_dict(sd, strict=True)
        m.eval()
        self.transform = transforms.Compose([   # verbatim from upstream DINOv2Wrapper.load_model
            transforms.Resize(size=self.smaller_edge_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        return m.to(self.device)

    B.DINOv2Wrapper.load_model = local_load_model

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)
    print('Upstream reference run; objects =', keep, '; cwd =', out, '; args =', up_args, flush=True)
    sys.argv = [str(UPSTREAM / 'run_anomalydino.py')] + up_args
    runpy.run_path(str(UPSTREAM / 'run_anomalydino.py'), run_name='__main__')


if __name__ == '__main__':
    main()
