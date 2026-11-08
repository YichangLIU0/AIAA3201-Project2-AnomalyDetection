"""Fixed DINO interface for member A's directions C and F (member B, Part 2 handover).

Same model and logic as run_dino_formal.py: dinov2_vits14, smaller edge 448, agnostic, references from the
shared manifest, 8 rotations, exact 1-NN, mean_top1p, native-size dists2map. Backbone frozen.

    from methods.part2.dino_interface import DinoDetector
    detector = DinoDetector('candle', shots=1, seed=0)
    score, anomaly_map = detector.predict(path)          # raw score, native-size float32 map
    features, mask, grid = detector.features(path)       # patch features, foreground mask, patch grid

Self-check against saved results (needs a GPU):
    python methods/part2/dino_interface.py --run outputs/dino/candle/1shot_seed0 --images 5
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared_protocol.data import load_manifest, reference_paths  # noqa: E402
from shared_protocol.protocol import DATA_ROOT, inside  # noqa: E402
from run_dino_official import (import_upstream, load_model, read_rgb, extract,  # noqa: E402
                               NearestNeighbour, patch_distances)


class DinoDetector:
    """One category/shots/seed: builds the normal bank once, then scores images one at a time."""

    def __init__(self, category, shots, seed, device='cuda', model=None):
        self.augment_image, self.dists2map, self.mean_top1p = import_upstream()
        self.model = model if model is not None else load_model(device)
        self.category, self.shots, self.seed = category, shots, seed
        references = reference_paths(load_manifest(category), shots, seed)
        self.bank = np.concatenate([extract(self.model, a, masking=False)[0]
                                    for p in references for a in self.augment_image(read_rgb(p))])
        self.nn = NearestNeighbour(self.bank, backend='torch')

    def features(self, path, masking=True):
        return extract(self.model, read_rgb(path), masking=masking)

    def predict(self, path):
        image = read_rgb(path)
        features, mask, grid = extract(self.model, image, masking=True)
        distances = patch_distances(self.nn, features, mask, grid)
        return float(self.mean_top1p(distances.flatten())), self.dists2map(distances, image.shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, help='a finished formal run folder to compare against')
    ap.add_argument('--images', type=int, default=5)
    args = ap.parse_args()
    folder = Path(args.run)
    run = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
    detector = DinoDetector(run['category'], run['shots'], run['seed'])
    worst_score = worst_map = 0.0
    for item in run['test'][:args.images]:
        score, amap = detector.predict(inside(DATA_ROOT, item['path']))
        saved = np.load(folder / item['map'])
        worst_score = max(worst_score, abs(score - item['score']))
        worst_map = max(worst_map, float(np.abs(amap - saved).max()))
    print(f'{args.images} images: max abs score diff {worst_score:.3e}, max abs map diff {worst_map:.3e}', flush=True)
    if worst_score > 1e-5:
        raise SystemExit('INTERFACE_CHECK_FAILED: scores differ from the formal run by more than 1e-5')
    print('INTERFACE_CHECK_PASSED', flush=True)


if __name__ == '__main__':
    main()
