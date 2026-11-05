from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
DATA_ROOT = ROOT / 'data/VisA_pytorch/1cls'
MANIFEST_ROOT = PACKAGE / 'manifests'
CATEGORIES = ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'pipe_fryum')
SHOTS = (1, 2, 4, 8)
SEEDS = (0, 1, 2)
PROTOCOL_ID = 'visa6_sorted20_official_refs_native_v2'
OFFICIAL_SPLIT_SHA256 = 'a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995'
SPOT_DIFF_REVISION = '2a692ab575001cbde74d402d897a7286086c6199'
ANOMALYDINO_REVISION = 'b9d1c2648e3a5247437d4d953d907a8f3d994457'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Path escapes data root: {relative}')
    return path


def official_references(pool, shots, seed):
    """Upstream src/detection.py: sorted(pool)[seed*K:(seed+1)*K].

    Our ONLY selection change is to remove the fixed 20 calibration images first.
    Seeds are upstream deterministic block indices, not random-number seeds.
    Different K values need not have nested references for seeds 1 and 2.
    """
    if shots not in SHOTS or seed not in SEEDS:
        raise ValueError('Unsupported shot/seed')
    selected = sorted(pool)[seed * shots:(seed + 1) * shots]
    if len(selected) != shots:
        raise ValueError('Not enough training images')
    return selected
