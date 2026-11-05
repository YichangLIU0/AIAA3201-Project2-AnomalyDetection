"""Check the complete handoff on this machine and generate a data-only preview."""
from pathlib import Path
import argparse
import platform
import sys

import numpy as np
import scipy
import sklearn
import PIL
from PIL import Image, ImageDraw, ImageOps

from .data import load_manifest, load_rgb, load_mask, verify_inventory
from .protocol import CATEGORIES, DATA_ROOT, MANIFEST_ROOT, PACKAGE, PROTOCOL_ID, inside, save_json
from .release import verify as verify_release


def check(data_root=DATA_ROOT, manifest_root=MANIFEST_ROOT, verify_hashes=True, output_dir=None):
    release = verify_release()
    rows, checks = [], []
    width, height = 240, 170
    preview = Image.new('RGB', (4 * width, 40 + len(CATEGORIES) * (height + 30)), 'white')
    draw = ImageDraw.Draw(preview)
    for column, title in enumerate(['Category / normal test', 'Anomalous test', 'Ground truth', 'GT overlay (red)']):
        draw.text((column * width + 8, 10), title, fill='black')
    for row_index, category in enumerate(CATEGORIES):
        manifest = load_manifest(category, manifest_root)
        if verify_hashes:
            verify_inventory(manifest, data_root)
        normal = next(item for item in manifest['test'] if item['label'] == 0)
        abnormal = next(item for item in manifest['test'] if item['label'] == 1)
        if load_mask(normal, data_root).any():
            raise ValueError('Normal mask is not zero')
        image = load_rgb(inside(data_root, abnormal['path']))
        mask = load_mask(abnormal, data_root)
        if image.shape[:2] != mask.shape or not mask.any():
            raise ValueError('Abnormal image/mask mismatch')
        overlay = image.copy()
        overlay[mask] = (.4 * overlay[mask] + .6 * np.array([255, 0, 0])).astype(np.uint8)
        panels = [Image.open(inside(data_root, normal['path'])).convert('RGB'), Image.fromarray(image),
                  Image.fromarray(mask.astype(np.uint8) * 255).convert('RGB'), Image.fromarray(overlay)]
        y = 40 + row_index * (height + 30)
        for column, panel in enumerate(panels):
            thumbnail = ImageOps.contain(panel, (width - 12, height), Image.Resampling.LANCZOS)
            preview.paste(thumbnail, (column * width + (width - thumbnail.width) // 2, y))
        draw.text((8, y + height + 4), category, fill='black')
        draw.text((width + 8, y + height + 4), Path(abnormal['path']).name, fill='black')
        rows.append({'category': category, 'normal': normal['path'], 'anomaly': abnormal['path'], 'mask': abnormal['mask']})
        checks.append({'category': category, 'inventory_hashes_verified': verify_hashes,
                       'frozen_manifest_and_txt_consistency_verified': True,
                       'official_split_and_sorted20_exclusion_verified': True,
                       'n_calibration': len(manifest['calibration']),
                       'n_reference_lists': sum(len(values) for values in manifest['references'].values()),
                       'n_test': len(manifest['test']), 'example_normal_mask_is_zero': True})
        print('READY', category, checks[-1], flush=True)
    reports = Path(output_dir or PACKAGE / 'reports')
    reports.mkdir(parents=True, exist_ok=True)
    preview.save(reports / 'data_preview.png')
    save_json(reports / 'data_preview_samples.json', rows)
    save_json(reports / 'readiness.json', {'protocol_id': PROTOCOL_ID, 'status': 'passed' if verify_hashes else 'partial',
              'release_id': release['release_id'], 'source_and_docs_frozen_hashes_verified': True,
              'checks': checks, 'validation_scope': 'Data/loading checks only; no model predictions.',
              'runtime': {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__,
                          'scipy': scipy.__version__, 'scikit_learn': sklearn.__version__, 'Pillow': PIL.__version__}})
    print('PREVIEW', reports / 'data_preview.png', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--manifest-root', type=Path, default=MANIFEST_ROOT)
    parser.add_argument('--skip-hashes', action='store_true', help='Quick check only; default verifies all image/mask hashes')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    check(args.data_root, args.manifest_root, not args.skip_hashes, args.output_dir)
