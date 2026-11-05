"""Portable data access. Normal masks are generated as zeros, never omitted."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .protocol import DATA_ROOT, MANIFEST_ROOT, CATEGORIES, PROTOCOL_ID, inside, sha256
from .validation import validate_manifest, verify_manifest_lock
from .experiments import expected_references


def load_manifest(category, manifest_root=MANIFEST_ROOT):
    if category not in CATEGORIES:
        raise ValueError(f'Category is outside this protocol: {category}')
    manifest = json.loads((Path(manifest_root) / f'{category}.json').read_text(encoding='utf-8'))
    verify_manifest_lock(category, manifest_root)
    validate_manifest(manifest, manifest_root)
    return manifest


def calibration_paths(manifest, data_root=DATA_ROOT):
    validate_manifest(manifest)
    return [inside(data_root, p) for p in manifest['calibration']]


def training_paths(manifest, data_root=DATA_ROOT):
    """Pool for AE / synthetic-defect training, excluding all 20 calibration images."""
    validate_manifest(manifest)
    return [inside(data_root, p) for p in manifest['normal_training_pool']]


def reference_paths(manifest, shots, seed, data_root=DATA_ROOT):
    validate_manifest(manifest)
    return [inside(data_root, p) for p in expected_references(manifest, shots, seed)]


def load_rgb(path, size=None):
    """Read RGB uint8. size is (height, width); None preserves native aspect ratio.

    DINO should receive the original image and use its official preprocessing.
    Any explicit square resize used by a method must be recorded in its config.
    """
    with Image.open(path) as original:
        image = original.convert('RGB')
        if size is not None:
            image = image.resize((size[1], size[0]), Image.Resampling.BICUBIC)
        return np.array(image, dtype=np.uint8)


def load_mask(sample, data_root=DATA_ROOT, size=None):
    """Return bool GT, using nearest-neighbor resize. Normal GT is all zeros."""
    if size is None:
        size = (sample['height'], sample['width'])
    if sample['label'] == 0:
        if sample['mask'] is not None:
            raise ValueError('Normal sample unexpectedly has an anomaly mask')
        return np.zeros(size, dtype=bool)
    if sample['mask'] is None:
        raise ValueError('Anomalous sample is missing its mask')
    with Image.open(inside(data_root, sample['mask'])) as original:
        image = original.resize((size[1], size[0]), Image.Resampling.NEAREST)
        return np.array(image) > 0


def iter_test(manifest, data_root=DATA_ROOT, size=None):
    """Yield (sample record, RGB, GT) in exactly the saved manifest order."""
    validate_manifest(manifest)
    for sample in manifest['test']:
        yield sample, load_rgb(inside(data_root, sample['path']), size), load_mask(sample, data_root, size)


def verify_inventory(manifest, data_root=DATA_ROOT):
    """Verify another machine's dataset against the frozen content hashes."""
    validate_manifest(manifest)
    expected = {sample['path'] for sample in manifest['inventory']}
    expected.update(sample['mask'] for sample in manifest['inventory'] if sample['mask'])
    category_root = inside(data_root, manifest['category'])
    actual = {p.relative_to(Path(data_root).resolve()).as_posix() for p in category_root.rglob('*') if p.is_file()}
    if actual != expected:
        raise ValueError('Prepared category has extra or missing data files')
    for sample in manifest['inventory']:
        if sha256(inside(data_root, sample['path'])) != sample['sha256']:
            raise ValueError(f'Image checksum mismatch: {sample["path"]}')
        if sample['mask'] and sha256(inside(data_root, sample['mask'])) != sample['mask_sha256']:
            raise ValueError(f'Mask checksum mismatch: {sample["mask"]}')
    return True
