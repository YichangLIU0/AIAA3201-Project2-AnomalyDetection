"""Validate manifests against the pinned official CSV and frozen handoff hashes."""
import csv
import json
from pathlib import Path, PurePosixPath

from .protocol import (PACKAGE, CATEGORIES, PROTOCOL_ID, SHOTS, SEEDS,
                       OFFICIAL_SPLIT_SHA256, sha256, official_references, save_json)


def official_members(category):
    source = PACKAGE / 'upstream/spot_diff_1cls.csv'
    if sha256(source) != OFFICIAL_SPLIT_SHA256:
        raise ValueError('Pinned official split CSV has changed')
    train, test = [], {}
    with source.open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['object'] != category:
                continue
            abnormal = row['label'] != 'normal'
            path = f'{category}/{row["split"]}/{"bad" if abnormal else "good"}/{PurePosixPath(row["image"]).name}'
            if row['split'] == 'train':
                if abnormal:
                    raise ValueError('Official training split must be normal only')
                train.append(path)
            else:
                mask = f'{category}/ground_truth/bad/{PurePosixPath(row["mask"]).name}' if abnormal else None
                test[path] = (int(abnormal), mask)
    return sorted(train), test


def validate_manifest(manifest, manifest_root=None):
    category = manifest.get('category')
    if category not in CATEGORIES or manifest.get('protocol_id') != PROTOCOL_ID:
        raise ValueError('Wrong manifest category or protocol version')
    train, test = official_members(category)
    if manifest.get('official_split_sha256') != OFFICIAL_SPLIT_SHA256:
        raise ValueError('Wrong official split identity')
    if manifest.get('calibration') != train[:20] or manifest.get('normal_training_pool') != train[20:]:
        raise ValueError('Calibration/training pool differs from official sorted-20 protocol')
    expected_refs = {str(k): {str(s): official_references(train[20:], k, s) for s in SEEDS} for k in SHOTS}
    if manifest.get('references') != expected_refs:
        raise ValueError('Reference lists differ from official selection or include calibration images')
    samples = manifest.get('test', [])
    if [row['path'] for row in samples] != sorted(test):
        raise ValueError('Test manifest must contain ALL official test images in saved order')
    for sample in samples:
        if (sample['label'], sample['mask']) != test[sample['path']]:
            raise ValueError(f'Wrong test label/mask: {sample["path"]}')
    inventory = manifest.get('inventory', [])
    expected_inventory = set(train) | set(test)
    if len(inventory) != len(expected_inventory) or {r['path'] for r in inventory} != expected_inventory:
        raise ValueError('Inventory differs from complete official training/test membership')
    lookup = {row['path']: row for row in inventory}
    for sample in samples:
        if lookup[sample['path']] != sample:
            raise ValueError('Test record differs from its frozen inventory record')
    for row in inventory:
        if any(not isinstance(row.get(key), int) or row[key] <= 0 for key in ('width', 'height')):
            raise ValueError('Missing/invalid native image dimensions')
        digest = row.get('sha256', '')
        if len(digest) != 64 or any(ch not in '0123456789abcdef' for ch in digest):
            raise ValueError('Invalid image content hash')
        if row['path'] in train and (row['label'] != 0 or row['mask'] is not None):
            raise ValueError('Training image is not normal')
        if row['label'] and (not row.get('mask_sha256') or row.get('anomaly_pixels', 0) <= 0):
            raise ValueError('Missing anomaly annotation metadata')
    if manifest_root is not None:
        root = Path(manifest_root)
        for group in ('calibration', 'normal_training_pool'):
            if (root / group / f'{category}.txt').read_text(encoding='utf-8').splitlines() != manifest[group]:
                raise ValueError(f'TXT/JSON mismatch in {group}: {category}')
        for k in SHOTS:
            for s in SEEDS:
                path = root / 'references' / f'{category}_{k}shot_seed{s}.txt'
                if path.read_text(encoding='utf-8').splitlines() != expected_refs[str(k)][str(s)]:
                    raise ValueError(f'TXT/JSON reference mismatch: {path.name}')
    return True


def verify_manifest_lock(category, manifest_root):
    root = Path(manifest_root)
    lock = json.loads((root / 'manifest_lock.json').read_text(encoding='utf-8'))
    if lock.get('protocol_id') != PROTOCOL_ID or lock.get('official_split_sha256') != OFFICIAL_SPLIT_SHA256:
        raise ValueError('Wrong frozen manifest-lock version')
    if lock.get('protocol_sha256') != sha256(PACKAGE / 'protocol.json'):
        raise ValueError('Protocol settings differ from frozen manifest lock')
    if lock.get('manifests', {}).get(category) != sha256(root / f'{category}.json'):
        raise ValueError(f'Frozen manifest checksum changed: {category}')
    return True


def freeze_manifest_lock(manifest_root):
    """Called only after successful generation/audit, never by a normal reader."""
    root = Path(manifest_root)
    hashes = {}
    for category in CATEGORIES:
        path = root / f'{category}.json'
        validate_manifest(json.loads(path.read_text(encoding='utf-8')), root)
        hashes[category] = sha256(path)
    lock = {'protocol_id': PROTOCOL_ID, 'official_split_sha256': OFFICIAL_SPLIT_SHA256,
            'protocol_sha256': sha256(PACKAGE / 'protocol.json'), 'manifests': hashes}
    target = root / 'manifest_lock.json'
    if target.exists() and json.loads(target.read_text(encoding='utf-8')) != lock:
        raise ValueError('Frozen lock would change; use an explicitly versioned migration')
    save_json(target, lock)
    return lock
