"""Extract only six categories, apply official 1cls CSV, audit and freeze manifests."""
from collections import Counter
from pathlib import Path, PurePosixPath
import argparse
import csv
import json
import shutil
import tarfile

import numpy as np
from PIL import Image

from .protocol import (ROOT, PACKAGE, DATA_ROOT, MANIFEST_ROOT, CATEGORIES, SHOTS, SEEDS,
                       PROTOCOL_ID, SPOT_DIFF_REVISION, ANOMALYDINO_REVISION,
                       sha256, save_json, inside, official_references)
from .validation import freeze_manifest_lock


def rows_for_six():
    with (PACKAGE / 'upstream/spot_diff_1cls.csv').open(newline='', encoding='utf-8') as stream:
        return [row for row in csv.DictReader(stream) if row['object'] in CATEGORIES]


def extract_required(archive, raw_root, rows):
    wanted = {row['image'] for row in rows} | {row['mask'] for row in rows if row['mask']}
    found = set()
    raw_root = Path(raw_root)
    with tarfile.open(archive, 'r:') as source:
        for member in source:
            parts = PurePosixPath(member.name).parts
            # Official archive may include a leading VisA/ or ./ directory.
            start = next((i for i, part in enumerate(parts) if part in CATEGORIES), None)
            if start is None:
                continue
            relative = PurePosixPath(*parts[start:]).as_posix()
            if relative not in wanted:
                continue
            if relative in found or not member.isfile():
                raise ValueError(f'Unexpected duplicate or non-regular member: {member.name}')
            target = inside(raw_root, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            # Always overwrite from archive: same-size corrupt files are not trusted.
            with source.extractfile(member) as incoming, target.open('wb') as output:
                shutil.copyfileobj(incoming, output)
            found.add(relative)
    missing = wanted - found
    if missing:
        raise ValueError(f'Archive is missing {len(missing)} official files: {sorted(missing)[:5]}')
    print(f'EXTRACTED {len(found)} required images/masks; only the six planned categories', flush=True)


def prepare_split(raw_root, data_root, rows):
    """Same copies and nonzero->255 mask conversion as pinned official prepare_data.py.

    The adaptation filters the CSV to our six categories; no split is resampled.
    """
    destinations = set()
    for row in rows:
        category, split = row['object'], row['split']
        label = 'good' if row['label'] == 'normal' else 'bad'
        if split not in ('train', 'test') or (split == 'train' and label != 'good'):
            raise ValueError(f'Unexpected official split row: {row}')
        relative = f'{category}/{split}/{label}/{PurePosixPath(row["image"]).name}'
        if relative in destinations:
            raise ValueError(f'Duplicate destination: {relative}')
        destinations.add(relative)
        target = inside(data_root, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(inside(raw_root, row['image']), target)
        if label == 'bad':
            mask_path = inside(data_root, f'{category}/ground_truth/bad/{PurePosixPath(row["mask"]).name}')
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(inside(raw_root, row['mask'])) as original:
                values = np.asarray(original)
                if values.ndim != 2:
                    raise ValueError('Expected single-channel VisA mask')
                Image.fromarray(np.where(values != 0, 255, 0).astype(np.uint8)).save(mask_path)


def inspect_image(path):
    with Image.open(path) as image:
        image.load()  # Decode the whole file, not only its header.
        if image.mode != 'RGB':
            raise ValueError(f'Unexpected image mode {image.mode}: {path}')
        width, height = image.size
    return {'width': width, 'height': height, 'sha256': sha256(path)}


def audit_and_build(data_root, raw_root, manifest_root, rows):
    counts = []
    for category in CATEGORIES:
        category_rows = [row for row in rows if row['object'] == category]
        train, tests, inventory = [], [], []
        expected_files = set()
        for row in category_rows:
            label = int(row['label'] != 'normal')
            relative = f'{category}/{row["split"]}/{"bad" if label else "good"}/{PurePosixPath(row["image"]).name}'
            expected_files.add(relative)
            path = inside(data_root, relative)
            item = {'path': relative, 'label': label, 'mask': None, **inspect_image(path)}
            if item['sha256'] != sha256(inside(raw_root, row['image'])):
                raise ValueError(f'Image differs from official archive: {relative}')
            if label:
                mask = f'{category}/ground_truth/bad/{PurePosixPath(row["mask"]).name}'
                expected_files.add(mask)
                with Image.open(inside(data_root, mask)) as image:
                    values = np.asarray(image)
                    if image.size != (item['width'], item['height']):
                        raise ValueError(f'Mask/image dimension mismatch: {relative}')
                    if values.ndim != 2 or not set(np.unique(values)).issubset({0, 255}) or not np.any(values):
                        raise ValueError(f'Missing, nonbinary or empty anomaly mask: {mask}')
                with Image.open(inside(raw_root, row['mask'])) as original:
                    if not np.array_equal(values > 0, np.asarray(original) > 0):
                        raise ValueError(f'Mask differs from official annotation: {mask}')
                item.update(mask=mask, mask_sha256=sha256(inside(data_root, mask)),
                            anomaly_pixels=int(np.count_nonzero(values)))
            inventory.append(item)
            if row['split'] == 'train':
                train.append(relative)
            else:
                tests.append(item)
        actual_files = {p.relative_to(data_root).as_posix() for p in (data_root / category).rglob('*') if p.is_file()}
        if actual_files != expected_files:
            raise ValueError(f'{category}: extra/missing prepared files, refusing an ambiguous test set')
        train.sort(key=lambda p: PurePosixPath(p).name)
        calibration, pool = train[:20], train[20:]
        if len(calibration) != 20 or len(pool) < 24:
            raise ValueError(f'{category}: insufficient normals')
        tests.sort(key=lambda sample: sample['path'])
        references = {str(k): {str(seed): official_references(pool, k, seed) for seed in SEEDS} for k in SHOTS}
        for per_seed in references.values():
            for selected in per_seed.values():
                if set(selected) & set(calibration):
                    raise ValueError('Calibration/reference overlap')
        manifest = {
            'protocol_id': PROTOCOL_ID, 'category': category,
            'path_base': 'VisA_pytorch/1cls (supply data_root on each machine)',
            'input_resolution': 448, 'preprocess': 'agnostic',
            'pixel_evaluation': 'native_original_image_and_mask', 'aupro_connectivity': 8,
            'calibration_rule': 'first_20_filename_sorted_official_normal_training_images',
            'reference_rule': 'sorted(remaining_pool)[seed*shots:(seed+1)*shots]',
            'reference_selection_is_random': False,
            'spot_diff_revision': SPOT_DIFF_REVISION, 'anomalydino_revision': ANOMALYDINO_REVISION,
            'official_split_sha256': sha256(PACKAGE / 'upstream/spot_diff_1cls.csv'),
            'calibration': calibration, 'normal_training_pool': pool,
            'references': references, 'test': tests,
            'inventory': sorted(inventory, key=lambda sample: sample['path']),
        }
        destination = manifest_root / f'{category}.json'
        if destination.exists() and json.loads(destination.read_text(encoding='utf-8')) != manifest:
            raise ValueError(f'Frozen manifest changed: {destination}. Use a new protocol directory.')
        save_json(destination, manifest)
        for name, lines in [('calibration', calibration), ('normal_training_pool', pool)]:
            output = manifest_root / name / f'{category}.txt'
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        for k in SHOTS:
            for seed in SEEDS:
                output = manifest_root / 'references' / f'{category}_{k}shot_seed{seed}.txt'
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text('\n'.join(references[str(k)][str(seed)]) + '\n', encoding='utf-8')
        label_counts = Counter(sample['label'] for sample in tests)
        counts.append({'category': category, 'train_total': len(train), 'calibration': 20,
                       'training_pool': len(pool), 'test_normal': label_counts[0],
                       'test_anomaly': label_counts[1], 'masks': label_counts[1],
                       'manifest_sha256': sha256(destination)})
        print('AUDIT OK', counts[-1], flush=True)
    report = {'protocol_id': PROTOCOL_ID, 'status': 'passed', 'categories': counts,
              'checks': ['official CSV membership and exact file counts', 'all image files fully decoded',
                         'image SHA256 matches archive', 'all anomaly masks binary, nonempty and image-sized',
                         'prepared masks match original nonzero annotations',
                         'calibration/reference/train/test separation',
                         'only six categories; all official test images retained'],
              'note': 'Data and manifest audit only; no detector has been trained or evaluated.'}
    save_json(manifest_root / 'data_audit.json', report)
    with (manifest_root / 'data_counts.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(counts[0]))
        writer.writeheader()
        writer.writerows(counts)
    freeze_manifest_lock(manifest_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT / 'data/downloads/VisA_20220922.tar')
    parser.add_argument('--raw-root', type=Path, default=ROOT / 'data/VisA')
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--manifest-root', type=Path, default=MANIFEST_ROOT)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    sources = json.loads((PACKAGE / 'upstream/sources.json').read_text())
    for filename, metadata in sources.items():
        if sha256(PACKAGE / 'upstream' / filename) != metadata['sha256']:
            raise ValueError(f'Pinned upstream file changed: {filename}')
    rows = rows_for_six()
    if not args.audit_only:
        extract_required(args.archive, args.raw_root, rows)
        prepare_split(args.raw_root, args.data_root, rows)
    audit_and_build(args.data_root, args.raw_root, args.manifest_root, rows)


if __name__ == '__main__':
    main()
