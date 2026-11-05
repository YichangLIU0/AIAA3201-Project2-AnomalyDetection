"""Read-only audit of delivered data/tools against pinned teacher/upstream rules.

Only audit evidence is written. Detector code, data, and frozen manifests are not changed.
Synthetic guard probes and numerical examples are never model results.
Run from the project root: python shared_protocol/reports/audit_20261105/run_independent_audit.py
"""
from pathlib import Path, PurePosixPath
from collections import Counter, defaultdict
from bisect import bisect
import ast
import contextlib
import csv
import hashlib
import io
import json
import sys
import tarfile
import tempfile
import time

import numpy as np
from PIL import Image
from scipy.ndimage import label
from sklearn.metrics import roc_auc_score, average_precision_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))
from shared_protocol.metrics import pixel_statistics, normal_threshold, aggregate_runs
from shared_protocol.data import load_manifest, reference_paths

CATEGORIES = ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'pipe_fryum')
PACKAGE = ROOT / 'shared_protocol'
DATA = ROOT / 'data/VisA_pytorch/1cls'


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def check(condition, description):
    if not condition:
        raise AssertionError(description)


def audit_data():
    with (PACKAGE / 'upstream/spot_diff_1cls.csv').open(encoding='utf-8', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['object'] in CATEGORIES]
    reports = []
    current_hashes = {}
    masks_to_check = []
    for category in CATEGORIES:
        manifest = json.loads((PACKAGE / 'manifests' / f'{category}.json').read_text(encoding='utf-8'))
        cr = [r for r in rows if r['object'] == category]
        expected_train, expected_test, expected_masks, expected_files = [], {}, {}, set()
        for row in cr:
            is_anomaly = row['label'] == 'anomaly'
            relative = f'{category}/{row["split"]}/{"bad" if is_anomaly else "good"}/{PurePosixPath(row["image"]).name}'
            expected_files.add(relative)
            if row['split'] == 'train':
                check(not is_anomaly, 'Non-normal training image')
                expected_train.append(relative)
            else:
                mp = f'{category}/ground_truth/bad/{PurePosixPath(row["mask"]).name}' if is_anomaly else None
                expected_test[relative] = (int(is_anomaly), mp)
                if mp:
                    expected_files.add(mp)
                    expected_masks[mp] = row['mask']
            current_hashes[row['image']] = digest(DATA / relative)
        actual_files = {p.relative_to(DATA).as_posix() for p in (DATA / category).rglob('*') if p.is_file()}
        check(expected_files == actual_files, f'{category}: exact official file membership')
        expected_train.sort(key=lambda p: PurePosixPath(p).name)
        check(manifest['calibration'] == expected_train[:20], f'{category}: first 20 calibration')
        check(manifest['normal_training_pool'] == expected_train[20:], f'{category}: remaining train pool')
        test_paths = [r['path'] for r in manifest['test']]
        check(len(test_paths) == len(set(test_paths)) and set(test_paths) == set(expected_test), f'{category}: all test images')
        for sample in manifest['test']:
            check((sample['label'], sample['mask']) == expected_test[sample['path']], f'{category}: test label/mask link')
            if sample['label']:
                masks_to_check.append((category, sample))
        for shots in (1, 2, 4, 8):
            for seed in (0, 1, 2):
                expected = expected_train[20:][seed * shots:(seed + 1) * shots]
                actual = manifest['references'][str(shots)][str(seed)]
                txt = PACKAGE / 'manifests/references' / f'{category}_{shots}shot_seed{seed}.txt'
                check(actual == expected == txt.read_text(encoding='utf-8').splitlines(), 'Reference JSON/TXT/official rule mismatch')
        for key in ('calibration', 'normal_training_pool'):
            txt = PACKAGE / 'manifests' / key / f'{category}.txt'
            check(txt.read_text(encoding='utf-8').splitlines() == manifest[key], f'{category}: TXT/JSON mismatch')
        by_hash = defaultdict(list)
        for sample in manifest['inventory']:
            actual_hash = digest(DATA / sample['path'])
            check(actual_hash == sample['sha256'], f'{category}: current image checksum')
            by_hash[actual_hash].append(sample['path'])
            if sample['mask']:
                check(digest(DATA / sample['mask']) == sample['mask_sha256'], f'{category}: current mask checksum')
        leaks = []
        for paths in by_hash.values():
            roles = {'calibration' if p in manifest['calibration'] else 'train' if '/train/' in p else 'test' for p in paths}
            if len(roles) > 1:
                leaks.append(paths)
        check(not leaks, f'{category}: byte-identical image leakage across calibration/train/test')
        reports.append({'category': category, 'train': len(expected_train), 'calibration': 20,
                        'test': len(expected_test), 'anomaly_masks': len(expected_masks),
                        'reference_lists': 12, 'txt_json_match': True, 'checksum_match': True,
                        'cross_split_identical_image_leaks': leaks,
                        'test_shapes': dict(Counter(f'{r["height"]}x{r["width"]}' for r in manifest['test']))})
        print('INDEPENDENT DATA PASS', category, flush=True)

    # Compare every delivered original image directly with the official TAR members,
    # rather than trusting the previous extraction directory or audit flags.
    found = set()
    with tarfile.open(ROOT / 'data/downloads/VisA_20220922.tar', 'r:') as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            index = next((i for i, p in enumerate(parts) if p in CATEGORIES), None)
            if index is None or not member.isfile():
                continue
            relative = PurePosixPath(*parts[index:]).as_posix()
            if relative in current_hashes:
                with archive.extractfile(member) as incoming:
                    actual = hashlib.file_digest(incoming, 'sha256').hexdigest()
                check(actual == current_hashes[relative], f'Archive versus prepared image mismatch: {relative}')
                found.add(relative)
    check(found == set(current_hashes), 'Not all delivered images found in official archive')
    print('ARCHIVE COMPARISON PASS', len(found), 'original images', flush=True)

    resize = []
    for category in CATEGORIES:
        findings = {'category': category, 'anomaly_images': 0, 'entire_masks_lost_at_448': [],
                    'native_8_connected_regions': 0, 'native_4_connected_regions': 0,
                    'native_regions_lost_at_448': 0, 'images_losing_at_least_one_region': [],
                    'images_with_different_4_vs_8_connectivity': 0}
        for _, sample in [pair for pair in masks_to_check if pair[0] == category]:
            with Image.open(DATA / sample['mask']) as image:
                native = np.asarray(image) > 0
                reduced = np.asarray(image.resize((448, 448), Image.Resampling.NEAREST)) > 0
            with Image.open(ROOT / 'data/VisA' / expected_raw_mask(sample['mask'])) as original:
                check(np.array_equal(native, np.asarray(original) > 0), 'Raw/prepared mask mismatch')
            native_regions, n8 = label(native, np.ones((3, 3), dtype=int))
            _, n4 = label(native)
            resized_labels = np.asarray(Image.fromarray(native_regions.astype(np.int32)).resize((448, 448), Image.Resampling.NEAREST))
            retained = set(np.unique(resized_labels)) - {0}
            lost = n8 - len(retained)
            findings['anomaly_images'] += 1
            findings['native_8_connected_regions'] += n8
            findings['native_4_connected_regions'] += n4
            findings['native_regions_lost_at_448'] += lost
            findings['images_with_different_4_vs_8_connectivity'] += int(n8 != n4)
            if lost:
                findings['images_losing_at_least_one_region'].append({'path': sample['path'], 'regions_lost': lost})
            if not reduced.any():
                findings['entire_masks_lost_at_448'].append(sample['path'])
        resize.append(findings)
        print('MASK RESIZE CHECK', category, 'lost components:', findings['native_regions_lost_at_448'], flush=True)
    return reports, resize


def expected_raw_mask(prepared):
    category = prepared.split('/')[0]
    return f'{category}/Data/Masks/Anomaly/{Path(prepared).name}'


def audit_numerics_and_guards():
    # Execute only reviewed pure numerical functions from the pinned upstream file.
    # No upstream imports or top-level executable statements are evaluated.
    source = OUT / 'official_post_eval.py'
    parsed = ast.parse(source.read_text(encoding='utf-8'))
    functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in ('compute_pro', 'trapezoid')]
    env = {'np': np, 'label': label, 'bisect': bisect}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), env)
    gt = np.zeros((2, 4, 4), dtype=bool)
    gt[1, 0, 0] = gt[1, 1, 1] = gt[1, 1, 2] = True
    maps = np.full(gt.shape, .5)
    maps[gt] = 0.
    maps[1, 0, 0] = 1.
    ours = pixel_statistics(maps, gt)
    with contextlib.redirect_stdout(io.StringIO()):
        fpr, pro = env['compute_pro'](maps, gt)
    official = float(env['trapezoid'](fpr, pro, x_max=.3) / .3)
    numerical = {'scope': 'Synthetic numerical examples, never detector results',
                 'threshold_arange20': normal_threshold(np.arange(20)),
                 'threshold_expected': 18.05, 'diagonal_fixture_shared_aupro': ours['aupro_03'],
                 'diagonal_fixture_official_aupro': official,
                 'pixel_auroc_fixture_shared': ours['pixel_auroc'],
                 'pixel_auroc_fixture_sklearn': float(roc_auc_score(gt.ravel(), maps.ravel())),
                 'pixel_ap_fixture_shared': ours['pixel_ap'],
                 'pixel_ap_fixture_sklearn': float(average_precision_score(gt.ravel(), maps.ravel()))}

    # A changed manifest should not be trusted just because its ID looks right.
    with tempfile.TemporaryDirectory(prefix='audit_fixture_', dir=OUT) as directory:
        temp = Path(directory).resolve()
        check(temp.is_relative_to(OUT.resolve()), 'Temporary test path outside audit directory')
        manifest = json.loads((PACKAGE / 'manifests/candle.json').read_text(encoding='utf-8'))
        manifest['references']['1']['0'] = manifest['calibration'][:1]
        (temp / 'candle.json').write_text(json.dumps(manifest), encoding='utf-8')
        accepted = load_manifest('candle', manifest_root=temp)
        overlap_accepted = reference_paths(accepted, 1, 0)[0].relative_to(DATA).as_posix() in manifest['calibration']
    records = [{'category': c, 'shots': 1, 'seed': s, 'method': 'AUDIT_FIXTURE', 'config_id': 'TEST_ONLY',
                'protocol_id': 'TEST_ONLY'} for c in CATEGORIES for s in (0, 1, 2)]
    missing_metrics_accepted = aggregate_runs(records, CATEGORIES, shots=[1])
    guards = {'scope': 'Temporary or in-memory fixtures; delivered manifests were not changed',
              'loader_accepts_changed_reference_overlapping_calibration': overlap_accepted,
              'aggregate_accepts_runs_with_no_required_metrics': missing_metrics_accepted,
              'interpretation': 'Guard gaps; these probes do not imply the current saved lists are wrong.'}
    return numerical, guards


def main():
    started = time.monotonic()
    data, resize = audit_data()
    numeric, guards = audit_numerics_and_guards()
    result = {'audit_scope': 'Previously delivered public preparation, teacher PDF and v2 plan',
              'data': data, 'resize_and_connectivity': resize, 'numerical_crosscheck': numeric,
              'guard_probes': guards, 'seconds': time.monotonic() - started,
              'no_new_model_experiments': True}
    (OUT / 'independent_audit_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('NUMERICAL', numeric, flush=True)
    print('GUARDS', guards, flush=True)
    print('AUDIT EVIDENCE SAVED', OUT / 'independent_audit_results.json', flush=True)


if __name__ == '__main__':
    main()
