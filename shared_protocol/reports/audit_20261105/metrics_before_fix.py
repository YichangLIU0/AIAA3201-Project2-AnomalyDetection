"""Shared raw-score metrics, with no per-image min-max normalization.

Pixel ROC/AP/PRO tie-aware sorting and region weights adapted from the existing
coursework implementation. Required ROC metrics are checked against sklearn.
AUPRO: 4-connected GT components, equal component weights, all normal pixels
included in FPR, linear ROC interpolation at 0.30, integral divided by 0.30.
"""
import numpy as np
from scipy.ndimage import label
from scipy.integrate import trapezoid
from sklearn.metrics import roc_auc_score, average_precision_score


def normal_threshold(scores, expected_count=20):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) != expected_count or not np.isfinite(scores).all():
        raise ValueError(f'Expected exactly {expected_count} finite normal calibration scores')
    return float(np.quantile(scores, 0.95, method='linear'))


def _validate_pixels(maps, masks):
    maps, masks = np.asarray(maps), np.asarray(masks)
    if maps.ndim != 3 or maps.shape != masks.shape or not maps.size:
        raise ValueError('Expected matching nonempty [N,H,W] maps and GT masks')
    if maps.dtype.kind not in 'biuf':
        raise ValueError('Anomaly maps must contain real numeric scores')
    if maps.dtype.kind != 'f':
        maps = maps.astype(np.float64)  # Avoid unsigned overflow in descending-score sorting.
    if not np.isfinite(maps).all():
        raise ValueError('Anomaly maps contain NaN or infinity')
    if not np.isin(masks, [0, 1]).all():
        raise ValueError('GT masks must be binary 0/1 or bool; convert 0/255 masks first')
    masks = masks.astype(bool, copy=False)
    if not masks.any() or masks.all():
        raise ValueError('Pixel AUROC requires both normal and abnormal pixels')
    return maps, masks


def pixel_statistics(maps, masks, optional_metrics=True, pro_limit=0.30):
    maps, masks = _validate_pixels(maps, masks)
    if not 0 < pro_limit <= 1:
        raise ValueError('Invalid PRO FPR limit')
    scores, labels = maps.ravel(), masks.ravel()
    positives = int(labels.sum())
    negatives = len(labels) - positives
    order = np.argsort(-scores, kind='stable')
    ordered = scores[order]
    ends = np.r_[np.flatnonzero(ordered[1:] != ordered[:-1]), len(scores) - 1]
    del ordered
    tp = np.cumsum(labels[order], dtype=np.float64)[ends]
    fp = ends + 1 - tp
    fpr = np.r_[0., fp / negatives]
    result = {'pixel_auroc': float(trapezoid(np.r_[0., tp / positives], fpr))}
    if not optional_metrics:
        return result
    result['pixel_ap'] = float(np.sum(np.diff(np.r_[0., tp]) * (tp / (ends + 1))) / positives)
    del tp, fp
    # One component contributes a total weight of one, irrespective of its area.
    weights = np.zeros(masks.shape, dtype=np.float32)
    n_regions = 0
    for i, mask in enumerate(masks):
        regions, count = label(mask)  # scipy default: 4-connected in 2D.
        sizes = np.bincount(regions.ravel())
        if count:
            lookup = np.zeros(len(sizes), dtype=np.float32)
            lookup[1:] = 1. / sizes[1:]
            weights[i] = lookup[regions]
        n_regions += count
    pro = np.r_[0., np.cumsum(weights.ravel()[order], dtype=np.float64)[ends] / n_regions]
    k = np.searchsorted(fpr, pro_limit, side='right')
    x, y = fpr[:k], pro[:k]
    if x[-1] < pro_limit:
        y_end = y[-1] + (pro[k] - y[-1]) * (pro_limit - x[-1]) / (fpr[k] - x[-1])
        x, y = np.r_[x, pro_limit], np.r_[y, y_end]
    result['aupro_03'] = float(np.clip(trapezoid(y, x) / pro_limit, 0., 1.))
    return result


def evaluate(maps, masks, scores, calibration_scores, image_labels=None, optional_metrics=True):
    maps, masks = _validate_pixels(maps, masks)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(maps),) or not np.isfinite(scores).all():
        raise ValueError('Expected one finite raw image score per test image')
    # Explicit image labels prevent tiny defects disappearing after GT downsampling
    # from silently turning a known anomalous image into a normal image.
    y = masks.reshape(len(masks), -1).any(axis=1) if image_labels is None else np.asarray(image_labels)
    if y.shape != scores.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError('Image labels must include both normal and anomalous samples')
    if masks[y == 0].any():
        raise ValueError('Normal image GT must be all zeros')
    threshold = normal_threshold(calibration_scores)
    pred = scores > threshold  # Equality is OK, consistently for both partners.
    result = {
        'image_auroc': float(roc_auc_score(y, scores)),
        'image_fpr': float(pred[y == 0].mean()),
        'image_fnr': float((~pred[y == 1]).mean()),
        'image_threshold': threshold, 'threshold_quantile': 0.95,
        'threshold_quantile_method': 'linear', 'decision_rule': 'NG iff score > threshold',
        'n_calibration': 20, 'n_test': len(scores),
        'n_test_normal': int((y == 0).sum()), 'n_test_anomaly': int((y == 1).sum()),
    }
    if optional_metrics:
        result['image_ap'] = float(average_precision_score(y, scores))
    result.update(pixel_statistics(maps, masks, optional_metrics))
    return result


def aggregate_runs(records, categories, shots=(1, 2, 4, 8), seeds=(0, 1, 2)):
    """One method/config at a time; macro across categories, then mean/std seeds."""
    keys = ('image_auroc', 'pixel_auroc', 'image_ap', 'pixel_ap', 'aupro_03', 'image_fpr', 'image_fnr')
    indexed = {}
    methods = set()
    for row in records:
        key = (row['category'], int(row['shots']), int(row['seed']))
        if key in indexed:
            raise ValueError(f'Duplicate run: {key}')
        indexed[key] = row
        methods.add((row.get('method'), row.get('config_id'), row.get('protocol_id')))
    if len(methods) != 1:
        raise ValueError('Do not aggregate different methods/configs/protocols together')
    expected = {(c, k, s) for c in categories for k in shots for s in seeds}
    if set(indexed) != expected:
        raise ValueError(f'Run grid mismatch: missing {len(expected - set(indexed))}, extra {len(set(indexed) - expected)}')
    output = []
    for k in shots:
        summary = {'shots': k, 'n_categories': len(categories), 'seeds': list(seeds), 'std_ddof': 0}
        for metric in keys:
            values = [indexed[c, k, s].get(metric) for c in categories for s in seeds]
            if all(value is None for value in values):
                continue
            if any(value is None or not np.isfinite(value) for value in values):
                raise ValueError(f'Missing or nonfinite metric {metric}')
            macros = [float(np.mean([indexed[c, k, s][metric] for c in categories])) for s in seeds]
            summary[metric] = {'mean': float(np.mean(macros)), 'std': float(np.std(macros, ddof=0)),
                               'per_seed_macro': macros,
                               'per_category': {c: {'mean': float(np.mean([indexed[c, k, s][metric] for s in seeds])),
                                                    'std': float(np.std([indexed[c, k, s][metric] for s in seeds], ddof=0))}
                                                for c in categories}}
        output.append(summary)
    return output
