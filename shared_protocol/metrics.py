"""Exact native-size metrics, eight-connected AUPRO, no score normalization.

Streaming positive-score thresholds avoid stacking all native-resolution maps.
ROC, AP and PRO remain exact, including tied scores; no binning is performed.
"""
import numpy as np
from scipy.ndimage import label
from scipy.integrate import trapezoid
from sklearn.metrics import roc_auc_score, average_precision_score
from .experiments import validate_experiment_metadata


def normal_threshold(scores, expected_count=20):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) != expected_count or not np.isfinite(scores).all():
        raise ValueError(f'Expected exactly {expected_count} finite normal calibration scores')
    return float(np.quantile(scores, .95, method='linear'))


def _validate_pair(scores, mask):
    scores, mask = np.asarray(scores), np.asarray(mask)
    if scores.ndim != 2 or scores.shape != mask.shape or not scores.size:
        raise ValueError('Expected matching nonempty 2D anomaly map and mask')
    if scores.dtype.kind not in 'biuf' or not np.isfinite(scores).all():
        raise ValueError('Map scores must be finite real numbers')
    if not np.isin(mask, [0, 1]).all():
        raise ValueError('GT must be bool or binary 0/1; convert 0/255 first')
    if scores.dtype.kind != 'f':
        scores = scores.astype(np.float64)
    return scores, mask.astype(bool, copy=False)


def pixel_statistics_stream(pairs_factory, optional_metrics=True, pro_limit=.30, background_chunk_pixels=4_000_000):
    """Two passes over fresh (native_map, native_mask) pairs, including normal images."""
    if pro_limit != .30:
        raise ValueError('This protocol fixes AUPRO FPR limit at 0.30')
    if background_chunk_pixels <= 0:
        raise ValueError('Background chunk size must be positive')
    positives, weights, negative_total, n_regions = [], [], 0, 0
    layout = []
    for scores, mask in pairs_factory():
        scores, mask = _validate_pair(scores, mask)
        count = int(mask.sum())
        layout.append((scores.shape, count))
        negative_total += mask.size - count
        positives.append(scores[mask])
        if optional_metrics:
            regions, n = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
            sizes = np.bincount(regions.ravel())
            lookup = np.zeros(len(sizes), dtype=np.float64)
            if n:
                lookup[1:] = 1. / sizes[1:]
            weights.append(lookup[regions[mask]])
            n_regions += n
    if not positives or sum(len(p) for p in positives) == 0 or negative_total == 0:
        raise ValueError('Pixel AUROC requires both normal and anomalous pixels')
    scores = np.concatenate(positives)
    del positives
    order = np.argsort(-scores, kind='stable')
    sorted_scores = scores[order]
    del scores
    ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(sorted_scores) - 1]
    thresholds = sorted_scores[ends]
    positive_counts = np.diff(np.r_[-1, ends]).astype(np.int64)
    positive_total = int(positive_counts.sum())
    pro_after = None
    if optional_metrics:
        region_weights = np.concatenate(weights)
        del weights
        pro_after = np.cumsum(region_weights[order], dtype=np.float64)[ends] / n_regions
        pro_after = np.clip(pro_after, 0., 1.)
        del region_weights
    del order, sorted_scores, ends
    negative_less = np.zeros(len(thresholds), dtype=np.int64)
    negative_equal = np.zeros(len(thresholds), dtype=np.int64)
    chunks, chunk_length = [], 0

    def flush():
        nonlocal chunks, chunk_length
        if not chunks:
            return
        values = np.concatenate(chunks)
        values.sort()
        left = np.searchsorted(values, thresholds, side='left')
        right = np.searchsorted(values, thresholds, side='right')
        negative_less[:] += left
        negative_equal[:] += right - left
        chunks, chunk_length = [], 0

    second_layout = []
    for scores, mask in pairs_factory():
        scores, mask = _validate_pair(scores, mask)
        second_layout.append((scores.shape, int(mask.sum())))
        values = scores[~mask]
        chunks.append(values)
        chunk_length += len(values)
        if chunk_length >= background_chunk_pixels:
            flush()
    flush()
    if second_layout != layout:
        raise ValueError('Pixel data factory changed between passes')
    # Mann-Whitney ROC AUC, with half credit for equal positive/background scores.
    roc = np.sum(positive_counts * (negative_less.astype(np.float64) + .5 * negative_equal))
    result = {'pixel_auroc': float(roc / (positive_total * negative_total))}
    if optional_metrics:
        tp = np.cumsum(positive_counts, dtype=np.float64)
        fp = negative_total - negative_less
        result['pixel_ap'] = float(np.sum(positive_counts * (tp / (tp + fp))) / positive_total)
        # Before/after each positive threshold preserves horizontal background
        # segments and diagonal transitions caused by ties with background pixels.
        before = (negative_total - negative_less - negative_equal) / negative_total
        after = (negative_total - negative_less) / negative_total
        x = np.r_[0., np.column_stack((before, after)).ravel(), 1.]
        y = np.r_[0., np.column_stack((np.r_[0., pro_after[:-1]], pro_after)).ravel(), 1.]
        k = np.searchsorted(x, pro_limit, side='right')
        clipped_x, clipped_y = x[:k], y[:k]
        if clipped_x[-1] < pro_limit:
            end = clipped_y[-1] + (y[k] - clipped_y[-1]) * (pro_limit - clipped_x[-1]) / (x[k] - clipped_x[-1])
            clipped_x, clipped_y = np.r_[clipped_x, pro_limit], np.r_[clipped_y, end]
        result['aupro_03'] = float(np.clip(trapezoid(clipped_y, clipped_x) / pro_limit, 0., 1.))
    return result


def pixel_statistics(maps, masks, optional_metrics=True, pro_limit=.30):
    if len(maps) != len(masks) or not len(maps):
        raise ValueError('Expected one map for each GT mask')
    return pixel_statistics_stream(lambda: zip(maps, masks), optional_metrics, pro_limit)


def image_statistics(scores, calibration_scores, image_labels, optional_metrics=True):
    scores, y = np.asarray(scores, dtype=np.float64), np.asarray(image_labels)
    if scores.ndim != 1 or scores.shape != y.shape or not np.isfinite(scores).all():
        raise ValueError('Expected one finite score per labelled test image')
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Image labels must include both normal and anomalous samples')
    threshold = normal_threshold(calibration_scores)
    pred = scores > threshold
    result = {'status': 'success', 'image_auroc': float(roc_auc_score(y, scores)),
              'image_fpr': float(pred[y == 0].mean()), 'image_fnr': float((~pred[y == 1]).mean()),
              'image_threshold': threshold, 'threshold_quantile': .95, 'threshold_quantile_method': 'linear',
              'decision_rule': 'NG iff score > threshold', 'n_calibration': 20, 'n_test': len(scores),
              'n_test_normal': int((y == 0).sum()), 'n_test_anomaly': int((y == 1).sum())}
    if optional_metrics:
        result['image_ap'] = float(average_precision_score(y, scores))
    return result


def evaluate(maps, masks, scores, calibration_scores, image_labels=None, optional_metrics=True):
    if len(maps) != len(masks) or len(maps) != len(scores):
        raise ValueError('Expected matching image/map/mask counts')
    labels = [bool(np.asarray(m).any()) for m in masks] if image_labels is None else image_labels
    for mask, y in zip(masks, labels):
        if y == 0 and np.asarray(mask).any():
            raise ValueError('Normal image GT must be all zeros')
    result = image_statistics(scores, calibration_scores, labels, optional_metrics)
    result.update(pixel_statistics(maps, masks, optional_metrics))
    return result


def aggregate_runs(records, categories, shots=None, seeds=None):
    """Few-shot: 72 runs by default; zero-shot: one seed-0 run per category.

    Zero-shot seeds describe runtime randomness, never reference selection.
    A single run has no measured seed stability: its std is reported as null.
    """
    required = ('image_auroc', 'pixel_auroc')
    keys = (*required, 'image_ap', 'pixel_ap', 'aupro_03', 'image_fpr', 'image_fnr')
    indexed, methods, modes = {}, set(), set()
    for row in records:
        if row.get('integration_only'):
            raise ValueError('Integration-only checks must not enter formal experiment summaries')
        if row.get('status') != 'success':
            raise ValueError('Only successfully evaluated runs can be aggregated')
        if any(not row.get(k) for k in ('method', 'config_id', 'protocol_id')):
            raise ValueError('Run is missing method/config/protocol identity')
        modes.add(validate_experiment_metadata(row)['experiment_mode'])
        for metric in required:
            if metric not in row or not np.isscalar(row[metric]) or not np.isfinite(row[metric]) or not 0 <= row[metric] <= 1:
                raise ValueError(f'Missing/invalid required metric: {metric}')
        key = (row['category'], row['shots'], row['seed'])
        if key in indexed:
            raise ValueError(f'Duplicate run: {key}')
        indexed[key] = row
        methods.add((row['method'], row['config_id'], row['protocol_id']))
    if len(methods) != 1:
        raise ValueError('Do not mix methods/configs/protocols')
    if len(modes) != 1:
        raise ValueError('Do not mix zero-shot and few-shot runs in one summary')
    mode = next(iter(modes))
    shots = tuple(shots) if shots is not None else ((0,) if mode == 'zero_shot' else (1, 2, 4, 8))
    seeds = tuple(seeds) if seeds is not None else ((0,) if mode == 'zero_shot' else (0, 1, 2))
    if not shots or not seeds or len(set(shots)) != len(shots) or len(set(seeds)) != len(seeds):
        raise ValueError('Expected unique nonempty shot and seed lists')
    if (mode == 'zero_shot' and shots != (0,)) or (mode == 'few_shot' and 0 in shots):
        raise ValueError('Requested shot grid contradicts experiment mode')
    expected = {(c, k, s) for c in categories for k in shots for s in seeds}
    if set(indexed) != expected:
        raise ValueError(f'Run grid mismatch: missing {len(expected-set(indexed))}, extra {len(set(indexed)-expected)}')
    output = []
    for k in shots:
        method, config_id, protocol_id = next(iter(methods))
        summary = {'status': 'success', 'method': method, 'config_id': config_id, 'protocol_id': protocol_id,
                   'shots': k, 'n_categories': len(categories), 'seeds': list(seeds), 'std_ddof': 0}
        summary.update(experiment_mode=mode, n_runs_per_category=len(seeds),
                       seed_role='runtime_only' if mode == 'zero_shot' else 'reference_selection_and_runtime',
                       seed_stability_measured=len(seeds) > 1)
        for metric in keys:
            values = [indexed[c, k, s].get(metric) for c in categories for s in seeds]
            if all(value is None for value in values):
                continue
            if any(value is None or not np.isscalar(value) or not np.isfinite(value) or not 0 <= value <= 1 for value in values):
                raise ValueError(f'Missing/invalid metric {metric}')
            macros = [float(np.mean([indexed[c, k, s][metric] for c in categories])) for s in seeds]
            measure_std = mode != 'zero_shot' or len(seeds) > 1
            summary[metric] = {'mean': float(np.mean(macros)), 'std': float(np.std(macros, ddof=0)) if measure_std else None,
                               'per_seed_macro': macros,
                               'per_category': {c: {'mean': float(np.mean([indexed[c, k, s][metric] for s in seeds])),
                                                    'std': float(np.std([indexed[c, k, s][metric] for s in seeds], ddof=0)) if measure_std else None}
                                                for c in categories}}
        output.append(summary)
    return output
