"""Rigid ECC reference-to-query registration, with explicit safe fallbacks.

API direction follows https://docs.opencv.org/4.x/dc/d6b/group__video__track.html
Query is templateImage, reference is inputImage; sample with WARP_INVERSE_MAP.
No masks, labels or calibration outcomes choose the geometric transformation.
"""
from dataclasses import dataclass
import math

import cv2
import numpy as np

from .template import CONFIG as BASE_CONFIG, TemplateDetector, TemplatePrediction
from .template import preprocess, descriptor, process_difference, image_score

CONFIG = {**BASE_CONFIG, 'method': 'template_rgb_rot8_ecc', 'version': 1,
          'alignment': 'ECC Euclidean; warp selected reference into unchanged query coordinates',
          'ecc_max_edge': 224, 'ecc_max_iterations': 50, 'ecc_epsilon': 1e-5,
          'ecc_gaussian_kernel': 5, 'ecc_init': 'identity after discrete reference rotation',
          'ecc_max_rotation_degrees': 30., 'ecc_max_center_displacement_fraction': .15,
          'ecc_min_valid_fraction': .75,
          'ecc_acceptance': 'finite bounded transform, sufficient overlap, nondecreasing valid-pixel grayscale correlation',
          'ecc_interpolation': 'INTER_LINEAR | WARP_INVERSE_MAP; constant border',
          'ecc_border': 'outside valid warped-reference support, retain unaligned reference; never copy query or zero error',
          'ecc_fallback': 'original unaligned selected reference, explicit reason logged',
          'parameter_selection': 'fixed geometric settings before formal ECC test outcomes; synthetic registration checks only'}


def correlation(a, b, valid):
    x, y = a[valid].astype(np.float64), b[valid].astype(np.float64)
    x -= x.mean(); y -= y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denom) if denom > 1e-12 else None


def align_reference(query, reference):
    if query.shape != reference.shape:
        raise ValueError('Align equal RGB geometry only')
    h, w = query.shape[:2]
    identity = np.eye(2, 3, dtype=np.float32)
    trace = {'status': 'fallback', 'reason': None, 'warp_query_to_reference': identity.tolist(),
             'valid_fraction': 1., 'correlation_before': None, 'correlation_after': None}
    qgray, rgray = (cv2.cvtColor(v, cv2.COLOR_RGB2GRAY) for v in [query, reference])
    if min(float(qgray.std()), float(rgray.std())) < 1e-5:
        trace['reason'] = 'constant_image'
        return reference.copy(), trace
    scale = min(1., CONFIG['ecc_max_edge'] / max(h, w))
    nw, nh = max(8, round(w * scale)), max(8, round(h * scale))
    small_query, small_ref = [cv2.resize(v, (nw, nh), interpolation=cv2.INTER_AREA) for v in [qgray, rgray]]
    try:
        _, small_warp = cv2.findTransformECC(
            small_query, small_ref, identity.copy(), cv2.MOTION_EUCLIDEAN,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, CONFIG['ecc_max_iterations'], CONFIG['ecc_epsilon']),
            None, CONFIG['ecc_gaussian_kernel'])
    except cv2.error:
        trace['reason'] = 'ecc_nonconvergence'
        return reference.copy(), trace
    scaling = np.diag([nw / w, nh / h, 1.])
    warp = (np.linalg.inv(scaling) @ np.vstack([small_warp, [0., 0., 1.]]) @ scaling)[:2].astype(np.float32)
    if not np.isfinite(warp).all():
        trace['reason'] = 'nonfinite_transform'
        return reference.copy(), trace
    angle = math.degrees(math.atan2(float(small_warp[1, 0]), float(small_warp[0, 0])))
    center = np.array([w/2, h/2, 1.])
    center_delta = warp @ center - center[:2]
    trace.update(candidate_warp_query_to_reference=warp.tolist(), candidate_rotation_degrees=angle,
                 candidate_center_displacement_xy=center_delta.tolist())
    if abs(angle) > CONFIG['ecc_max_rotation_degrees'] or np.any(
            np.abs(center_delta) > np.array([w, h]) * CONFIG['ecc_max_center_displacement_fraction']):
        trace['reason'] = 'transform_out_of_bounds'
        return reference.copy(), trace
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    support = cv2.warpAffine(np.ones((h, w), np.float32), warp, (w, h), flags=flags,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = support >= .999
    trace['candidate_valid_fraction'] = float(valid.mean())
    if valid.mean() < CONFIG['ecc_min_valid_fraction']:
        trace['reason'] = 'insufficient_overlap'
        return reference.copy(), trace
    aligned = cv2.warpAffine(reference, warp, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT)
    before = correlation(qgray, rgray, valid)
    after = correlation(qgray, cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY), valid)
    trace.update(correlation_before=before, correlation_after=after)
    if before is None or after is None or after < before - 1e-6:
        trace['reason'] = 'correlation_not_improved'
        return reference.copy(), trace
    aligned[~valid] = reference[~valid]
    trace.update(status='aligned', reason=None, warp_query_to_reference=warp.tolist(), valid_fraction=float(valid.mean()))
    return aligned, trace


@dataclass
class ECCPrediction(TemplatePrediction):
    alignment_trace: dict


class ECCDetector(TemplateDetector):
    def predict_rgb(self, original):
        query = preprocess(original)
        distances = ((self.descriptors - descriptor(query)[None]) ** 2).mean(axis=(1, 2, 3))
        index = int(np.argmin(distances))
        reference = self.templates[index]
        if reference.shape != query.shape:
            reference = cv2.resize(reference, (query.shape[1], query.shape[0]), interpolation=cv2.INTER_CUBIC)
        aligned, trace = align_reference(query, reference)
        error = process_difference(query, aligned)
        native = cv2.resize(error, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR)
        name, angle = self.tags[index]
        return ECCPrediction(image_score(error), native, error, name, angle,
                             float(distances[index]), query.shape[:2], trace)
