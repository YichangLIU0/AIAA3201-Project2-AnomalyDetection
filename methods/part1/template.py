"""Fixed classical Template Difference baseline; no fitting on test labels.

Geometric input handling follows the pinned AnomalyDINO convention. Reference
rotation uses its OpenCV center/angle/interpolation/reflection convention.
The required baseline is deliberately simple; optional ECC/AE are separate work.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


CONFIG = {
    'method': 'template_rgb_rot8', 'version': 1, 'input_resolution': 448,
    'geometry': 'short_edge_448_bicubic_then_top_left_crop_to_multiple14',
    'reference_angles_degrees': [0, 45, 90, 135, 180, 225, 270, 315],
    'rotation': 'original RGB canvas, OpenCV linear, BORDER_REFLECT_101',
    'descriptor': '28x28 RGB INTER_AREA, float32, MSE',
    'selection': 'minimum descriptor MSE over actual K references and 8 rotations',
    'reference_shape_fallback': 'INTER_CUBIC resize to query shape if processed shapes differ',
    'alignment': 'none', 'difference': 'mean absolute RGB / 255 global constant',
    'smoothing': 'OpenCV Gaussian sigma2 BORDER_REFLECT_101',
    'morphology': '3x3 grayscale closing BORDER_REFLECT_101',
    'image_score': 'mean of largest floor(1 percent * input_map_pixel_count), at least 1',
    'native_map': 'OpenCV INTER_LINEAR to original image height/width',
    'mask_processing': 'unchanged original GT, normal GT all zeros',
    'device': 'CPU', 'opencv_threads': 4, 'test_time_augmentation': 'none',
    'parameter_selection': 'fixed a priori from coursework roadmap; no test-label search',
}


def read_rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert('RGB'))


def preprocess(rgb):
    h, w = rgb.shape[:2]
    if h <= w:
        nh, nw = 448, int(448 * w / h)
    else:
        nh, nw = int(448 * h / w), 448
    image = np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.Resampling.BICUBIC))
    return np.ascontiguousarray(image[:nh - nh % 14, :nw - nw % 14])


def rotate_reference(rgb, angle):
    h, w = rgb.shape[:2]
    transform = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.)
    return cv2.warpAffine(rgb, transform, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def descriptor(rgb):
    return cv2.resize(rgb, (28, 28), interpolation=cv2.INTER_AREA).astype(np.float32)


def process_difference(query, reference):
    error = np.abs(query.astype(np.float32) - reference.astype(np.float32)).mean(axis=2) / 255.
    error = cv2.GaussianBlur(error, (0, 0), 2., borderType=cv2.BORDER_REFLECT_101)
    return cv2.morphologyEx(error, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8),
                            borderType=cv2.BORDER_REFLECT_101)


def image_score(error):
    flat = np.asarray(error).ravel()
    k = max(1, int(len(flat) * .01))
    return float(np.partition(flat, len(flat) - k)[-k:].mean())


@dataclass
class TemplatePrediction:
    score: float
    native_map: np.ndarray
    input_map: np.ndarray
    chosen_reference: str
    chosen_angle: int
    descriptor_mse: float
    input_shape: tuple


class TemplateDetector:
    def __init__(self, reference_paths, reference_names=None):
        if not reference_paths:
            raise ValueError('Template needs at least one genuine normal reference')
        self.names = list(reference_names or [str(p) for p in reference_paths])
        if len(self.names) != len(reference_paths):
            raise ValueError('Reference path/name count mismatch')
        self.templates, self.tags = [], []
        for path, name in zip(reference_paths, self.names):
            rgb = read_rgb(path)
            for angle in CONFIG['reference_angles_degrees']:
                self.templates.append(preprocess(rotate_reference(rgb, angle)))
                self.tags.append((name, angle))
        self.descriptors = np.stack([descriptor(rgb) for rgb in self.templates])

    def predict_rgb(self, original):
        query = preprocess(original)
        distances = ((self.descriptors - descriptor(query)[None]) ** 2).mean(axis=(1, 2, 3))
        index = int(np.argmin(distances))
        reference = self.templates[index]
        # Categories normally share native geometry. This fixed fallback is recorded
        # explicitly and never uses a test label or a defect mask.
        if reference.shape != query.shape:
            reference = cv2.resize(reference, (query.shape[1], query.shape[0]), interpolation=cv2.INTER_CUBIC)
        error = process_difference(query, reference)
        native = cv2.resize(error, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR)
        name, angle = self.tags[index]
        return TemplatePrediction(image_score(error), native, error, name, angle, float(distances[index]), query.shape[:2])

    def predict_path(self, path):
        return self.predict_rgb(read_rgb(path))
