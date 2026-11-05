"""Minimal smoke example for both teammates; this does not run a detector."""
import argparse

from .data import load_manifest, reference_paths, calibration_paths, training_paths, iter_test
from .protocol import CATEGORIES, DATA_ROOT


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--category', choices=CATEGORIES, default='candle')
    parser.add_argument('--shots', type=int, choices=[1, 2, 4, 8], default=1)
    parser.add_argument('--seed', type=int, choices=[0, 1, 2], default=0)
    parser.add_argument('--data-root', default=str(DATA_ROOT))
    args = parser.parse_args()
    manifest = load_manifest(args.category)
    print('Protocol:', manifest['protocol_id'])
    print('References:', reference_paths(manifest, args.shots, args.seed, args.data_root))
    print('Calibration images:', len(calibration_paths(manifest, args.data_root)))
    print('Normal training pool:', len(training_paths(manifest, args.data_root)))
    print('ALL official test images:', len(manifest['test']))
    sample, image, mask = next(iter_test(manifest, args.data_root))
    print('First test image:', sample['path'], 'label:', sample['label'])
    print('Original RGB shape:', image.shape, 'GT shape:', mask.shape, 'GT anomaly pixels:', int(mask.sum()))
    print('Data loaded successfully. No model has been run.')
