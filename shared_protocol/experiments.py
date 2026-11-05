"""Experiment metadata: reference selection and runtime randomness are separate."""
from numbers import Integral

from .protocol import SHOTS, SEEDS


def experiment_spec(shots, seed, method=None, experiment_mode=None):
    if isinstance(shots, bool) or not isinstance(shots, Integral):
        raise ValueError('shots must be an integer: 0 for zero-shot, or 1/2/4/8 for few-shot')
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError('seed must be a nonnegative runtime/reference index')
    mode = 'zero_shot' if shots == 0 else 'few_shot'
    if experiment_mode is not None and experiment_mode != mode:
        raise ValueError('Experiment mode contradicts the number of reference images')
    canonical_method = str(method or '').lower().replace('_', '').replace('-', '').replace(' ', '')
    if canonical_method == 'anomalyclip' and shots != 0:
        raise ValueError('AnomalyCLIP must use shots=0 with no reference images; do not fake a one-shot run')
    if mode == 'few_shot' and (shots not in SHOTS or seed not in SEEDS):
        raise ValueError('Few-shot protocol requires shots 1/2/4/8 and reference seeds 0/1/2')
    return {'experiment_mode': mode, 'run_seed': int(seed),
            'reference_seed': None if mode == 'zero_shot' else int(seed),
            'seed_role': 'runtime_only' if mode == 'zero_shot' else 'reference_selection_and_runtime',
            'calibration_usage': '20 normal images for threshold only; no training or reference bank'}


def expected_references(manifest, shots, seed, method=None, experiment_mode=None):
    spec = experiment_spec(shots, seed, method, experiment_mode)
    return [] if spec['experiment_mode'] == 'zero_shot' else manifest['references'][str(shots)][str(seed)]


def validate_experiment_metadata(metadata):
    spec = experiment_spec(metadata['shots'], metadata['seed'], metadata.get('method'),
                           metadata.get('experiment_mode'))
    for key in ('experiment_mode', 'reference_seed', 'run_seed', 'seed_role'):
        if key in metadata and metadata[key] != spec[key]:
            raise ValueError(f'Contradictory experiment metadata: {key}')
    if spec['experiment_mode'] == 'zero_shot':
        if metadata.get('reference_paths') != []:
            raise ValueError('Zero-shot runs must have reference_paths=[]')
        if metadata.get('experiment_mode') != 'zero_shot' or 'reference_seed' not in metadata:
            raise ValueError('Zero-shot runs must explicitly record mode=zero_shot and reference_seed=null')
    return spec
