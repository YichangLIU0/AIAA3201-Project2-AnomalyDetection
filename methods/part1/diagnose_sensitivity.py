"""Fixed appearance stress checks on calibration normals, not new test metrics."""
from pathlib import Path
import argparse
import csv
import json

import cv2
import numpy as np

from shared_protocol.data import load_manifest, reference_paths
from shared_protocol.protocol import ROOT, CATEGORIES, inside, save_json, sha256
from .template import TemplateDetector, read_rgb, CONFIG


def translated(rgb):
    h, w = rgb.shape[:2]
    return cv2.warpAffine(rgb, np.array([[1., 0., .03*w], [0., 1., .03*h]]), (w, h),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def brighter(rgb):
    return np.clip(rgb.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--result-root', type=Path, default=ROOT / 'results/part1_template')
    args = parser.parse_args()
    audit = json.loads((args.result_root / 'completion_audit.json').read_text(encoding='utf-8'))
    if audit['status'] != 'pass' or audit['formal_runs'] != 72:
        raise ValueError('Finish and audit the formal experiment first')
    config = json.loads((args.result_root / 'frozen_config.json').read_text(encoding='utf-8'))
    assert all(sha256(Path(__file__).parent / name) == digest
               for name, digest in config['algorithm_source_sha256'].items())
    cv2.setNumThreads(CONFIG['opencv_threads'])
    detail, summary = [], []
    for category in CATEGORIES:
        manifest = load_manifest(category)
        metric = json.loads((args.result_root / 'runs' / category / '8shot_seed0/metrics.json').read_text(encoding='utf-8'))
        threshold = metric['image_threshold']
        model = TemplateDetector(reference_paths(manifest, 8, 0, args.data_root), manifest['references']['8']['0'])
        scores = {condition: [] for condition in ['clean', 'translate_xy_3percent', 'brightness_gain_1.15']}
        for relative in manifest['calibration']:
            rgb = read_rgb(inside(args.data_root, relative))
            for condition, image in [('clean', rgb), ('translate_xy_3percent', translated(rgb)),
                                     ('brightness_gain_1.15', brighter(rgb))]:
                score = model.predict_rgb(image).score
                scores[condition].append(score)
                detail.append({'category': category, 'source': 'calibration_normal_not_independent_test',
                               'shots': 8, 'seed': 0, 'condition': condition, 'path': relative,
                               'score': score, 'fixed_clean_threshold': threshold, 'predicted_ng': int(score > threshold)})
        for condition, values in scores.items():
            summary.append({'category': category, 'condition': condition, 'n': len(values),
                            'ng_count': int(np.sum(np.array(values) > threshold)),
                            'ng_fraction': float(np.mean(np.array(values) > threshold)),
                            'mean_score': float(np.mean(values)),
                            'mean_score_change_from_clean': float(np.mean(np.array(values) - scores['clean'])),
                            'fixed_clean_threshold': threshold})
    for filename, rows in [('calibration_sensitivity.csv', summary), ('calibration_sensitivity_images.csv', detail)]:
        with (args.result_root / filename).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    save_json(args.result_root / 'calibration_sensitivity.json', {
        'scope': 'Post-hoc appearance sensitivity on the SAME 20 calibration normals, not independent test accuracy or formal robustness metrics.',
        'model': 'Frozen 8-shot seed 0 per category; clean thresholds unchanged; no tuning after observation.',
        'translation': 'Original RGB translated +3% width and +3% height; OpenCV linear and reflect101.',
        'brightness': 'Original RGB multiplied by fixed 1.15 with uint8 clipping; no defect synthesized.',
        'limitations': 'Small diagnostic, uses calibration images, only one seed and two perturbations. Does not isolate background or prove real lighting generalization.',
        'results': summary,
    })
    section = ['## 正常图外观敏感性诊断', '',
               '**以下只诊断正常校准图，不是独立测试准确率，也不并入 72 组正式指标。**'
               '固定各类 8-shot、seed 0 与原始阈值；对同一批 20 张正常校准图，分别平移宽高的 3%，'
               '或把 RGB 亮度乘以 1.15。参数不随结果调整。表中为被判成 NG 的张数（总数 20）。', '',
               '| 类别 | 原图 NG/20 | 平移后 NG/20 | 增亮后 NG/20 |', '|---|---|---|---|']
    for category in CATEGORIES:
        subset = {r['condition']: r for r in summary if r['category'] == category}
        section.append('| ' + ' | '.join([category, *[str(subset[c]['ng_count']) for c in
                       ['clean', 'translate_xy_3percent', 'brightness_gain_1.15']]]) + ' |')
    section += ['', '原图只出现少量 NG 是因为阈值本来就由这 20 张图校准，不能据此宣称独立测试误报很低。'
                '扰动前后变化可以辅助判断外观敏感性，但不能单独证明真实光照变化的效果。'
                '本诊断没有物体/背景分割，未把整图增亮解释为纯背景实验。逐图结果与完整说明见 `calibration_sensitivity*.csv/json`。', '']
    path = args.result_root / '实验结果与错误分析.md'
    text = path.read_text(encoding='utf-8').split('## 正常图外观敏感性诊断')[0].rstrip()
    path.write_text(text + '\n\n' + '\n'.join(section), encoding='utf-8')
    print('Saved calibration-only sensitivity check; formal scores unchanged.', flush=True)


if __name__ == '__main__':
    main()
