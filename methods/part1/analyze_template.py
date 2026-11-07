"""Post-hoc figures and completeness audit. Never changes frozen predictions."""
from pathlib import Path
import argparse
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from shared_protocol.data import load_manifest, load_mask
from shared_protocol.protocol import ROOT, CATEGORIES, inside, save_json, sha256
from shared_protocol.metrics import aggregate_runs
from .template import read_rgb, rotate_reference, preprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/part1_template_v1')
    parser.add_argument('--result-root', type=Path, default=ROOT / 'results/part1_template')
    args = parser.parse_args()
    config = json.loads((args.output_root / 'frozen_config.json').read_text(encoding='utf-8'))
    records = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(args.output_root.glob('*/*/metrics.json'))]
    summaries = aggregate_runs(records, CATEGORIES)  # Refuse partial/failed/mixed 72-cell grids.
    assert all(r['run_config'] == config for r in records)
    assert all(sha256(Path(__file__).parent / name) == digest
               for name, digest in config['algorithm_source_sha256'].items())
    maps = 0
    for row in records:
        folder = args.output_root / row['category'] / f"{row['shots']}shot_seed{row['seed']}"
        raw = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
        assert sha256(folder / 'run.json') == row['prediction_manifest_sha256']
        manifest = load_manifest(row['category'])
        assert row['n_test'] == len(manifest['test']) and row['n_calibration'] == 20
        assert raw['reference_paths'] == manifest['references'][str(row['shots'])][str(row['seed'])]
        assert len(list((folder / 'maps').glob('*.npy'))) == len(manifest['test'])
        assert (args.result_root / 'runs' / row['category'] / folder.name / 'image_predictions.csv').is_file()
        compact = args.result_root / 'runs' / row['category'] / folder.name
        # Lightweight provenance, including all native-map SHA256 values. The
        # maps themselves remain in outputs; do not imply this folder is standalone.
        save_json(compact / 'prediction_manifest.json', raw)
        with (compact / 'calibration_scores.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['path', 'score'])
            writer.writerows(zip(raw['calibration_paths'], raw['calibration_scores']))
        maps += len(raw['test'])
    save_json(args.result_root / 'completion_audit.json', {
        'status': 'pass', 'formal_runs': len(records), 'expected_runs': 72,
        'prediction_maps': maps, 'expected_prediction_maps': 960 * 12,
        'all_metrics_successful': True, 'all_integration_only_false': True,
        'source_hashes_match_frozen_config': True,
        'run_manifest_hashes_match_evaluation': True,
        'native_maps_and_original_masks_validated_by_shared_evaluator': True,
        'scope': 'Template only; does not certify ECC, AE, DINO, CLIP, Part 3 or final submission',
    })
    figures = args.result_root / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    shots = [s['shots'] for s in summaries]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, key, title in zip(axes, ['image_auroc', 'pixel_auroc'], ['Image AUROC', 'Native pixel AUROC']):
        for category in CATEGORIES:
            ax.plot(shots, [s[key]['per_category'][category]['mean'] for s in summaries], marker='o', alpha=.65, label=category)
        ax.errorbar(shots, [s[key]['mean'] for s in summaries], yerr=[s[key]['std'] for s in summaries],
                    color='black', linewidth=2.5, marker='s', capsize=4, label='6-class macro +/- seed std')
        ax.axhline(.5, color='gray', linestyle='--', linewidth=.8)
        ax.set(title=title, xlabel='Normal reference images (shots)', ylabel='AUROC', ylim=(0, 1), xticks=shots)
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=7, loc='lower right')
    fig.savefig(figures / 'auroc_by_shots.png', dpi=180)
    plt.close(fig)
    case_rows, thumbs = [], []
    # Fixed analysis slice: 8-shot seed 0, one representative error per class.
    # Alternate FP/FN preferences to show both types, never select a model by test performance.
    for i, category in enumerate(CATEGORIES):
        folder = args.output_root / category / '8shot_seed0'
        raw = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
        metric = next(r for r in records if r['category'] == category and r['shots'] == 8 and r['seed'] == 0)
        choices = {r['path']: r for r in json.loads((folder / 'template_choices.json').read_text(encoding='utf-8'))}
        manifest = load_manifest(category)
        threshold = metric['image_threshold']
        paired = list(zip(manifest['test'], raw['test']))
        fp = [(s, r) for s, r in paired if s['label'] == 0 and r['score'] > threshold]
        fn = [(s, r) for s, r in paired if s['label'] == 1 and r['score'] <= threshold]
        preference = ['FP', 'FN'][i % 2]
        if preference == 'FP' and fp:
            sample, prediction = max(fp, key=lambda x: x[1]['score'] - threshold)
            kind = 'FP'
        elif fn:
            sample, prediction = min(fn, key=lambda x: x[1]['score'] - threshold)
            kind = 'FN'
        elif fp:
            sample, prediction = max(fp, key=lambda x: x[1]['score'] - threshold)
            kind = 'FP'
        else:
            sample, prediction = min((p for p in paired if p[0]['label'] == 1), key=lambda x: x[1]['score'] - threshold)
            kind = 'TP (smallest margin; no image errors in this cell)'
        rgb = read_rgb(inside(args.data_root, sample['path']))
        mask = load_mask(sample, args.data_root)
        values = np.load(folder / prediction['map'], allow_pickle=False)
        choice = choices[sample['path']]
        reference = preprocess(rotate_reference(read_rgb(inside(args.data_root, choice['reference'])), choice['angle']))
        top = values >= np.quantile(values, .99)
        focus = float(mask[top].mean())
        row = {'category': category, 'shots': 8, 'seed': 0, 'error_type': kind, 'path': sample['path'],
               'reference': choice['reference'], 'angle': choice['angle'], 'label': sample['label'],
               'image_score': prediction['score'], 'threshold': threshold,
               'gt_area_fraction': float(mask.mean()), 'top1_percent_map_gt_fraction': focus,
               'heatmap_display_range': [0, .5], 'map': prediction['map']}
        case_rows.append(row)
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), constrained_layout=True)
        axes[0].imshow(rgb); axes[0].set_title('Test image')
        axes[1].imshow(mask, vmin=0, vmax=1, cmap='gray'); axes[1].set_title('Original GT (white=defect)')
        axes[2].imshow(rgb)
        heat = axes[2].imshow(values, cmap='inferno', alpha=.68, vmin=0, vmax=.5)
        if mask.any():
            axes[2].contour(mask, levels=[.5], colors=['cyan'], linewidths=.65)
        axes[2].set_title('Raw heatmap; cyan=GT')
        axes[3].imshow(reference); axes[3].set_title(f"Selected normal ref / {choice['angle']} deg")
        for ax in axes:
            ax.axis('off')
        fig.colorbar(heat, ax=axes[2], fraction=.045, pad=.02)
        fig.suptitle(f"{category} | {kind} | score={prediction['score']:.4f}, calibration threshold={threshold:.4f}\n"
                     f"8-shot, seed 0 | {Path(sample['path']).name} | display scale fixed 0..0.5 only", fontsize=10)
        filename = figures / f'case_{i+1}_{category}.png'
        fig.savefig(filename, dpi=150)
        plt.close(fig)
        with Image.open(filename) as image:
            thumbs.append(np.array(image.convert('RGB').resize((1300, 350))))
    Image.fromarray(np.concatenate(thumbs, axis=0)).save(figures / 'six_difficult_cases.png')
    save_json(args.result_root / 'difficult_cases.json', case_rows)
    notes_path = args.result_root / 'case_observations.json'
    notes = json.loads(notes_path.read_text(encoding='utf-8'))['cases'] if notes_path.exists() else {}
    lines = [
        '# Part 1 Template 正式实验结果', '',
        '## 完成范围', '',
        '已完成六类 × 1/2/4/8-shot × 三个参考 seed（0/1/2）共 **72 组正式 Template 实验**。'
        '每组使用完整官方测试集；六类共 960 张，跨配置共保存 11,520 张原尺寸预测图。'
        '全部记录为 `integration_only=false`，与旧的 Candle 联调成绩分开。', '',
        '**这份结果只代表 Template 基准完成，不代表 A 的全部工作、ECC、Autoencoder、Part 3 或最终提交完成。**', '',
        '## 方法与评价', '',
        '- 使用冻结清单中的真实 K 张正常参考图及八角度旋转，不增加额外真实训练图。',
        '- 输入短边 448，等比例缩放后从右侧/底部裁到 14 的倍数；以 28×28 RGB 描述子选最近模板。',
        '- RGB 绝对差、Gaussian sigma=2、3×3 灰度闭运算；输入差异图最高 1% 像素的均值作为整图分数。',
        '- 预测图线性插值回原图尺寸；GT 保留原尺寸，正常测试图的全零 GT 也参与像素评价。',
        '- 每类固定 20 张正常校准图，仅用于线性插值的 95% 分位阈值；分数严格大于阈值判 NG。',
        '- 未用测试集调参、选择最佳 seed、逐图 min–max 或更换正式结果。源码哈希见 `frozen_config.json`。',
        '- 三个 seed 反映不同共享参考块；先算每个 seed 的六类等权均值，再报三 seed 均值和总体标准差（ddof=0）。', '',
        '## 六类宏平均', '',
        '指标范围 0–1；AUROC 越高越好，FPR/FNR 越低越好。0.5 是随机排序的 AUROC 参考值。', '',
        '| shot | Image AUROC | Pixel AUROC | Image AP | Pixel AP | AUPRO@0.3 | FPR | FNR | CPU ms/图 |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    for summary in summaries:
        subset = [r for r in records if r['shots'] == summary['shots']]
        cells = [f"{summary[k]['mean']:.4f} ± {summary[k]['std']:.4f}" for k in
                 ['image_auroc', 'pixel_auroc', 'image_ap', 'pixel_ap', 'aupro_03', 'image_fpr', 'image_fnr']]
        milliseconds = np.mean([r['resources']['inference_seconds_mean'] * 1000 for r in subset])
        lines.append('| ' + ' | '.join([str(summary['shots']), *cells, f'{milliseconds:.2f}']) + ' |')
    lines += ['', '![不同 shot 的 AUROC](figures/auroc_by_shots.png)', '', '## 8-shot 各类结果', '',
              '8-shot 是预先指定的展示配置，不表示依据测试成绩选择了最佳配置。其他 shot 见 `summary.csv`。', '',
              '| 类别 | Image AUROC | Pixel AUROC | FPR | FNR |', '|---|---|---|---|---|']
    summary = next(s for s in summaries if s['shots'] == 8)
    for category in CATEGORIES:
        cells = [f"{summary[k]['per_category'][category]['mean']:.4f} ± {summary[k]['per_category'][category]['std']:.4f}"
                 for k in ['image_auroc', 'pixel_auroc', 'image_fpr', 'image_fnr']]
        lines.append('| ' + ' | '.join([category, *cells]) + ' |')
    lines += ['', '## 六个困难案例', '',
              '固定观察 8-shot、seed 0，每类选一个案例；交替优先展示误报和漏报，具体规则见分析程序。'
              '这些是每类较严重的错误示例，并非随机抽样或错误率估计；优先类型不存在时如实展示另一类型。'
              '案例用于解释已经冻结的模型，不回头调参。热图显示范围统一为 0–0.5，超范围仅在画图时截色，'
              '评价仍使用完整原始浮点预测。青色轮廓是原始 GT。', '']
    for i, row in enumerate(case_rows):
        lines += [f"### {i+1}. {row['category']}：{row['error_type']}", '',
                  f"测试图 `{row['path']}`，分数 {row['image_score']:.4f}，正常校准阈值 {row['threshold']:.4f}。"
                  f"GT 缺陷占全图 {row['gt_area_fraction']*100:.3f}%；热图最高 1% 区域中，"
                  f"落入 GT 的比例为 {row['top1_percent_map_gt_fraction']*100:.3f}%。"
                  '该重合率仅辅助理解热图，不是课程的独立评价指标。', '',
                  f"![{row['category']} 困难案例](figures/case_{i+1}_{row['category']}.png)", '']
        note = notes.get(row['path'])
        if note and note['reference'] == row['reference'] and note['angle'] == row['angle']:
            lines += ['人工查看原图、GT、热图与参考后的观察：' + note['observation'], '']
    lines += ['## 耗时、内存与限制', '',
              '当前为四 OpenCV 线程的 CPU 实现，机器和依赖写在每组 `metrics.json` 的 resources 中。'
              '每图耗时包含读 JPG、预处理、选模板、差分、后处理和恢复原尺寸；'
              '不包含参考库构建、预热、校准、保存预测图和指标计算。参考库构建耗时另行保存。', '',
              '**GPU 未使用，GPU 显存记为 0；不能把这解释为 GPU 版本的显存成绩，也不能据此直接宣称比 GPU 上的 DINO/CLIP 更快。**'
              '若最终表格要求所有方法在同一 GPU 上测量，这项硬件可比性仍需补充。进程 RSS 是推理期间采样到的最大值，'
              '不是操作系统捕获到的绝对内存峰值。', '',
              '模板只选整体最接近的参考图及离散旋转，没有 ECC 几何配准。位置、局部姿态、形状、阴影或亮度变化可能形成强差异；'
              '是否构成某个案例的主要原因，还要结合图像逐例判断，不能仅凭低 AUROC 断言。', '',
              '## 给 B 的使用方式', '',
              '- 汇总表使用 `summary.csv` 或 `summary.json`；逐组数据在 `runs/<category>/<shot>shot_seed<seed>/metrics.json`。',
              '- 各测试图分数、阈值、预测与参考选择在同目录的 `image_predictions.csv`，便于对照同一测试图。',
              '- `calibration_scores.csv` 保存真实 20 张校准图分数；`prediction_manifest.json` 保存原始预测清单和热图哈希。独立重算像素指标仍需要本机原始热图。',
              '- 72 组完整性与源码核查记录见 `completion_audit.json`。原始大体积预测在本机 `outputs/part1_template_v1/`，不进入 Git。',
              '- 与 DINO/CLIP 合并报告时分别保留方法、shot、参考 seed/运行 seed、设备和协议标识，不混用旧联调成绩。', '',
              '## 还没有完成的项目', '',
              'ECC 改进、Autoencoder、A 负责的其他 Part 3 实验及最终报告/展示/提交，不在本轮完成范围内。'
              'Template 实验完整和 Template 准确率理想是两件事；低于随机参考水平的结果也照实保留。', '']
    report_path = args.result_root / '实验结果与错误分析.md'
    marker = '## 正常图外观敏感性诊断'
    old = report_path.read_text(encoding='utf-8') if report_path.exists() else ''
    if marker in old:
        lines += [marker + old.split(marker, 1)[1]]
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print('PASS: 72 formal runs,', maps, 'native maps, six difficult cases, shot curves.', flush=True)


if __name__ == '__main__':
    main()
