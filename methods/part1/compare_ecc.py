"""Audit full paired ECC grid, plot fixed examples and report gains or losses."""
from pathlib import Path
import argparse
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

from shared_protocol.protocol import ROOT, CATEGORIES, SHOTS, inside, save_json, sha256
from shared_protocol.data import load_manifest, load_mask
from shared_protocol.metrics import aggregate_runs
from .run_ecc_full import configuration
from .template import read_rgb, rotate_reference, preprocess


def csv_file(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--ecc-root',type=Path,default=ROOT/'outputs/part1_ecc_full_v1')
    parser.add_argument('--baseline-root',type=Path,default=ROOT/'outputs/part1_template_v1')
    parser.add_argument('--result-root',type=Path,default=ROOT/'results/part1_ecc_full')
    args=parser.parse_args()
    records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(args.ecc_root.glob('*/*/metrics.json'))]
    summary=aggregate_runs(records,CATEGORIES)
    config=configuration()
    if any(r['run_config']!=config for r in records):
        raise ValueError('Frozen ECC configuration/source differs')
    ecc={(r['category'],r['shots'],r['seed']):r for r in records};base={};maps=0
    storage_shared=storage_saved=0
    for key,r in ecc.items():
        c,k,s=key;relative=Path(c)/f'{k}shot_seed{s}'
        bf,ef=args.baseline_root/relative,args.ecc_root/relative
        b=json.loads((bf/'metrics.json').read_text(encoding='utf-8'));base[key]=b
        er=json.loads((ef/'run.json').read_text(encoding='utf-8'))
        br=json.loads((bf/'run.json').read_text(encoding='utf-8'))
        manifest=load_manifest(c)
        assert sha256(ef/'run.json')==r['prediction_manifest_sha256']
        assert sha256(bf/'run.json')==r['baseline_prediction_manifest_sha256']==b['prediction_manifest_sha256']
        assert er['reference_paths']==br['reference_paths']==manifest['references'][str(k)][str(s)]
        assert er['calibration_paths']==br['calibration_paths']==manifest['calibration']
        assert [p['path'] for p in er['test']]==[p['path'] for p in br['test']]==[p['path'] for p in manifest['test']]
        assert len(list((ef/'maps').glob('*.npy')))==len(manifest['test'])
        bc={t['path']:t for t in json.loads((bf/'template_choices.json').read_text(encoding='utf-8'))}
        traces=json.loads((ef/'alignment_traces.json').read_text(encoding='utf-8'))
        assert len(traces)==len(manifest['test'])+20
        assert all((t['reference'],t['angle'])==(bc[t['path']]['reference'],bc[t['path']]['angle']) for t in traces)
        storage=r['resources'].get('map_storage_run',{})
        storage_shared+=storage.get('shared_files',0);storage_saved+=storage.get('saved_bytes',0)
        maps+=len(er['test'])
    baseline_summary=aggregate_runs(list(base.values()),CATEGORIES)
    fields=['image_auroc','pixel_auroc','image_ap','pixel_ap','aupro_03','image_fpr','image_fnr']
    table=[]
    for k in SHOTS:
        for category in [*CATEGORIES,'six_category_macro']:
            cats=CATEGORIES if category=='six_category_macro' else [category]
            row=dict(shots=k,category=category)
            for field in fields:
                bv=[np.mean([base[c,k,s][field] for c in cats]) for s in [0,1,2]]
                ev=[np.mean([ecc[c,k,s][field] for c in cats]) for s in [0,1,2]]
                for name,values in [('baseline',bv),('ecc',ev),('paired_delta',np.array(ev)-bv)]:
                    row[f'{field}_{name}_mean']=float(np.mean(values));row[f'{field}_{name}_std']=float(np.std(values))
            for method,index in [('baseline',base),('ecc',ecc)]:
                row[method+'_cpu_ms_mean']=float(np.mean([index[c,k,s]['resources']['inference_seconds_mean']*1000 for c in cats for s in [0,1,2]]))
            selected=[ecc[c,k,s] for c in cats for s in [0,1,2]]
            row['ecc_aligned_test_count']=sum(r['alignment_counts_test'].get('aligned',0) for r in selected)
            row['ecc_fallback_test_count']=sum(r['alignment_counts_test'].get('fallback',0) for r in selected)
            table.append(row)
    csv_file(args.result_root/'paired_comparison.csv',table)
    save_json(args.result_root/'paired_comparison.json',table)
    figures=args.result_root/'figures';figures.mkdir(exist_ok=True)
    macros=[r for r in table if r['category']=='six_category_macro']
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for ax,key,title in zip(axes.flat,['image_auroc','pixel_auroc','image_fnr','cpu_ms'],['Image AUROC','Native pixel AUROC','Calibrated FNR (lower is better)','CPU ms/image']):
        for method in ['baseline','ecc']:
            values=[r[f'{method}_cpu_ms_mean'] if key=='cpu_ms' else r[f'{key}_{method}_mean'] for r in macros]
            errors=None if key=='cpu_ms' else [r[f'{key}_{method}_std'] for r in macros]
            ax.errorbar(SHOTS,values,yerr=errors,marker='o',capsize=3,label=method)
        ax.set(title=title,xlabel='Shots',xticks=SHOTS);ax.grid(alpha=.2);ax.legend()
        if key!='cpu_ms':ax.set_ylim(0,1)
    fig.savefig(figures/'paired_macro_curves.png',dpi=180);plt.close(fig)
    examples=json.loads((ROOT/'results/part1_template/difficult_cases.json').read_text(encoding='utf-8'))
    cases=[]
    for i,case in enumerate(examples):
        c=case['category'];relative=Path(c)/'8shot_seed0';ef=args.ecc_root/relative;bf=args.baseline_root/relative
        sample=next(s for s in load_manifest(c)['test'] if s['path']==case['path'])
        er=json.loads((ef/'run.json').read_text(encoding='utf-8'));br=json.loads((bf/'run.json').read_text(encoding='utf-8'))
        ep=next(p for p in er['test'] if p['path']==case['path']);bp=next(p for p in br['test'] if p['path']==case['path'])
        trace=next(t for t in json.loads((ef/'alignment_traces.json').read_text(encoding='utf-8')) if t['path']==case['path'])
        rgb=read_rgb(inside(args.data_root,case['path']));mask=load_mask(sample,args.data_root)
        bm,em=np.load(bf/bp['map']),np.load(ef/ep['map'])
        ref=preprocess(rotate_reference(read_rgb(inside(args.data_root,trace['reference'])),trace['angle']))
        q=preprocess(rgb)
        if ref.shape!=q.shape:ref=cv2.resize(ref,(q.shape[1],q.shape[0]),interpolation=cv2.INTER_CUBIC)
        aligned=ref.copy()
        if trace['alignment']['status']=='aligned':
            warp=np.array(trace['alignment']['warp_query_to_reference'],np.float32);h,w=ref.shape[:2]
            flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP
            aligned=cv2.warpAffine(ref,warp,(w,h),flags=flags)
            valid=cv2.warpAffine(np.ones((h,w),np.float32),warp,(w,h),flags=flags)>=.999
            aligned[~valid]=ref[~valid]
        bt,et=base[c,8,0]['image_threshold'],ecc[c,8,0]['image_threshold']
        row=dict(category=c,path=case['path'],label=sample['label'],baseline_score=bp['score'],baseline_threshold=bt,
                 baseline_ng=int(bp['score']>bt),ecc_score=ep['score'],ecc_threshold=et,ecc_ng=int(ep['score']>et),
                 alignment_status=trace['alignment']['status'],fallback_reason=trace['alignment']['reason'])
        cases.append(row)
        fig,axes=plt.subplots(1,6,figsize=(17,3.4),constrained_layout=True)
        for ax,values,title in zip(axes,[rgb,mask,bm,em,ref,aligned],['Input','GT','Baseline raw map','ECC raw map','Normal reference','Registered reference']):
            ax.axis('off');ax.set_title(title,fontsize=9)
            if title.endswith('raw map'):
                ax.imshow(values,cmap='inferno',vmin=0,vmax=.5)
                if mask.any():ax.contour(mask,levels=[.5],colors=['cyan'],linewidths=.6)
            elif title=='GT':ax.imshow(values,cmap='gray',vmin=0,vmax=1)
            else:ax.imshow(values)
        fig.suptitle(f"{c} / 8-shot seed 0 / GT={sample['label']} / {Path(case['path']).name}\n"
                     f"Base {bp['score']:.4f} > {bt:.4f}: {row['baseline_ng']} | ECC {ep['score']:.4f} > {et:.4f}: {row['ecc_ng']} | {row['alignment_status']} | map display 0..0.5",fontsize=10)
        fig.savefig(figures/f'case_{i+1}_{c}.png',dpi=150);plt.close(fig)
        # GT is used only to choose an explanatory zoom after inference/evaluation.
        # Normal cases use the baseline heatmap peak; neither ROI changes a score.
        if mask.any():
            yy,xx=np.nonzero(mask);x0,x1=int(xx.min()),int(xx.max())+1;y0,y1=int(yy.min()),int(yy.max())+1
            padx,pady=max(32,(x1-x0)//2),max(32,(y1-y0)//2)
        else:
            cy,cx=np.unravel_index(np.argmax(bm),bm.shape);x0=x1=int(cx);y0=y1=int(cy);padx=pady=100
        x0,x1=max(0,x0-padx),min(rgb.shape[1],x1+padx)
        y0,y1=max(0,y0-pady),min(rgb.shape[0],y1+pady)
        fig,axes=plt.subplots(1,4,figsize=(12,3.2),constrained_layout=True)
        roi_mask=mask[y0:y1,x0:x1]
        for ax,values,title in zip(axes,[rgb[y0:y1,x0:x1],roi_mask,bm[y0:y1,x0:x1],em[y0:y1,x0:x1]],['Input detail','GT detail','Baseline raw map','ECC raw map']):
            ax.axis('off');ax.set_title(title)
            if title.endswith('raw map'):
                ax.imshow(values,cmap='inferno',vmin=0,vmax=.5)
                if roi_mask.any():ax.contour(roi_mask,levels=[.5],colors=['cyan'],linewidths=1.)
            elif title=='GT detail':ax.imshow(values,cmap='gray',vmin=0,vmax=1)
            else:ax.imshow(values)
        fig.suptitle(f'{c}: post-hoc display ROI x={x0}:{x1}, y={y0}:{y1}; raw heatmap scale 0..0.5',fontsize=10)
        fig.savefig(figures/f'case_detail_{i+1}_{c}.png',dpi=150);plt.close(fig)
    csv_file(args.result_root/'fixed_case_comparison.csv',cases)
    panels=[]
    for i,c in enumerate(CATEGORIES):
        with Image.open(figures/f'case_{i+1}_{c}.png') as image:
            panels.append(np.asarray(image.convert('RGB').resize((1700,340))))
    Image.fromarray(np.concatenate(panels,axis=0)).save(figures/'six_case_overview.png')
    save_json(args.result_root/'completion_audit.json',dict(status='pass',formal_runs=72,native_prediction_maps=maps,
        expected_native_prediction_maps=11520,all_sources_match_frozen_config=True,
        paired_reference_rotation_calibration_and_test_paths_identical=True,
        shared_evaluator_checked_native_maps_and_original_masks=True,
        byte_identical_storage_shared_files=storage_shared,storage_duplicate_bytes_avoided=storage_saved,
        analysis_source_sha256=sha256(Path(__file__).resolve()),
        scope='ECC full grid only; does not certify AE, Part3, GPU comparability or final submission'))
    lines=['# ECC 完整 72 组配准对照结果','','## 完成情况','',
           '六类 × 1/2/4/8-shot × 三 seed，共 72 组正式 ECC 对照已完成。逐张核对与基准所选参考及旋转角度一致，校准和测试清单完全一致。'
           '仅新增配准；差异、后处理、评分和原尺寸评价规则保持不变。每个方法用相同 20 张正常图计算各自阈值。'
           '输入短边为 448；ECC 只在最长边 224 的灰度副本上估计变换，再换算到输入尺寸配准参考。测试图和原始 GT 保持不动。', '',
           '均值为六类等权宏平均；标准差来自三个参考 seed（ddof=0）。配对变化的标准差直接由三个 seed 的差值计算，不是两组标准差相减。', '',
           '| shot | 基准 Image AUROC | ECC Image AUROC | 基准 Pixel AUROC | ECC Pixel AUROC | 基准 FNR | ECC FNR | 基准/ECC FPR | 基准/ECC CPU ms |',
           '|---|---|---|---|---|---|---|---|---|']
    for r in macros:
        cols=[str(r['shots'])]+[f"{r[f'{key}_{method}_mean']:.4f} ± {r[f'{key}_{method}_std']:.4f}" for key in ['image_auroc','pixel_auroc'] for method in ['baseline','ecc']]
        cols += [f"{r[f'image_fnr_{m}_mean']:.4f}" for m in ['baseline','ecc']]+[f"{r['image_fpr_baseline_mean']:.4f} / {r['image_fpr_ecc_mean']:.4f}", f"{r['baseline_cpu_ms_mean']:.2f} / {r['ecc_cpu_ms_mean']:.2f}"]
        lines.append('| '+' | '.join(cols)+' |')
    lines += ['', '![宏平均对照](figures/paired_macro_curves.png)', '', '## 配准状态', '',
              '| shot | 测试预测总次数 | 接受配准 | 回退 |', '|---|---|---|---|']
    for r in macros:
        a,b=r['ecc_aligned_test_count'],r['ecc_fallback_test_count'];lines.append(f"| {r['shots']} | {a+b} | {a} | {b} |")
    lines += ['', '回退图仍进入全部指标，未删除困难样本。每图回退原因、有效范围、相关性和矩阵保存在各组 alignment_traces.json。'
              '`aligned` 只表示满足预设的几何范围、重叠和相关性条件，不等于与真实物体精确对应，也不等于检测正确。', '',
              '## 沿用基准的六个困难案例','',
              '样本沿用基准已选好的 8-shot seed 0 六例，没有根据 ECC 成绩改选好看的案例。两张热图共用 0–0.5 显示范围，指标仍用完整原始浮点值；青色为 GT。1 表示判 NG，0 表示判正常。', '']
    lines += ['![六类配准前后总览](figures/six_case_overview.png)','']
    lines += ['局部放大图仅用于事后观察：异常图取 GT 包围框并扩展上下文，正常图取基准热图峰值附近。区域不用于推理、参数选择或修改任何评价分数。','']
    for i,row in enumerate(cases):
        verdict='纠正了这例判断' if row['baseline_ng']!=row['label'] and row['ecc_ng']==row['label'] else ('这例仍判断错误' if row['ecc_ng']!=row['label'] else '这例判断正确')
        lines += [f"### {row['category']}：{verdict}",'',
                  f"GT={row['label']}；基准判定={row['baseline_ng']}，ECC 判定={row['ecc_ng']}；配准状态={row['alignment_status']}。",'',
                  f"![{row['category']} 配准对照](figures/case_{i+1}_{row['category']}.png)",'']
        lines += [f"![{row['category']} 局部放大](figures/case_detail_{i+1}_{row['category']}.png)",'']
    lines += ['## 限制与文件入口','',
              '完整性通过不代表每类效果均提升，请同时查看各类和每个 shot 的 paired_comparison.csv；AP、AUPRO、FPR/FNR 及配对变化均已保留。'
              'ECC 是预先列入分工的选做方向，本轮没有按测试成绩修改配准参数。使用相同公开测试集进行方法对照，不声称这是对全新未见数据的验证。','',
              '采用 CPU 四个 OpenCV 线程。计时包含读图、预处理、模板选择、ECC 与检查、后处理和原尺寸恢复，不含保存及指标评价。'
              'GPU 未使用，显存记录 0；与 GPU 神经模型仍存在设备可比性限制。全部变换与失败策略见 methods/part1/ECC对照方案.md。','',
              '轻量分数、校准分数、完整预测清单和配准日志在 runs/；原始预测在本机 outputs/part1_ecc_full_v1/。'
              'AE、Part 3 C/F 和最终报告/提交仍需另行完成。','']
    lines += ['## 原始预测存储','',
              f'11,520 个原始预测路径全部保留，其中 {storage_shared:,} 次写出使用字节完全相同的硬链接共享，累计避免约 {storage_saved/1024**3:.2f} GiB 重复存储。'
              '没有压缩精度或缩小标注；逐图共享发生在推理计时之外。逻辑文件大小按路径相加仍可能很大，磁盘实际占用因共享而减少。','']
    if (args.result_root/'案例核查与结论.md').is_file():
        lines += ['逐例人工观察、改善与退步的解释，以及 A 的剩余任务见 [案例核查与大白话结论](案例核查与结论.md)。','']
    (args.result_root/'ECC结果与基准对照.md').write_text('\n'.join(lines),encoding='utf-8')
    print('PASS: 72 paired ECC cells,',maps,'native maps, source and reference checks, CSV/report/figures.',flush=True)


if __name__=='__main__':
    main()
