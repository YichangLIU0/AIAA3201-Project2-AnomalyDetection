"""Audit all18 AE cells and visualize the six already-frozen difficult cases."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from shared_protocol.data import load_manifest,load_mask
from shared_protocol.protocol import CATEGORIES,inside,save_json,sha256
from shared_protocol.metrics import image_statistics
from methods.normal_training import validate_training
from .run_ae import summarize
from .ae import CompactAE,tensor_image
from .template import read_rgb,preprocess

ROOT=Path(__file__).resolve().parents[2]
CASES=['candle/test/bad/026.JPG','capsules/test/bad/052.JPG','cashew/test/good/344.JPG',
       'chewinggum/test/bad/030.JPG','fryum/test/good/392.JPG','pipe_fryum/test/bad/060.JPG']


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/part1_ae_v1')
    p.add_argument('--results',type=Path,default=ROOT/'results/part1_ae')
    a=p.parse_args();config=json.loads((a.output/'frozen_config.json').read_text(encoding='utf-8'))
    benchmark=json.loads((a.results/'serial_benchmark.json').read_text(encoding='utf-8'))
    if benchmark['n_cells']!=18 or benchmark['test_predictions_recomputed']!=2880:
        raise ValueError('Missing complete serial timing benchmark')
    for name,digest in config['algorithm_source_sha256'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Frozen algorithm source changed')
    records=[];count=0;compressed=0;float_bytes=0
    for category in CATEGORIES:
        manifest=load_manifest(category)
        for seed in range(3):
            folder=a.output/category/f'seed{seed}';compact=a.results/'runs'/category/f'seed{seed}'
            raw=json.loads((folder/'run.json').read_text(encoding='utf-8'))
            metrics=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
            validate_training(manifest,raw['training_paths'])
            if raw['run_config']!=config or raw['integration_only'] or metrics['status']!='success':
                raise ValueError('Nonformal/mixed AE cells')
            if metrics['run_manifest_sha256']!=sha256(folder/'run.json') or sha256(compact/'run.json')!=sha256(folder/'run.json'):
                raise ValueError('Raw/compact prediction log changed')
            if sha256(folder/'model.pt')!=raw['checkpoint_sha256']:raise ValueError('Checkpoint changed')
            fresh=image_statistics([r['score'] for r in raw['test']], [r['score'] for r in raw['calibration']],
                                   [r['label'] for r in manifest['test']])
            if any(not np.isclose(fresh[k],metrics[k],rtol=0,atol=1e-12) for k in
                   ['image_auroc','image_ap','image_fpr','image_fnr','image_threshold']):
                raise ValueError('Independent image-metric recomputation failed')
            if raw['training_count']!=len(manifest['normal_training_pool']):raise ValueError('Wrong training count')
            if len(raw['resources']['training_loss_by_epoch'])!=12:raise ValueError('Incomplete training')
            with (compact/'calibration_scores.csv').open('w',newline='',encoding='utf-8-sig') as stream:
                writer=csv.DictWriter(stream,fieldnames=['path','score']);writer.writeheader()
                writer.writerows(raw['calibration'])
            for prediction,sample in zip(raw['test'],manifest['test']):
                path=folder/prediction['map']
                if sha256(path)!=prediction['map_sha256'] or prediction['native_shape']!=[sample['height'],sample['width']]:
                    raise ValueError('Saved native prediction changed')
                compressed+=path.stat().st_size;float_bytes+=sample['height']*sample['width']*4;count+=1
            weights=a.results/'checkpoints'/category/f'seed{seed}.pt';weights.parent.mkdir(parents=True,exist_ok=True)
            if weights.exists() and sha256(weights)!=raw['checkpoint_sha256']:raise ValueError('Published weight would be replaced')
            if not weights.exists():shutil.copyfile(folder/'model.pt',weights)
            records.append(metrics)
    summary=summarize(records)
    existing=json.loads((a.results/'summary.json').read_text(encoding='utf-8'))
    if summary!=existing:raise ValueError('Independent grid summary differs')
    reported={**summary,'raw_parallel_inference_ms_mean':summary['inference_ms_mean'],
              'inference_ms_mean':benchmark['inference_ms_class_seed_mean'],
              'peak_gpu_allocated_mb_max':benchmark['peak_gpu_allocated_mb_max'],
              'timing_source':'serial_benchmark.json; raw producer summary.json retains initial parallel wall times'}
    save_json(a.results/'reported_summary.json',reported)
    save_json(a.results/'completion_audit.json',{'status':'complete','n_runs':18,'test_predictions':count,
        'calibration_per_run':20,'training_pool_excludes_calibration_and_test':True,
        'source_hashes_match_frozen_config':True,'checkpoint_hashes_verified':True,
        'all_native_map_file_hashes_verified':True,'image_metrics_and_macro_summary_recomputed':True,
        'published_manifests_identical_to_actual_runs':True,'serial_benchmark_all_scores_exactly_reproduced':True,
        'stored_prediction_bytes':compressed,'uncompressed_float32_pixel_bytes':float_bytes,
        'lossless_compression_ratio':compressed/float_bytes,'free_disk_bytes':shutil.disk_usage(a.output).free,
        'shared_protocol_unchanged':True,'comparison_mode':'normal_training; no K-shot claim'})
    figures=a.results/'figures';figures.mkdir(exist_ok=True)
    # Diagnose optimization using training normals only. Never remove or replace a
    # seed because of either this diagnostic or its observed test performance.
    torch.set_num_threads(4)
    training_quality=[];worst=None
    for category in CATEGORIES:
        manifest=load_manifest(category);names=manifest['normal_training_pool'][:8]
        images=[preprocess(read_rgb(inside(a.data_root,p))) for p in names]
        for seed in range(3):
            payload=torch.load(a.output/category/f'seed{seed}/model.pt',weights_only=True,map_location='cpu')
            model=CompactAE().eval();model.load_state_dict(payload['state_dict'])
            mse=[];black=[];fractions=[];preview=None
            for image in images:
                x=tensor_image(image,'cpu')
                with torch.inference_mode():reconstructed=model(x)
                mse.append(float((reconstructed-x).square().mean()))
                black.append(float(x.square().mean()))
                fractions.append(float((reconstructed<1e-4).float().mean()))
                if preview is None:preview=reconstructed[0].permute(1,2,0).numpy()
            ratio=float(np.mean(mse)/np.mean(black))
            collapsed=ratio>=.95 and float(np.mean(fractions))>=.99
            row={'category':category,'seed':seed,'probe_paths':names,'probe_device':'CPU posthoc diagnostic',
                 'final_model_probe_mse':float(np.mean(mse)),'all_black_predictor_probe_mse':float(np.mean(black)),
                 'mse_ratio_to_black_predictor':ratio,'fraction_reconstruction_values_below_1e_minus4':float(np.mean(fractions)),
                 'black_collapse_flag':bool(collapsed),'kept_in_all_test_summaries':True}
            training_quality.append(row)
            if worst is None or row['fraction_reconstruction_values_below_1e_minus4']>worst['fraction']:
                worst={'fraction':row['fraction_reconstruction_values_below_1e_minus4'],
                       'image':images[0],'category':category,'seed':seed}
    save_json(a.results/'training_quality_audit.json',{'selection':'first8 normal training-pool images per class, never calibration/test',
        'diagnostic_rule':'posthoc flag if normal-probe MSE>=95% of zero predictor and >=99% reconstruction values<1e-4',
        'used_to_select_or_replace_models':False,'runs':training_quality})
    fig,axes=plt.subplots(1,4,figsize=(13,4),layout='constrained')
    axes[0].imshow(worst['image']);axes[0].set_title(worst['category']+' normal TRAIN input')
    for seed,ax in enumerate(axes[1:]):
        payload=torch.load(a.output/worst['category']/f'seed{seed}/model.pt',weights_only=True,map_location='cpu')
        model=CompactAE().eval();model.load_state_dict(payload['state_dict'])
        with torch.inference_mode():reconstructed=model(tensor_image(worst['image'],'cpu'))[0].permute(1,2,0).numpy()
        ax.imshow(np.clip(reconstructed,0,1));ax.set_title(f'seed{seed} reconstruction')
    for ax in axes:ax.axis('off')
    fig.suptitle('Training-only diagnostic; example selected by largest near-zero reconstruction fraction')
    fig.savefig(figures/'training_quality.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for ax,category in zip(axes.ravel(),CATEGORIES):
        for r in [r for r in records if r['category']==category]:
            ax.plot(range(1,13),r['resources']['training_loss_by_epoch'],label=f"seed{r['seed']}")
        ax.set(title=category,xlabel='Epoch',ylabel='Normal training MSE',yscale='log');ax.grid(alpha=.2)
    axes[0,0].legend();fig.savefig(figures/'training_loss.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for ax,category in zip(axes.ravel(),CATEGORIES):
        manifest=load_manifest(category)
        raw=json.loads((a.output/category/'seed0/run.json').read_text(encoding='utf-8'))
        metric=next(r for r in records if r['category']==category and r['seed']==0)
        scores=np.array([r['score'] for r in raw['test']]);labels=np.array([r['label'] for r in manifest['test']])
        calibration=np.array([r['score'] for r in raw['calibration']])
        edges=np.linspace(min(scores.min(),calibration.min()),max(scores.max(),calibration.max())+1e-12,31)
        for label,title,color in [(0,'normal test','tab:blue'),(1,'anomaly test','tab:orange')]:
            ax.hist(scores[labels==label],bins=edges,alpha=.5,density=True,label=title,color=color)
        ax.plot(calibration,np.zeros(20),'|',color='black',label='20 normal calibration')
        ax.axvline(metric['image_threshold'],color='red',linestyle='--',label='calibration95% threshold')
        ax.set(title=category,xlabel='Raw image score',ylabel='Density');ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=7)
    fig.suptitle('AE seed0: score distributions for interpretation only; threshold never fitted to test labels')
    fig.savefig(figures/'score_distributions.png',dpi=140);plt.close(fig)
    observations=[]
    extra_case=next(r['path'] for r in load_manifest('cashew')['test'] if r['label']==1)
    for name in [*CASES,extra_case]:
        category=name.split('/')[0];manifest=load_manifest(category)
        index=next(i for i,r in enumerate(manifest['test']) if r['path']==name);sample=manifest['test'][index]
        original=read_rgb(inside(a.data_root,name));mask=load_mask(sample,a.data_root)
        variants=[]
        for method,rawroot,cell in [('Template','part1_template_v1','8shot_seed0'),('ECC','part1_ecc_full_v1','8shot_seed0'),('AE','part1_ae_v1','seed0')]:
            folder=ROOT/'outputs'/rawroot/category/cell
            raw=json.loads((folder/'run.json').read_text(encoding='utf-8'));pred=raw['test'][index]
            metrics=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
            if pred['path']!=name:raise ValueError('Case index mismatch')
            path=folder/pred['map']
            if path.suffix=='.npz':
                with np.load(path) as archive:values=archive['anomaly_map']
            else:values=np.load(path,allow_pickle=False)
            decision=int(pred['score']>metrics['image_threshold'])
            variants.append({'method':method,'score':pred['score'],'threshold':metrics['image_threshold'],
                             'predicted_ng':decision,'correct':decision==sample['label'],'values':values,
                             'gt_map_mean':float(values[mask].mean()) if mask.any() else None,
                             'non_gt_map_mean':float(values[~mask].mean())})
        # Qualitative reconstruction uses CPU, after formal inference has finished.
        payload=torch.load(a.output/category/'seed0/model.pt',weights_only=True,map_location='cpu')
        model=CompactAE().eval();model.load_state_dict(payload['state_dict']);image=preprocess(original)
        with torch.inference_mode():reconstruction=model(tensor_image(image,'cpu'))[0].permute(1,2,0).numpy()
        reconstructed=cv2.resize(np.clip(reconstruction,0,1),(original.shape[1],original.shape[0]),interpolation=cv2.INTER_LINEAR)
        if mask.any():
            ys,xs=np.where(mask);pad=max(20,int(max(xs.max()-xs.min(),ys.max()-ys.min())*.4))
            y0=max(0,int(ys.min())-pad);y1=min(len(mask),int(ys.max())+pad+1)
            x0=max(0,int(xs.min())-pad);x1=min(mask.shape[1],int(xs.max())+pad+1)
        else:
            # Normal case: show the detector's strongest baseline response, without a fictitious GT defect.
            y,x=np.unravel_index(np.argmax(variants[0]['values']),mask.shape)
            pad=130;y0=max(0,y-pad);y1=min(mask.shape[0],y+pad);x0=max(0,x-pad);x1=min(mask.shape[1],x+pad)
        stem=category+'_'+Path(name).stem
        for detail in [False,True]:
            crop=(slice(y0,y1),slice(x0,x1)) if detail else (slice(None),slice(None))
            fig,axes=plt.subplots(1,6,figsize=(19,4),layout='constrained')
            axes[0].imshow(original[crop]);axes[0].set_title('Input')
            axes[1].imshow(mask[crop],cmap='gray',vmin=0,vmax=1);axes[1].set_title('GT (native)')
            for ax,v in zip(axes[2:5],variants):
                ax.imshow(original[crop]);ax.imshow(v['values'][crop],cmap='turbo',vmin=0,vmax=.25,alpha=.65)
                ax.set_title(f"{v['method']} {'NG' if v['predicted_ng'] else 'OK'}\ns={v['score']:.4f}; t={v['threshold']:.4f}")
            axes[5].imshow(reconstructed[crop]);axes[5].set_title('AE reconstruction')
            for ax in axes:ax.axis('off')
            fig.suptitle(name+' | display-only common heatmap range0..0.25; '+('post-hoc ROI' if detail else 'full image'))
            fig.savefig(figures/f'{stem}{"_detail" if detail else ""}.png',dpi=140);plt.close(fig)
        observations.append({'path':name,'label':sample['label'],'gt_fraction':float(mask.mean()),
            'variants':[{k:v for k,v in row.items() if k!='values'} for row in variants],
            'figure_stem':stem,'detail_roi_xyxy':[int(v) for v in [x0,y0,x1,y1]],
            'selection':'same six cases frozen in original Template analysis' if name in CASES else
                        'posthoc localization diagnostic: first official Cashew anomaly in manifest order, not selected by individual AE score'})
    save_json(a.results/'case_observations.json',observations)
    fields=['category','training_count']+[f'{k}_{s}' for k in ['image_auroc','pixel_auroc','image_ap','pixel_ap','aupro_03','image_fpr','image_fnr'] for s in ['mean','std']]
    with (a.results/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for category,item in summary['per_category'].items():
            row={'category':category,'training_count':item['training_count']}
            row.update({f'{k}_{s}':v[s] for k,v in item.items() if isinstance(v,dict) for s in ['mean','std']});writer.writerow(row)
    text=['# AE 正式结果与完成边界','',
        '已完成六类 × 三个训练 seed，共18组，每组使用该类完整正常训练池，20张校准图全部排除。共2880张原尺寸测试预测；不缩小GT、不逐图归一化。',
        'AE 是正常图训练扩展，不是1/2/4/8-shot。不得把它与 K-shot 方法称为相同训练预算的实验。三个 seed 仅改变训练初始化、增强和顺序。','',
        '|类别|正常训练图数/模型|Image AUROC|Pixel AUROC|误报率|漏报率|','|---|---:|---:|---:|---:|---:|']
    for category,item in summary['per_category'].items():
        vals=[f"{item[k]['mean']:.4f} ± {item[k]['std']:.4f}" for k in ['image_auroc','pixel_auroc']]
        text.append(f"|{category}|{item['training_count']}|{vals[0]}|{vals[1]}|{item['image_fpr']['mean']:.2%}|{item['image_fnr']['mean']:.2%}|")
    m=summary['macro']
    text.extend(['',f"六类等权 Image AUROC **{m['image_auroc']['mean']:.4f} ± {m['image_auroc']['std']:.4f}**；Pixel AUROC **{m['pixel_auroc']['mean']:.4f} ± {m['pixel_auroc']['std']:.4f}**。先按 seed 对六类等权平均，再取三个 seed 的均值/总体标准差。",
        f"校准阈值下误报率 {m['image_fpr']['mean']:.2%}，漏报率 {m['image_fnr']['mean']:.2%}。95%校准分位数不保证真实测试误报恰好5%。",'',
        '下图为固定seed0的原始图像分数分布，黑色短线是20张正常校准图，红线仅由校准图确定。图用于事后解释，不用测试直方图重新挑阈值。',
        '![分数与校准阈值](figures/score_distributions.png)','',
        '## 效率与保存','',
        f"RTX4060 Laptop GPU 串行复测：**{benchmark['inference_ms_class_seed_mean']:.2f} ms/图**，推理峰值 allocated GPU **{benchmark['peak_gpu_allocated_mb_max']:.2f} MiB**。包括读图、预处理和恢复原尺寸，排除训练、压缩存盘和评价。18组全部测试图重新推理，图像分数与正式保存值逐个完全一致。",
        '正式运行中存在两个独立类别 worker 并行，因此 raw run.json 中的首次墙钟耗时受竞争影响；报告速度以 serial_benchmark.json 为准。Template/ECC 是CPU运行，不能把这种硬件差异当成算法速度优势。',
        '程序原始 summary.json 保留运行时的初次耗时，便于审计；对外使用 reported_summary.json，其中耗时已明确引用串行复测，精度与原汇总完全相同。',
        f"原始预测 float32 像素内容 {float_bytes/1024**3:.2f} GiB，无损 NPZ 实际 {compressed/1024**3:.2f} GiB；没有删除已有结果。模型18份、每份约0.4MB，已提供在 checkpoints/，与原始模型 SHA256 一致。",'',
        '## 学到了什么、仍缺什么','',
        '重建模型用正常图学习局部颜色和纹理。正常训练 MSE 下降只能说明重建拟合，并不证明它能把真实异常和正常外观变化分开。压缩瓶颈可能抹去小缺陷，也可能同时平滑正常纹理；需要结合逐图分数、阈值与 GT 看实际结果。',
        '额外用每类固定前8张正常训练图核查最终模型，并比较直接输出全黑的 MSE。这个检查只诊断训练质量，不用于删除 seed 或替换模型；完整数值见 training_quality_audit.json。',
        f"发现 {sum(r['black_collapse_flag'] for r in training_quality)} 组接近全黑重建的训练退化，仍全部纳入均值/标准差。这些运行的高 Pixel AUROC 也不一定代表细小缺陷识别好：整件亮产品对黑色重建都有误差，可能把产品前景当作异常。应同时看重建图和 AP/AUPRO，不能只挑高AUROC宣称成功。",
        '后续改进应预先固定正常训练池内部的验证划分，研究较小学习率、避免 sigmoid 饱和的输出/损失或更稳定的初始化，再作为新的独立实验；不能根据这次测试成绩只重跑差的 seed。',
        '![训练图上的种子差异](figures/training_quality.png)',
        '![正常训练损失](figures/training_loss.png)','',
        '下面复用原 Template 已固定的六个困难案例，seed0；另补官方顺序首张Cashew异常图，作为事后定位诊断，不根据它的AE成绩挑选。Template/ECC 是8-shot，AE 使用完整正常池。热图仅展示统一0..0.25色标并可视化截断，计算指标时始终使用未归一化原值。局部 ROI 在评价后根据 GT 或正常图的基准最大响应确定，未用于模型训练或调参。'])
    for item in observations:
        category=item['path'].split('/')[0];v=item['variants'][-1];stem=item['figure_stem']
        text.extend(['',f"### {item['path']}",'',
            f"真实标签 {'异常' if item['label'] else '正常'}；AE {'NG' if v['predicted_ng'] else 'OK'}，分数 {v['score']:.5f} / 阈值 {v['threshold']:.5f}，{'判断正确' if v['correct'] else '仍判断错误'}。这是固定困难例的个案，不是总体成功率。",
            (f"GT区域平均误差 {v['gt_map_mean']:.5f}，其余像素平均误差 {v['non_gt_map_mean']:.5f}；其余像素包括正常产品与背景，不能把这个数当独立背景因果实验。" if v['gt_map_mean'] is not None else '正常图没有GT缺陷；热图高响应应结合正常外形、纹理和光照分析，不虚构缺陷。'),
            f'![全图](figures/{stem}.png)',f'![局部](figures/{stem}_detail.png)'])
    text.extend(['','## 与分工的关系','',
        'A 的 Part1 Template、ECC 和本 AE 扩展均有正式六类实验。Part3 C 的生成器/评分头准备不等于训练完成；F 在这三个检测器上的计算也不等于 DINO+REVIEW。B 的新版 DINO 接口和正式结果未到位前，不能宣布 A 的所有 Part3 已完成。'])
    if (a.results/'人工案例核查.md').exists():
        text.extend(['', '人工查看后的逐例说明见 [人工案例核查](人工案例核查.md)，包括正常误报、异常漏报与整图判断正确但定位偏离的情况。'])
    (a.results/'AE正式结果与分析.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('AE_AUDIT_AND_REPORT_COMPLETE',count,flush=True)


if __name__=='__main__':main()
