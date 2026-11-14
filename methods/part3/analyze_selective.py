"""Readable negative/positive selective results, curves and direct count checks."""
import json
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from shared_protocol.protocol import CATEGORIES,save_json,sha256
from .run_selective import read_scores
from .selective import ReviewPolicy,counts

ROOT=Path(__file__).resolve().parents[2]
CASES=['candle/test/bad/026.JPG','capsules/test/bad/052.JPG','cashew/test/good/344.JPG',
       'chewinggum/test/bad/030.JPG','fryum/test/good/392.JPG','pipe_fryum/test/bad/060.JPG']


def case_report(name,rows,figures):
    raw_root={'template':'part1_template_v1','ecc':'part1_ecc_full_v1','ae':'part1_ae_v1'}[name]
    data=ROOT.parent/'data/VisA_pytorch/1cls'
    fig,axes=plt.subplots(6,3,figsize=(12,19),layout='constrained');notes=[]
    from shared_protocol.data import load_mask
    for path,rowaxes in zip(CASES,axes):
        category=path.split('/')[0]
        row=next(r for r in rows if r['category']==category and r['seed']==0 and r['shots']==(None if name=='ae' else 8))
        raw,manifest,calibration,_=read_scores((ROOT/row['source_manifest']).parent)
        index=next(i for i,r in enumerate(manifest['test']) if r['path']==path)
        sample=manifest['test'][index];pred=raw['test'][index]
        policy=ReviewPolicy.fit(calibration);decision=policy.decisions([pred['score']])[0]
        base='NG' if pred['score']>policy.threshold else 'OK'
        source=ROOT/'outputs'/raw_root/category/('seed0' if name=='ae' else '8shot_seed0')/pred['map']
        if source.suffix=='.npz':
            with np.load(source) as archive:heat=archive['anomaly_map']
        else:heat=np.load(source,allow_pickle=False)
        with Image.open(data/path) as image:rgb=np.array(image.convert('RGB'))
        mask=load_mask(sample,data)
        rowaxes[0].imshow(rgb);rowaxes[0].set_title(path,fontsize=9)
        rowaxes[1].imshow(mask,cmap='gray',vmin=0,vmax=1);rowaxes[1].set_title('GT: '+('anomaly' if sample['label'] else 'normal'))
        rowaxes[2].imshow(rgb);rowaxes[2].imshow(heat,cmap='turbo',vmin=0,vmax=.25,alpha=.65)
        rowaxes[2].set_title(f"{base} -> {decision}; score={pred['score']:.4f}\nthreshold={policy.threshold:.4f}; width={policy.width:.4f}",fontsize=9)
        for ax in rowaxes:ax.axis('off')
        notes.append({'path':path,'label':sample['label'],'baseline_decision':base,'review_decision':str(decision),
                      'score':pred['score'],'threshold':policy.threshold,'width':policy.width})
    fig.suptitle(f'{name}: same6 frozen difficult cases; REVIEW is manual inspection, not a true positive')
    fig.savefig(figures/'case_decisions.png',dpi=110);plt.close(fig)
    return notes


def main():
    available=[]
    for name in ['template','ecc','ae']:
        folder=ROOT/f'results/part3_f_{name}'
        if not (folder/'summary.json').exists():continue
        summaries=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        shots=None if name=='ae' else 8
        summary=next(r for r in summaries if r['shots']==shots)
        rows=[json.loads(p.read_text(encoding='utf-8')) for p in folder.glob('runs/*/*/selective.json')]
        for row in rows:
            origin=ROOT/row['source_manifest']
            if sha256(origin)!=row['source_manifest_sha256']:raise ValueError('Selective source changed')
            raw,manifest,calibration,_=read_scores(origin.parent)
            policy=ReviewPolicy.fit(calibration)
            scores=np.array([r['score'] for r in raw['test']]);labels=np.array([r['label'] for r in manifest['test']])
            if policy.threshold!=row['policy']['threshold'] or policy.width!=row['policy']['width']:
                raise ValueError('Non-calibration policy parameters')
            for mode in ['none','symmetric','lower_only']:
                fresh=counts(scores,labels,policy,policy.decisions(scores,mode)!='REVIEW')
                if fresh!=row['policies'][mode]:raise ValueError('Count accounting mismatch')
                if fresh['automatic_fn']+fresh['automatic_tp']+fresh['review_anomaly']!=int(labels.sum()):
                    raise ValueError('REVIEW miscounted as a true positive')
            with (origin.parent/'metrics.json').open(encoding='utf-8') as stream:baseline=json.load(stream)
            if not np.isclose(baseline['image_fnr'],row['policies']['none']['automatic_fnr']):
                raise ValueError('No-review does not reproduce original detector FNR')
            # Light export must preserve the actual prediction log byte for byte.
            raw_root={'template':'part1_template_v1','ecc':'part1_ecc_full_v1','ae':'part1_ae_v1'}[name]
            cell=origin.parent.name
            actual=ROOT/'outputs'/raw_root/row['category']/cell/'run.json'
            if not actual.exists() or sha256(actual)!=row['source_manifest_sha256']:
                raise ValueError('Light prediction manifest is not the actual raw run')
        save_json(folder/'independent_count_audit.json',{'checked_cells':len(rows),
            'raw_run_and_light_export_exactly_match':True,'policy_recomputed_from_calibration_only':True,
            'no_review_reproduces_original_fnr':True,'automatic_and_review_anomalies_exhaust_all_anomalies':True})
        figures=folder/'figures';figures.mkdir(exist_ok=True)
        selected=[r for r in rows if r['shots']==shots]
        fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
        for ax,category in zip(axes.ravel(),CATEGORIES):
            for row in [r for r in selected if r['category']==category]:
                curve=row['risk_coverage'][1:]
                ax.step([r['coverage'] for r in curve],[r['risk'] for r in curve],where='pre',label=f"seed{row['seed']}",alpha=.8)
                main=row['policies']['symmetric']
                if main['risk'] is not None:ax.scatter(main['coverage'],main['risk'],s=18)
            ax.set(title=category,xlabel='Automatic coverage',ylabel='Error / automatic decisions',xlim=(0,1),ylim=(0,1))
            ax.grid(alpha=.2)
        axes[0,0].legend();fig.suptitle(f'{name}: tied-confidence risk-coverage; dots = normal-calibrated policy')
        fig.savefig(figures/'risk_coverage.png',dpi=140);plt.close(fig)
        table=[]
        for category in [*CATEGORIES,'six_category_macro']:
            item=summary['per_category'][category]
            row={'category':category}
            for mode in ['none','symmetric','lower_only']:
                for key in ['coverage','risk','automatic_fnr','automatic_fpr']:
                    row[f'{mode}_{key}']=item[mode][key]['mean']
            row['fixed_fpr_test_roc_fnr']=item['fixed_fpr_fnr']['mean'];table.append(row)
        with (folder/'main_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
        macro=summary['per_category']['six_category_macro']
        cases=case_report(name,rows,figures)
        save_json(folder/'case_decisions.json',cases)
        def number(value):return '未定义' if value is None else f'{value:.2%}'
        text=[f'# {name} 的选择性预测：真实结果与限制','',
            f'主展示设置：{"正常池训练 AE" if shots is None else "8-shot"}，六类等权，三个 seed；该方法完整的所有 shot 分析另见 summary.json。',
            '本结果是在现有检测器上加复核策略，不是 DINO 实验。原分数、热图、Image/Pixel AUROC 均不变。','',
            '|策略|自动处理比例|自动判断错误率|自动漏检/全部异常|自动误报/全部正常|',
            '|---|---:|---:|---:|---:|']
        for mode,title in [('none','不复核'),('symmetric','主方案：阈值两侧复核'),('lower_only','消融：只复核阈值下侧')]:
            row=macro[mode]
            text.append('|'+title+'|'+'|'.join(number(row[k]['mean']) for k in ['coverage','risk','automatic_fnr','automatic_fpr'])+'|')
        before=macro['none']['risk']['mean'];after=macro['symmetric']['risk']['mean']
        direction='上升' if after is not None and after>before else '下降'
        text.extend(['',f'主方案自动判断错误率{direction}；变化为 {(after-before)*100:+.3f} 个百分点。减少自动漏检数不代表模型学会了缺陷，一部分图只是转交人工。',
            f'测试 ROC 上实际可达到 FPR≤5% 的 FNR，六类三 seed 均值为 {macro["fixed_fpr_fnr"]["mean"]:.2%}。这是使用测试标签计算的评价点，不是部署阈值。实际部署仍用20张正常校准图的95%分位数，测试 FPR 可以超过5%。','',
            '## 为什么复核不一定提高可靠性','',
            '该规则认为“靠近阈值”才不确定。但阈值由少量正常图估计，若异常图的分数远低于阈值，模型会很有把握地判错，并且不会送复核。正常校准图不能证明对真实异常的置信度可靠。扩大复核比例也不是免费收益，必须报告人工负担和剩余自动风险。',
            '同覆盖率随机复核控制、完整逐图 REVIEW 决策、所有 risk–coverage 点及 AURC 见 runs/。每组随机控制固定种子，重复100次；不依赖标签挑样本。','',
            'AURC 使用按置信度整组接收同分样本的右阶梯面积；零覆盖时 risk 未定义，记录 null。每类先评价再等权，不把测试数量多的类别加权更大。',
            '部署阈值与复核宽度均在测试标签评价前从校准分数冻结。没有根据本次风险曲线重新选参数。','',
            '![各类风险覆盖曲线](figures/risk_coverage.png)','',
            '## 固定困难案例：哪些送复核、哪些仍自动判错','',
            '复用原Template预先固定的六例，seed0；下面只展示实际策略输出，不把送复核说成正确检出。图中热图继承原检测器，仅可视化色标0..0.25，F没有改热图。',
            '![六例复核前后](figures/case_decisions.png)','',
            *[f"- `{c['path']}`：真实{'异常' if c['label'] else '正常'}，原判 {c['baseline_decision']}，现在 {c['review_decision']}。" for c in cases],
            '',
            '## 完成边界','',
            '这个检测器上的真实 F 计算、消融和数量核对已完成。需要 B 的新版 DINO 完整逐图/校准分数后，才能按分工完成未改 DINO 与 DINO+REVIEW 的正式比较。'])
        (folder/'选择性预测结果与限制.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
        available.append({'method':name,'cells':len(rows),'coverage':macro['symmetric']['coverage']['mean'],
                          'risk_before':before,'risk_after':after})
    print('SELECTIVE_REPORTS',available,flush=True)


if __name__=='__main__':main()
