"""Reuse exact detector scores; do not duplicate heatmaps or change detectors."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from shared_protocol.data import load_manifest
from shared_protocol.protocol import CATEGORIES, PROTOCOL_ID, save_json, sha256
from .selective import CONFIG, ReviewPolicy, evaluate_selective

ROOT=Path(__file__).resolve().parents[2]


def read_scores(folder):
    manifest_path=folder/'run.json'
    if not manifest_path.exists(): manifest_path=folder/'prediction_manifest.json'
    raw=json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest=load_manifest(raw['category'])
    if raw['protocol_id']!=PROTOCOL_ID or raw.get('integration_only',True):
        raise ValueError('Only formal frozen shared-protocol predictions are accepted')
    if [r['path'] for r in raw['test']]!=[r['path'] for r in manifest['test']]:
        raise ValueError('Test list is incomplete or different')
    if raw.get('experiment_mode')=='normal_training':
        if [r['path'] for r in raw['calibration']]!=manifest['calibration']:
            raise ValueError('Calibration path mismatch')
        calibration=[r['score'] for r in raw['calibration']]
    else:
        if raw['calibration_paths']!=manifest['calibration']:
            raise ValueError('Calibration path mismatch')
        if raw['reference_paths']!=manifest['references'][str(raw['shots'])][str(raw['seed'])]:
            raise ValueError('Unexpected reference selection')
        calibration=raw['calibration_scores']
    return raw,manifest,calibration,manifest_path


def run(source, output):
    source=Path(source).resolve(); output=Path(output).resolve()
    files=sorted(set(source.glob('runs/*/*/run.json')) | set(source.glob('runs/*/*/prediction_manifest.json')))
    if not files: raise ValueError('No exported complete prediction manifests found')
    frozen={**CONFIG,'source_sha256':{p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name('selective.py')]}}
    config_id='review_'+hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()[:12]
    output.mkdir(parents=True,exist_ok=True)
    if list(output.glob('runs/*/*/selective.json')) and (output/'frozen_config.json').exists() and json.loads((output/'frozen_config.json').read_text(encoding='utf-8'))!=frozen:
        raise ValueError('Selective source/config changed: choose a new result root')
    save_json(output/'frozen_config.json',frozen)
    rows=[]
    for path in files:
        raw,manifest,calibration,origin=read_scores(path.parent)
        scores=[r['score'] for r in raw['test']]; labels=[r['label'] for r in manifest['test']]
        started=time.perf_counter(); result=evaluate_selective(calibration,scores,labels)
        elapsed=time.perf_counter()-started
        result.update(category=raw['category'],shots=raw['shots'],seed=raw['seed'],source_method=raw['method'],
                      source_config_id=raw['config_id'],source_manifest_sha256=sha256(origin),
                      source_manifest=origin.relative_to(ROOT).as_posix(),config_id=config_id,
                      protocol_id=PROTOCOL_ID,integration_only=False,
                      scope='formal selective analysis of the named detector; not a DINO reproduction',
                      evaluation_cpu_seconds=elapsed,detector_scores_unchanged=True)
        destination=output/'runs'/raw['category']/path.parent.name
        save_json(destination/'selective.json',result)
        policy=ReviewPolicy.fit(calibration); decisions=policy.decisions(scores)
        with (destination/'decisions.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=['path','label','score','threshold','review_width','decision','baseline_decision'])
            writer.writeheader()
            for sample,pred,decision in zip(manifest['test'],raw['test'],decisions):
                writer.writerow({'path':sample['path'],'label':sample['label'],'score':pred['score'],
                    'threshold':policy.threshold,'review_width':policy.width,'decision':decision,
                    'baseline_decision':'NG' if pred['score']>policy.threshold else 'OK'})
        rows.append(result)
    grid={(r['category'],r['shots'],r['seed']) for r in rows}
    if len(grid)!=len(rows): raise ValueError('Duplicated detector cells')
    shot_values=sorted(set(r['shots'] for r in rows),key=lambda x:-1 if x is None else x)
    summaries=[]
    for shot in shot_values:
        group=[r for r in rows if r['shots']==shot]
        if {(r['category'],r['seed']) for r in group}!={(c,s) for c in CATEGORIES for s in range(3)}:
            raise ValueError('Incomplete six-category three-seed grid')
        summary={'shots':shot,'source_method':group[0]['source_method'],'per_category':{}}
        for category in [*CATEGORIES,'six_category_macro']:
            selected=group if category=='six_category_macro' else [r for r in group if r['category']==category]
            modes={}
            for mode in ['none','symmetric','lower_only']:
                modes[mode]={}
                for key in ['coverage','review_rate','risk','automatic_fnr','automatic_fpr','accepted_anomaly_fnr']:
                    byseed=[]
                    for seed in range(3):
                        values=[r['policies'][mode][key] for r in selected if r['seed']==seed]
                        byseed.append(float(np.mean(values)) if all(v is not None for v in values) else None)
                    modes[mode][key]={'mean':float(np.mean(byseed)) if None not in byseed else None,
                                     'std':float(np.std(byseed)) if None not in byseed else None}
            modes['aurc']={'mean':float(np.mean([r['aurc'] for r in selected]))}
            modes['fixed_fpr_fnr']={'mean':float(np.mean([r['fixed_fpr_test_roc']['fnr'] for r in selected]))}
            summary['per_category'][category]=modes
        summaries.append(summary)
    save_json(output/'summary.json',summaries)
    save_json(output/'completion_audit.json',{'status':'complete_for_named_detector','n_runs':len(rows),
              'expected_cells':len(CATEGORIES)*3*len(shot_values),'source_manifests_verified':True,
              'source_method':rows[0]['source_method'],'dino_comparison_complete':False,
              'review_not_counted_as_correct':True,'test_labels_not_used_to_fit_policy':True,
              'no_additional_maps_stored':True,'config_id':config_id})
    print('SELECTIVE_COMPLETE',source,len(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); run(args.source,args.output)
