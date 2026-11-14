"""Full six-category, three-initialization normal-only AE experiment."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import sys
import time
import cv2
import numpy as np
import psutil
import torch

from shared_protocol.data import load_manifest
from shared_protocol.protocol import CATEGORIES, PROTOCOL_ID, inside, save_json, sha256
from methods.normal_training import validate_training, save_map, evaluate_run
from .ae import CONFIG, CompactAE, predict
from .template import read_rgb, preprocess

ROOT = Path(__file__).resolve().parents[2]


def frozen_config(output, results):
    files = [Path(__file__), Path(__file__).with_name('ae.py'), Path(__file__).with_name('template.py'),
             ROOT / 'methods/normal_training.py']
    config = {**CONFIG, 'algorithm_source_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in files},
              'python': sys.version, 'dependencies': {p: importlib.metadata.version(p) for p in
                   ['torch','numpy','scipy','scikit-learn','Pillow','opencv-python-headless']}}
    config['config_id'] = 'normal_ae_' + hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:12]
    for folder in [output, results]:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'frozen_config.json'
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != config:
            raise ValueError('Frozen experiment/source/environment changed; use a new experiment root')
        save_json(path, config)
    return config


def train(manifest, seed, data_root, folder, config):
    names = manifest['normal_training_pool']
    validate_training(manifest, names)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = CompactAE().cuda()
    checkpoint = folder / 'model.pt'
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if payload['config_id'] != config['config_id'] or payload['seed'] != seed or payload['training_paths'] != names:
            raise ValueError('Checkpoint metadata mismatch')
        model.load_state_dict(payload['state_dict'])
        return model.eval(), payload['training_resources']
    started = time.perf_counter()
    images = [preprocess(read_rgb(inside(data_root, p))) for p in names]
    groups = {}
    for i, image in enumerate(images):
        groups.setdefault(image.shape, []).append(i)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    generator = np.random.default_rng(seed)
    losses = []
    torch.cuda.reset_peak_memory_stats()
    model.train()
    for epoch in range(CONFIG['epochs']):
        total, seen = 0., 0
        batches = []
        for indices in groups.values():
            order = generator.permutation(indices)
            batches.extend(order[i:i+CONFIG['batch_size']] for i in range(0,len(order),CONFIG['batch_size']))
        generator.shuffle(batches)
        for indices in batches:
            array = np.stack([images[i] for i in indices]).transpose(0,3,1,2).copy()
            x = torch.from_numpy(array).cuda().float() / 255.
            for axis in [-1,-2]:
                flipped = torch.rand(len(x), device='cuda') < .5
                x[flipped] = x[flipped].flip(axis)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x) - x).square().mean()
            loss.backward(); optimizer.step()
            total += loss.item() * len(indices); seen += len(indices)
        if seen != len(names):
            raise ValueError('Epoch did not cover each normal training image exactly once')
        losses.append(total / seen)
        print('TRAIN', manifest['category'], seed, epoch+1, '/', CONFIG['epochs'], 'MSE', round(losses[-1],7), flush=True)
    torch.cuda.synchronize()
    resources = {'training_seconds_including_read_preprocess': time.perf_counter()-started,
                 'training_peak_gpu_allocated_mb': torch.cuda.max_memory_allocated()/1024**2,
                 'training_count': len(names), 'epochs': CONFIG['epochs'],
                 'training_shapes': [{'height':s[0], 'width':s[1], 'count':len(g)} for s,g in groups.items()],
                 'parameter_count': sum(p.numel() for p in model.parameters()), 'training_loss_by_epoch': losses}
    torch.save({'config_id': config['config_id'], 'seed':seed, 'training_paths': names,
                'training_resources':resources, 'state_dict':model.cpu().state_dict()}, checkpoint)
    save_json(folder / 'training.json', resources)
    return model.cuda().eval(), resources


def run_one(category, seed, args, config):
    manifest = load_manifest(category)
    folder = args.output / category / f'seed{seed}'
    compact = args.results / 'runs' / category / f'seed{seed}'
    folder.mkdir(parents=True, exist_ok=True); compact.mkdir(parents=True, exist_ok=True)
    if not (folder / 'run.json').exists():
        model, training = train(manifest, seed, args.data_root, folder, config)
        for name in manifest['calibration'][:3]:
            predict(model, inside(args.data_root, name))
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        calibration = [{'path':p, 'score':predict(model,inside(args.data_root,p))[0]} for p in manifest['calibration']]
        maps = folder / 'maps'; maps.mkdir(exist_ok=True)
        existing = folder / 'prediction_progress.json'
        records = json.loads(existing.read_text(encoding='utf-8')) if existing.exists() else []
        if [r['path'] for r in records] != [r['path'] for r in manifest['test'][:len(records)]]:
            raise ValueError('Prediction resume order mismatch')
        peak_rss = psutil.Process().memory_info().rss
        for index, sample in enumerate(manifest['test']):
            if index < len(records):
                if sha256(folder / records[index]['map']) != records[index]['map_sha256']:
                    raise ValueError('Resume map was modified')
                continue
            torch.cuda.synchronize(); started = time.perf_counter()
            score, values, shape = predict(model,inside(args.data_root,sample['path']))
            torch.cuda.synchronize(); elapsed = time.perf_counter()-started
            relative = f'maps/{index:04d}.npz'
            digest = save_map(folder / relative, values)
            records.append({'path':sample['path'], 'score':score, 'map':relative,
                            'map_sha256':digest, 'native_shape':list(values.shape), 'input_shape':shape,
                            'inference_seconds':elapsed})
            save_json(existing, records)
            peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
            if (index+1)%50==0 or index+1==len(manifest['test']):
                print('PREDICT',category,seed,index+1,'/',len(manifest['test']),flush=True)
        resources = {**training, 'device':torch.cuda.get_device_name(), 'gpu_used':True,
                     'peak_gpu_allocated_mb':torch.cuda.max_memory_allocated()/1024**2,
                     'peak_process_rss_mb':peak_rss/1024**2,
                     'inference_seconds_mean':float(np.mean([r['inference_seconds'] for r in records])),
                     'timing_includes':'image reading, resize/crop, AE forward, smoothing, top1% score, native interpolation, CUDA synchronization',
                     'timing_excludes':'training, calibration, warmup, NPZ compression and verification, evaluation',
                     'warmup_images':3}
        save_json(folder / 'run.json', {'method':CONFIG['method'], 'experiment_mode':'normal_training',
                  'shots':None, 'reference_paths':[], 'seed':seed, 'seed_role':'training_initialization_and_shuffle',
                  'category':category,'protocol_id':PROTOCOL_ID,'config_id':config['config_id'], 'run_config':config,
                  'integration_only':False,'training_paths':manifest['normal_training_pool'],
                  'training_count':len(manifest['normal_training_pool']), 'calibration':calibration,
                  'test':records,'resources':resources,'checkpoint_sha256':sha256(folder / 'model.pt')})
        del model; torch.cuda.empty_cache()
    raw = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
    if raw['run_config'] != config:
        raise ValueError('Existing run is from a different frozen configuration')
    if (folder / 'metrics.json').exists():
        result = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
        if result['run_manifest_sha256'] != sha256(folder / 'run.json'):
            raise ValueError('Previously evaluated manifest changed')
    else:
        print('EVALUATE NATIVE',category,seed,flush=True)
        result = evaluate_run(folder,manifest,args.data_root,optional_metrics=True)
    save_json(compact / 'metrics.json',result)
    for name in ['run.json','training.json']:
        shutil.copyfile(folder / name, compact / name)
    with (compact / 'image_predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['path','label','score','threshold','predicted_ng','inference_seconds'])
        writer.writeheader()
        for sample,pred in zip(manifest['test'],raw['test']):
            writer.writerow({'path':sample['path'],'label':sample['label'],'score':pred['score'],
                             'threshold':result['image_threshold'],'predicted_ng':int(pred['score']>result['image_threshold']),
                             'inference_seconds':pred['inference_seconds']})
    print('RESULT',category,seed,result['image_auroc'],result['pixel_auroc'],flush=True)
    return result


def summarize(records):
    keys=['image_auroc','pixel_auroc','image_ap','pixel_ap','aupro_03','image_fpr','image_fnr']
    output={'experiment_mode':'normal_training','shots':None,'seeds':[0,1,2],'std_ddof':0,'per_category':{}}
    indexed={(r['category'],r['seed']):r for r in records}
    expected={(c,s) for c in CATEGORIES for s in range(3)}
    if len(indexed)!=len(records) or set(indexed)!=expected:
        raise ValueError('Expected complete six-category three-initialization grid')
    for category in CATEGORIES:
        output['per_category'][category]={'training_count':indexed[category,0]['training_count']}
        for key in keys:
            values=[indexed[category,s][key] for s in range(3)]
            output['per_category'][category][key]={'mean':float(np.mean(values)), 'std':float(np.std(values))}
    output['macro']={}
    for key in keys:
        values=[np.mean([indexed[c,s][key] for c in CATEGORIES]) for s in range(3)]
        output['macro'][key]={'mean':float(np.mean(values)), 'std':float(np.std(values))}
    output['inference_ms_mean']=float(np.mean([r['resources']['inference_seconds_mean']*1000 for r in records]))
    output['peak_gpu_allocated_mb_max']=max(r['resources']['peak_gpu_allocated_mb'] for r in records)
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/part1_ae_v1')
    p.add_argument('--results',type=Path,default=ROOT/'results/part1_ae')
    p.add_argument('--categories',nargs='+',choices=CATEGORIES,default=CATEGORIES)
    p.add_argument('--seeds',nargs='+',type=int,choices=range(3),default=[0,1,2])
    args=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('This frozen experiment requires CUDA')
    torch.set_num_threads(4); cv2.setNumThreads(4)
    config=frozen_config(args.output,args.results)
    for category in args.categories:
        for seed in args.seeds:
            run_one(category,seed,args,config)
            records=[json.loads(x.read_text(encoding='utf-8')) for x in args.results.glob('runs/*/*/metrics.json')]
            save_json(args.results/'progress.json',{'completed':len(records),'required':18,
                      'runs':[{'category':r['category'],'seed':r['seed']} for r in records]})
    records=[json.loads(x.read_text(encoding='utf-8')) for x in args.results.glob('runs/*/*/metrics.json')]
    if len(records)==18:
        save_json(args.results/'summary.json',summarize(records))
        print('ALL_AE_COMPLETE',flush=True)


if __name__=='__main__': main()
