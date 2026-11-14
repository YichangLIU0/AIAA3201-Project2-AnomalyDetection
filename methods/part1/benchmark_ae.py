"""Serial GPU latency on all official tests, after training workers have exited.

Verifies each recomputed image score equals its saved formal prediction; does not
retune anything or overwrite stored predictions. Excludes NPZ compression/I/O.
"""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
import psutil
import torch
from shared_protocol.data import load_manifest
from shared_protocol.protocol import CATEGORIES,inside,save_json,sha256
from .ae import CompactAE,predict


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('outputs/part1_ae_v1'))
    p.add_argument('--results',type=Path,default=Path('results/part1_ae'))
    a=p.parse_args();rows=[]
    # Avoid inadvertently claiming a benchmark while another training Python is active.
    others=[]
    for proc in psutil.process_iter(['pid','name','cmdline']):
        if proc.pid==psutil.Process().pid: continue
        command=proc.info.get('cmdline') or []
        if any('methods.part1.run_ae'==str(x) for x in command): others.append(proc.pid)
    if others: raise RuntimeError(f'Wait for AE training workers to finish: {others}')
    torch.set_num_threads(4);cv2.setNumThreads(4)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for category in CATEGORIES:
        manifest=load_manifest(category)
        for seed in range(3):
            folder=a.output/category/f'seed{seed}'
            raw=json.loads((folder/'run.json').read_text(encoding='utf-8'))
            payload=torch.load(folder/'model.pt',weights_only=True,map_location='cpu')
            if sha256(folder/'model.pt')!=raw['checkpoint_sha256'] or payload['config_id']!=raw['config_id']:
                raise ValueError('Wrong checkpoint for frozen prediction')
            model=CompactAE().cuda();model.load_state_dict(payload['state_dict']);model.eval()
            for path in manifest['calibration'][:3]: predict(model,inside(a.data_root,path))
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            timings=[]
            for sample,record in zip(manifest['test'],raw['test']):
                torch.cuda.synchronize();start=time.perf_counter()
                score,_,_=predict(model,inside(a.data_root,sample['path']))
                torch.cuda.synchronize();timings.append(time.perf_counter()-start)
                if score!=record['score']: raise ValueError(f'Serial score differs from saved prediction: {sample["path"]}')
            row={'category':category,'seed':seed,'n_test':len(timings),
                 'inference_seconds_mean':float(np.mean(timings)),
                 'inference_seconds_std':float(np.std(timings)),
                 'peak_gpu_allocated_mb':torch.cuda.max_memory_allocated()/1024**2,
                 'each_recomputed_image_score_exactly_equals_saved':True,
                 'run_manifest_sha256':sha256(folder/'run.json'),'checkpoint_sha256':sha256(folder/'model.pt')}
            rows.append(row);save_json(a.results/'serial_benchmark_progress.json',rows)
            print('SERIAL_BENCH',category,seed,round(row['inference_seconds_mean']*1000,2),'ms',flush=True)
            del model;torch.cuda.empty_cache()
    save_json(a.results/'serial_benchmark.json',{
        'device':torch.cuda.get_device_name(),'n_cells':18,'test_predictions_recomputed':sum(r['n_test'] for r in rows),
        'inference_ms_class_seed_mean':float(np.mean([r['inference_seconds_mean']*1000 for r in rows])),
        'peak_gpu_allocated_mb_max':max(r['peak_gpu_allocated_mb'] for r in rows),
        'worker_check':'no run_ae workers active before benchmark',
        'includes':'read image, resize/crop, forward, RGB error, smoothing, top1% score, native resize, CUDA sync',
        'excludes':'training, warmup3 calibration images, compression/storage, metrics',
        'source_sha256':{p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name('ae.py')]},'runs':rows})


if __name__=='__main__':main()
