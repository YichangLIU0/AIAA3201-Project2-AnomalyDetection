import subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parents[1];python=str(r/'.venv/Scripts/python.exe')
extra=['candle','capsules','cashew','chewinggum','fryum','pipe_fryum','macaroni1','macaroni2','pcb1','pcb2','pcb3','pcb4']
for script in ['run_dino.py','run_coreset.py','run_baselines.py']:
 print('START_PHASE',script,flush=True)
 with (r/'outputs'/('all12_'+script+'.log')).open('w',encoding='utf-8') as log:
  subprocess.run([python,str(r/'coursework'/script),'--categories',*extra],cwd=r,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('DONE_PHASE',script,flush=True)
with (r/'outputs/all12_anomalyclip.log').open('w',encoding='utf-8') as log:
 code="import sys;sys.path.insert(0,'coursework');import run_anomalyclip as m;m.CATEGORIES="+repr(extra)+";m.main()"
 subprocess.run([python,'-c',code],cwd=r,stdout=log,stderr=subprocess.STDOUT,check=True)
(r/'outputs/all12_baselines_complete.txt').write_text('All additional-category baselines completed.\n')
print('ALL12_BASELINES_COMPLETE',flush=True)
