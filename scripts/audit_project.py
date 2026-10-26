from pathlib import Path
import sys,json,zipfile
import numpy as np
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'coursework'))
from common import CATEGORIES,OUT,split,masks_for
from metrics import normal_threshold
counts={};checks=[]
for family,expected in [('dinov2',216),('baselines',72),('anomalyclip',12),('robustness',54),('coreset',36),('lowmiss',72),('peak',72)]:
 files=list((OUT/family).rglob('metrics.json'));counts[family]=len(files);assert len(files)==expected,(family,len(files),expected)
 for p in files:
  m=json.loads(p.read_text());assert all(np.isfinite(m[k]) and 0<=m[k]<=1 for k in ['image_auroc','image_ap','pixel_auroc','pixel_ap','aupro_03','image_fpr','image_fnr'])
  if family!='robustness':
   manifest=json.loads((OUT/'manifests'/f"{m['category']}.json").read_text())
   assert m['n_test']==len(manifest['test'])
   with np.load(p.parent/'calibration.npz') as d:
    assert len(d['scores'])==40
    assert abs(normal_threshold(d['scores'])-m['image_threshold'])<1e-7
   assert (p.parent/('scores.npz' if family=='peak' else 'predictions.npz')).exists()
   if family=='peak':
    source=OUT/m['map_source'];parent=json.loads((source/'metrics.json').read_text())
    assert all(m[k]==parent[k] for k in ['pixel_auroc','pixel_ap','aupro_03'])
   if family=='lowmiss':
    assert m['bank_bytes']==6291456 and m['bank_tokens']==4096
    assert m['reference_paths']==manifest['references'][str(m['seed'])]
    with np.load(p.parent/'demo_bank.npz') as bank:
     keys=['low_bank','high_bank'] if m['method']=='multiscale' else ['bank']
     assert sum(bank[k].nbytes for k in keys)==6291456
   if family=='coreset':
    with np.load(p.parent/'demo_bank.npz') as bank:
     assert bank['bank'].shape==(4096,384) and bank['bank'].nbytes==6291456
     assert len(np.unique(bank['selected_indices']))==4096
    old=json.loads((OUT/'dinov2'/m['category']/f"8shot_seed{m['seed']}_full"/'metrics.json').read_text())
    assert m['reference_paths']==old['reference_paths'] and m['selection_projection_dim']==128
 checks.append(f'{family}: {len(files)} finite metric records; expected count verified')
for c in CATEGORIES+['macaroni1','macaroni2','pcb1','pcb2','pcb3','pcb4']:
 pool,cal,refs,test=split(c)
 assert len(cal)==40 and set(cal).isdisjoint(pool)
 masks=masks_for(test);labels=masks.reshape(len(test),-1).any(1)
 assert all(bool(label)==(p.parent.name=='bad') for label,p in zip(labels,test))
checks.append('All twelve manifests consistent; calibration disjoint; resized-mask labels match folder labels')
demo=json.loads((r/'deliverables/demo/predictions.json').read_text());assert len(demo)==12 and all(x['matches_saved_prediction'] for x in demo)
checks.append('All 12 regenerated demo predictions match saved formal predictions')
core_demo=json.loads((r/'deliverables/demo_coreset/predictions.json').read_text());assert len(core_demo)==12 and all(x['matches_saved_prediction'] for x in core_demo)
checks.append('All 12 coreset demo predictions match saved formal predictions')
new_demo=json.loads((r/'deliverables/demo_lowmiss/predictions.json').read_text());assert len(new_demo)==96 and all(x['matches_saved_prediction'] for x in new_demo)
checks.append('96 new demo predictions across 12 categories and four variants match formal outputs')
def review_record(name):
 path=OUT/name
 return path if path.exists() else r/'reference_results'/name
annotations=json.loads(review_record('failure_annotations.json').read_text());assert len(annotations)==24
extra=json.loads(review_record('lowmiss_additional_errors.json').read_text());assert len(extra)==12 and all(x['visually_reviewed'] for x in extra)
checks.append('Reported-study review records present: 24 original and 12 additional-category errors; fresh rerun errors require a new visual review')
import fitz
doc=fitz.open(r/'reports/main.pdf');body=None
for i,p in enumerate(doc):
 text=p.get_text()
 if '\nReferences\n' in text or text.startswith('References'):body=i;break
assert body is not None and 6<=body<=8,('body pages',body,'total',len(doc))
for p in doc:assert len(p.get_text().strip())>100
checks.append(f'PDF: {body} body pages plus {len(doc)-body} reference pages; text on every page')
result={'passed':True,'counts':counts,'checks':checks,'body_pages':body,'pdf_pages':len(doc),'pending':['Second student ID and English spelling confirmation','arXiv deposit and actual identifier']}
(r/'deliverables/verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
