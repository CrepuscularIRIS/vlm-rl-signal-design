"""Verify the public evidence bundle without network, GPU or model calls."""
from pathlib import Path
from collections import Counter
import csv,gzip,hashlib,json,re
import scorer_reference as scorer
R=Path(__file__).resolve().parents[1]
errors=[]; count=0; methods={};base=None
sensitive=[re.compile(p) for p in [r'/home/[A-Za-z0-9_-]+/',('/'+'root/'),r'connect\.[\w.-]*seetacloud\.com',r'-----BEGIN [A-Z ]*PRIVATE KEY-----',r'gh[pousr]_[A-Za-z0-9]{20,}',r'github_pat_[A-Za-z0-9_]{20,}',r'sk-[A-Za-z0-9]{24,}',r'data:image/[^;]+;base64,',r'https?://[^\s/]+:[^\s@]+@']]
def scan(label,text):
 for rx in sensitive:
  if rx.search(text):errors.append('sensitive pattern in '+label+' / '+rx.pattern)
for p in R.rglob('*'):
 if not p.is_file() or '.git' in p.parts or 'recomputed' in p.parts or '__pycache__' in p.parts:continue
 if p.suffix in ['.md','.json','.jsonl','.csv','.py','.txt']:
  text=p.read_text();scan(str(p.relative_to(R)),text)
  if p.suffix=='.md':
   # Ignore code fences and inline code to avoid treating regex/code as links.
   t=re.sub(r'```[\s\S]*?```','',text)
   for dest in re.findall(r'!?\[[^\]]*\]\(([^)\s]+)\)',t):
    if dest.startswith(('https://','http://','#','mailto:')):continue
    q=(p.parent/dest.split('#')[0])
    if not q.exists():errors.append('broken link '+str(p.relative_to(R))+' -> '+dest)
for p in sorted((R/'data/rollouts').glob('*.gz')):
 rows=[]
 for line in gzip.open(p,'rt'):
  scan(p.name,line);rows.append(json.loads(line))
 keys=[(r['item_id'],r['sample_idx']) for r in rows]
 assert len(rows)==len(set(keys))==2208,p.name
 assert set(Counter(k[0] for k in keys).values())=={4}
 assert len({k[0] for k in keys})==552
 assert all({r['sample_idx'] for r in rows if r['item_id']==u}=={0,1,2,3} for u in {k[0] for k in keys})
 identity={k:(r['gold_letter'],r['bench']) for k,r in zip(keys,rows)}
 if base is None:base=identity
 else:assert base==identity,p.name
 for r in rows:
  pred,_=scorer.extract(r['turns'][-1]['text'] if r['turns'] else '')
  ok=int(pred is not None and pred==r['gold_letter'] and not r['error'] and not r['truncated'])
  if ok!=r['correct']:errors.append('score mismatch '+p.name+' '+r['item_id']+'/'+str(r['sample_idx']))
 methods[p.stem.removesuffix('.jsonl')]=sum(r['correct'] for r in rows)
 count+=len(rows)
assert len(methods)==25 and count==55200
A=list(map(json.loads,(R/'data/audit_findings.jsonl').read_text().splitlines()))
assert len(A)==len({(a['item_id'],a['sample_idx'],a['method']) for a in A})==96
assert len({a['item_id'] for a in A})==12
assert len(list((R/'observations/cases').glob('*.md')))==12
x=json.loads((R/'results/paired/results.json').read_text())
for s in x['summary']:
 assert s['q_improved']+s['q_worse']+s['q_equal']==552
 assert s['gain_rows']-s['loss_rows']==s['net_rows']
 assert abs(s['delta_pp']-s['net_rows']/2208*100)<1e-9
for r in json.loads((R/'provenance/CODE_HASHES.json').read_text()):
 assert hashlib.sha256((R/r['file']).read_bytes()).hexdigest()==r['sha256']
# Frozen training aggregates: denominators and endpoints, not a raw-log replay.
rows=list(csv.DictReader((R/'data/algorithm_curves/curves.csv').open()))
curves=json.loads((R/'data/algorithm_curves/summary.json').read_text())
assert len(rows)==192
for method,total in curves['training'].items():
 rs=sorted([r for r in rows if r['method']==method],key=lambda r:int(r['step']))
 assert [int(r['step']) for r in rs]==list(range(1,33))
 assert sum(int(r['generated_groups']) for r in rs)==total['generated_main_groups']
 assert sum(int(r['update_mixed_groups']) for r in rs)==total['update_mixed_groups']
 assert int(rs[-1]['cumulative_processed_tokens'])==total['processed_tokens_including_forks']
 assert abs(float(rs[-1]['cumulative_step_hours'])-total['summed_step_hours'])<1e-9
 assert [s for s,_ in curves['validation'][method]]==[0,25,32]
for v in json.loads((R/'data/algorithm_curves/generated_group_composition.json').read_text()).values():
 assert sum(v['binary_outcome_groups'].values())==v['generated_groups']
for r in json.loads((R/'provenance/CURVE_AUDIT.json').read_text())['outputs']:
 assert hashlib.sha256((R/r['path']).read_bytes()).hexdigest()==r['sha256']
f=R/'provenance/RELEASE_FILES.json'
if f.exists():
 for r in json.loads(f.read_text()):assert hashlib.sha256((R/r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
assert not errors,'\n'.join(errors[:40])
print(json.dumps(dict(status='pass',conditions=len(methods),trajectories=count,audits=len(A),cases=12,all_scores_recomputed=True,local_links_valid=True,sensitive_pattern_scan='pass'),indent=2))
