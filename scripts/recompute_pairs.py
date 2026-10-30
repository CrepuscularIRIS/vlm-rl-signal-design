"""CPU-only reaggregation of frozen E1 exports; no inference or changed scoring."""
import csv,gzip,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'recomputed');args=parser.parse_args()
OUT=args.out;OUT.mkdir(parents=True,exist_ok=True)
P=ROOT/'data/rollouts'
FILES={'base_cot':'B0','base_zoom':'B1','sft':'B3','grpo':'K-GRPO','dapo':'K-DAPO','gspo':'K-GSPO','gigpo':'K-GiGPO','rloo':'V53-control-32','decision':'V53-decision-32','mini':'X1','deepeyes':'X2'}
D={}; hashes={}
for m,f in FILES.items():
 p=P/(f+'.jsonl.gz');rows=[json.loads(s) for s in gzip.open(p,'rt')]
 D[m]={(r['item_id'],r['sample_idx']):r for r in rows}
 hashes[m]=hashlib.sha256(p.read_bytes()).hexdigest()
 assert len(D[m])==len(rows)==2208
keys=sorted(D['grpo']); uids=sorted({k[0] for k in keys});assert len(uids)==552
for m,rows in D.items():
 assert set(rows)==set(keys)
 for k,r in rows.items():
  assert (r['gold_letter'],r['bench'])==(D['grpo'][k]['gold_letter'],D['grpo'][k]['bench'])
Q={m:{u:sum(rows[u,s]['correct'] for s in range(4)) for u in uids} for m,rows in D.items()}
pairs=[('sft',m) for m in ['grpo','dapo','gspo','gigpo','rloo','decision']]+[('grpo',m) for m in ['dapo','gspo','gigpo','rloo','decision','mini','deepeyes']]+[('rloo','decision'),('dapo','mini'),('base_cot','sft'),('base_zoom','sft')]
summary=[]; strata=[]; details=[]
for a,b in pairs:
 gains=[k for k in keys if not D[a][k]['correct'] and D[b][k]['correct']]
 losses=[k for k in keys if D[a][k]['correct'] and not D[b][k]['correct']]
 diff=np.array([(Q[b][u]-Q[a][u])/4 for u in uids])
 rng=np.random.default_rng(20261029);boot=diff[rng.integers(0,len(uids),(5000,len(uids)))].mean(axis=1)*100
 row=dict(reference=a,method=b,gain_rows=len(gains),loss_rows=len(losses),net_rows=len(gains)-len(losses),delta_pp=float(diff.mean()*100),q_improved=int((diff>0).sum()),q_worse=int((diff<0).sum()),q_equal=int((diff==0).sum()),zero_to_four=sum(Q[a][u]==0 and Q[b][u]==4 for u in uids),four_to_zero=sum(Q[a][u]==4 and Q[b][u]==0 for u in uids),ci95_prompt_bootstrap_pp=[float(v) for v in np.quantile(boot,[.025,.975])])
 summary.append(row)
 for src in sorted({r['bench'] for r in D[a].values()}):
  ks=[k for k in keys if D[a][k]['bench']==src]
  strata.append(dict(reference=a,method=b,source=src,n=len(ks),gain=sum(k in gains for k in ks),loss=sum(k in losses for k in ks),correct_ref=sum(D[a][k]['correct'] for k in ks),correct_method=sum(D[b][k]['correct'] for k in ks)))
 for u in uids:
  details.append(dict(reference=a,method=b,item_id=u,source=D[a][u,0]['bench'],correct_ref_of4=Q[a][u],correct_method_of4=Q[b][u],delta_of4=Q[b][u]-Q[a][u],crops_ref=sum(D[a][u,s]['crops'] for s in range(4))/4,crops_method=sum(D[b][u,s]['crops'] for s in range(4))/4))
# Descriptive diagnostics, never mechanistic labels.
diag={}
for m,rows in D.items():
 rs=list(rows.values()); flagged=lambda r:any(t.get('check_errors') for t in r['turns'])
 diag[m]=dict(correct=sum(r['correct'] for r in rs),n=len(rs),zero_of4=sum(Q[m][u]==0 for u in uids),four_of4=sum(Q[m][u]==4 for u in uids),no_crop=sum(r['crops']==0 for r in rs),no_crop_correct=sum(r['correct'] for r in rs if r['crops']==0),flagged=(None if m in ['base_cot','base_zoom','mini','deepeyes'] else sum(flagged(r) for r in rs)),flagged_correct=(None if m in ['base_cot','base_zoom','mini','deepeyes'] else sum(r['correct'] for r in rs if flagged(r))),truncated=sum(bool(r['truncated']) for r in rs),no_parsed_answer=sum(r['final_answer'] is None for r in rs))
methods=['grpo','dapo','gspo','gigpo','rloo','decision','mini','deepeyes']
common_zero=[u for u in uids if all(Q[m][u]==0 for m in methods)]
result=dict(summary=summary,diagnostics=diag,all_eight_zero_of4=common_zero,input_sha256=hashes,notes=['Same sample_idx does not establish coupled randomness or a counterfactual.','Question-level count differences are primary; 4 samples not independent questions.','Prompt bootstrap conditions on fixed checkpoints and samples, not training seed variance.','RLOO versus GRPO changes reward and training protocol too; external protocols/backbones differ.'])
(OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
for name,rows in [('source_deltas',strata),('question_deltas',details)]:
 with (OUT/(name+'.csv')).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({'summary':summary,'diagnostics':diag,'all8_zero_n':len(common_zero)},ensure_ascii=False,indent=1))
