"""Render frozen public aggregate data; no training/inference or private paths."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--out',default='recomputed/curves');args=ap.parse_args()
OUT=Path(args.out);OUT.mkdir(parents=True,exist_ok=True)
rows=list(csv.DictReader((R/'data/algorithm_curves/curves.csv').open()))
for r in rows:
 for k,v in list(r.items()):
  if k!='method':r[k]=None if v=='' else int(v) if k=='step' else float(v)
assert len(rows)==192
methods=['grpo','dapo','gspo','gigpo','rloo','decision']
names=['GRPO','DAPO','GSPO','GiGPO','RLOO control','RLOO + first-action']
colors=['#377eb8','#e6550d','#4daf4a','#984ea3','#76624b','#e377c2']
vals=json.loads((R/'data/algorithm_curves/summary.json').read_text())['validation']
for m in methods:assert sorted(r['step'] for r in rows if r['method']==m)==list(range(1,33))
fig,axs=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
for m,name,col in zip(methods,names,colors):
 pts=sorted(vals[m]);mr={r['step']:r for r in rows if r['method']==m}
 axs[0].plot([s for s,v in pts],[100*v for s,v in pts],marker='o',ls='--',color=col,label=name)
 axs[1].plot([mr[s]['cumulative_processed_tokens']/1e6 if s else 0 for s,v in pts],[100*v for s,v in pts],marker='o',ls='--',color=col)
for ax in axs:ax.grid(alpha=.2);ax.set_ylabel('Development accuracy (%)')
axs[0].set_xlabel('Completed RL steps');axs[1].set_xlabel('Cumulative processed rollout tokens (millions; forks included)')
axs[0].set_title('Only measured validation points: 0 / 25 / 32');axs[1].set_title('Same checkpoints on a generation-cost axis')
axs[0].legend(fontsize=9);fig.suptitle('422-question development set; dashed segments are guides, not intermediate evaluations\nRLOO runs change training protocol and reward; single training seed',fontsize=11)
for ext in ['png','pdf']:fig.savefig(OUT/f'validation_curves.{ext}',dpi=180)
plt.close(fig)
fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
panels=[('generated_accuracy','Accuracy of ALL generated training rollouts'),('update_mixed_groups','Mixed binary-outcome groups in update / 128'),('crops','Executed crops / generated trajectory'),('entropy','Logged actor entropy (mask / reduction differ)'),('clip_higher','Logged upper clip fraction (token / sequence differ)'),('cumulative_generated_groups','Cumulative generated MAIN prompt-group occurrences')]
for ax,(key,title) in zip(axs.flat,panels):
 for m,name,col in zip(methods,names,colors):
  rs=[r for r in rows if r['method']==m];ax.plot([r['step'] for r in rs],[r[key] for r in rs],color=col,label=name,lw=1.3)
 ax.set(title=title,xlabel='RL step');ax.title.set_fontsize(10);ax.grid(alpha=.2)
axs[0,0].legend(fontsize=8);fig.suptitle('Observed training dynamics; different generated batches, not matched validation curves\nBinary mixed groups exclude signal from crop-cost reward or GiGPO step advantages',fontsize=12)
for ext in ['png','pdf']:fig.savefig(OUT/f'training_diagnostics.{ext}',dpi=180)
plt.close(fig)
