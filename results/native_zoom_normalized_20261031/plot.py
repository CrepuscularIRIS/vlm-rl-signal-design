from pathlib import Path
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
rows=list(csv.DictReader((P/'summary.csv').open()))
labels=['Original native','Normalized / 8192','Normalized / context only']
colors=['#9b9fa6','#367ca9','#d99a37']
fig,axs=plt.subplots(2,2,figsize=(12,7))
for ax,key,title,scale in zip(axs.flat,['accuracy','mean_crops','mean_generation_tokens','mean_processed_tokens'],['Accuracy (%)','Crops per trajectory','Generated tokens per trajectory','Processed tokens per trajectory'],[100,1,1,1]):
 vals=[float(r[key])*scale for r in rows];bars=ax.bar(labels,vals,color=colors,width=.62)
 ax.bar_label(bars,fmt='%.2f',padding=3);ax.set_title(title);ax.set_ylim(0,max(vals)*1.18);ax.tick_params(axis='x',labelsize=9);ax.spines[['top','right']].set_visible(False)
fig.suptitle('A-SFT Native Zoom: coordinate adapter and generation-cap comparison',fontsize=14)
fig.text(.5,.01,'552 questions x 4 samples; failures retained. All arms keep 8 crops and 32,768 context. E1 is not blind.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.035,1,.96])
for ext in ['png','pdf']:fig.savefig(P/f'comparison.{ext}',dpi=180)
plt.close(fig)
