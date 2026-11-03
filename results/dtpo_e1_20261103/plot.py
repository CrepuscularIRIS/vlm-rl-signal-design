"""Render the published CSV tables with Python and matplotlib."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
def read(name):
    with (root/name).open() as stream:
        return list(csv.DictReader(stream))
def save(fig,name):
    for ext in ('png','pdf'):
        fig.savefig(root/f'{name}.{ext}', dpi=160, bbox_inches='tight')
    plt.close(fig)

points=read('training_points.csv')
methods=['GRPO','GSPO','GiGPO','DAPO','RLOO','PO-SP','PO-CA','SAPO','DTPO']
fig,axs=plt.subplots(3,3,sharex=True,sharey=True,figsize=(10,7),layout='constrained')
for ax,method in zip(axs.flat,methods):
    for series,color in [('train','#286b83'),('val','#bd4e27')]:
        data=[r for r in points if r['series']==series and r['method']==method]
        ax.plot([int(r['step']) for r in data],[100*float(r['accuracy']) for r in data],
                color=color,lw=1.6,marker='o' if series=='val' else None,
                linestyle='none' if series=='val' else '-',ms=5,zorder=3)
    ax.set_title(method,loc='left',size=11,fontweight='bold')
    ax.grid(alpha=.18)
    ax.set_xticks([0,8,16,24,32])
fig.suptitle('Accuracy over 32 completed steps',fontsize=17)
fig.supxlabel('Completed training step  |  Blue: training batch     Orange dots: development set (422 questions)',size=10)
fig.supylabel('Accuracy (%)',size=11)
save(fig,'training_curves')

rows=sorted(read('fixed_e1.csv'),key=lambda r:float(r['accuracy']))
fig,axs=plt.subplots(1,2,figsize=(9,4.4),sharey=True,layout='constrained')
for ax,key,scale,lo,hi,title in [(axs[0],'accuracy',100,54,69,'Accuracy (%)'),(axs[1],'processed',.001,0,18,'Processed tokens / trajectory (thousands)')]:
    for i,r in enumerate(rows):
        val=float(r[key])*scale
        color={'DTPO':'#bd4e27','DAPO':'#286b83'}.get(r['method'],'#647787')
        ax.hlines(i,lo,val,color=color,alpha=.32,lw=2)
        ax.scatter(val,i,color=color,s=32)
        ax.annotate(f'{val:.2f}',(val,i),xytext=(5,0),textcoords='offset points',va='center',fontsize=8)
    ax.set_xlim(lo,hi);ax.set_title(title,loc='left',fontsize=11);ax.grid(axis='x',alpha=.18)
axs[0].set_yticks(range(len(rows)),[r['method'] for r in rows])
fig.suptitle('Fixed E1  |  552 questions × 4 samples',fontsize=15)
save(fig,'e1_cost')
signals=read('dtpo_signals.csv')
steps=[int(r['prepared_call']) for r in signals]
fig,axs=plt.subplots(1,3,figsize=(10,3.4),layout='constrained')
axs[0].plot(steps,[int(r['mixed_groups'])/int(r['groups'])*100 for r in signals],color='#286b83')
axs[0].set_title('Mixed groups (%)')
axs[1].stackplot(steps,*[[int(r['adv_'+k+'_trajectories'])/int(r['trajectories'])*100 for r in signals] for k in ('positive','negative','zero')],colors=['#267983','#c46a43','#c9d0d5'],labels=['Positive','Negative','Zero'])
axs[1].legend(loc='lower center',ncol=3,fontsize=7);axs[1].set_title('Advantage sign (%)')
axs[2].plot(steps,[int(r['crops'])/int(r['trajectories']) for r in signals],color='#286b83');axs[2].set_title('Crops per trajectory')
for ax in axs:
    ax.set_xlabel('Completed step');ax.set_xticks([1,8,16,24,32]);ax.grid(alpha=.15)
fig.suptitle('DTPO training signals  |  128 groups × 8 samples per step')
save(fig,'dtpo_signals')

diag=read('diagnostics.csv')
fig,axs=plt.subplots(1,3,figsize=(10,3.4),layout='constrained')
for ax,key,title in zip(axs,['reward','old_logprob_max_diff','gradient_preclip_mean'],['Outcome reward (0–1)','Max old/new log-prob gap¹','Pre-clip gradient norm²']):
    ax.plot([int(r['step']) for r in diag],[float(r[key]) for r in diag],color='#286b83')
    ax.set_title(title,fontsize=11);ax.set_xlabel('Completed step');ax.grid(alpha=.17)
fig.suptitle('DTPO actor diagnostics  |  entropy unavailable')
fig.supxlabel('¹ Maximum logged difference over both ranks and four updates.    ² Mean over four updates, shared gradient counted once.',fontsize=8)
save(fig,'actor_logs')

data=sorted(read('all_conditions.csv'),key=lambda r:float(r['accuracy']),reverse=True)
fig,axs=plt.subplots(1,4,sharey=True,figsize=(11,9),layout='constrained')
for ax,field,scale,title in zip(axs,['accuracy','first_crop','later_crops','tokens'],[100,100,1,.001],['E1 accuracy (%)','Used ≥1 crop (%)','Later crops per row','Processed tokens (k)']):
    values=[float(r[field])*scale for r in data]
    for i,(r,v) in enumerate(zip(data,values)):
        color={'DTPO':'#bd4e27','K-DAPO':'#286b83'}.get(r['condition'],'#83939f')
        ax.hlines(i,0,v,color=color,lw=2,alpha=.45);ax.scatter(v,i,s=15,color=color,zorder=2)
    ax.set_title(title,fontsize=10);ax.set_xlim(left=-max(values)*.03);ax.grid(axis='x',alpha=.17)
axs[0].set_yticks(range(len(data)),[r['condition'] for r in data],fontsize=8);axs[0].invert_yaxis()
fig.suptitle(f'Fixed E1  |  {len(data)} evaluated conditions',fontsize=16)
fig.supxlabel('Each condition has 2,208 rows. Interfaces and training budgets differ. Archived decision variant omitted.',fontsize=9)
save(fig,'all_conditions')
