"""Render published, fixed summary CSVs. No model calls or score changes."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

P = Path(__file__).resolve().parent
rows = {r['arm']: r for r in csv.DictReader((P / 'summary.csv').open())}
arms = ['old', 'norm8', 'norm_token_uncapped']
colors = ['#8796A5', '#167D9A', '#D47B39']
labels = ['Original', 'Normalized\n8,192 generation cap', 'Normalized\nContext bound only']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.labelcolor': '#334155', 'text.color': '#172B40',
                     'axes.edgecolor': '#CCD5DE', 'pdf.fonttype': 42,
                     'savefig.facecolor': 'white'})

def clean(ax):
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='both', length=0, pad=8)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#E9EEF3', linewidth=.8)

def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(P / f'{name}.{ext}', dpi=200, bbox_inches='tight', pad_inches=.22)
    plt.close(fig)

fig, axs = plt.subplots(2, 2, figsize=(13.6, 9.2))
fig.subplots_adjust(left=.075, right=.96, bottom=.15, top=.75, hspace=.70, wspace=.26)
fig.text(.075, .965, 'NATIVE ZOOM  /  A-SFT EVALUATION', fontsize=10, weight='bold', color='#167D9A')
fig.text(.075, .925, 'Coordinate normalization improved accuracy;\na larger generation budget left correctness unchanged.', fontsize=20, weight='bold', linespacing=1.25, va='top')
fig.text(.075, .805, 'Same checkpoint and E1 questions  •  552 questions × 4 samples per condition', fontsize=11, color='#526477')
metrics = [('accuracy', 100, 'A   Answer accuracy', 'Accuracy (%)', 60),
           ('mean_crops', 1, 'B   Executed crops', 'Crops / trajectory', 7),
           ('mean_generation_tokens', 1, 'C   Generated tokens', 'Tokens / trajectory', 650),
           ('mean_processed_tokens', 1, 'D   Total processed tokens', 'Tokens / trajectory', 18500)]
for ax, (key, factor, title, ylabel, ymax) in zip(axs.flat, metrics):
    vals = [float(rows[a][key]) * factor for a in arms]
    bars = ax.bar(range(3), vals, color=colors, width=.54, zorder=3)
    clean(ax)
    ax.set(ylabel=ylabel, ylim=(0, ymax))
    ax.set_title(title, loc='left', pad=17, weight='bold', fontsize=13)
    ax.set_xticks(range(3), labels, fontsize=10)
    for i, (bar, val) in enumerate(zip(bars, vals)):
        value = f'{val:.2f}%' if key == 'accuracy' else (f'{val:,.0f}' if val > 1000 else f'{val:.2f}')
        ax.text(bar.get_x()+bar.get_width()/2, val+ymax*.025, value,
                ha='center', weight='bold', fontsize=12)
        if key == 'accuracy':
            ax.text(i, 3, f"{rows[arms[i]]['correct']} / 2,208", ha='center', color='white', fontsize=9)
    if key == 'mean_processed_tokens':
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v/1000:g}k'))
fig.text(.075, .066, 'All conditions retain 8 crops and a 32,768-token context. Failures remain in the denominator; E1 is not blind.', fontsize=10, color='#526477')
fig.text(.075, .033, 'Processed tokens include repeated prompts, images and generation across turns. Context-bound generation is not unlimited.', fontsize=10, color='#526477')
save(fig, 'comparison')

sources = [('mme_realworld', 'MME-RealWorld'), ('mmmu_pro', 'MMMU-Pro'), ('vstar', 'V*'), ('worldbench', 'WorldBench')]
s = {(r['arm'], r['source']): r for r in csv.DictReader((P/'by_source.csv').open())}
fig, ax = plt.subplots(figsize=(12.8, 6.6))
fig.subplots_adjust(left=.09, right=.96, bottom=.24, top=.73)
fig.text(.09, .94, 'NATIVE ZOOM  /  FOUR DATA SOURCES', fontsize=10, weight='bold', color='#167D9A')
fig.text(.09, .875, 'The two normalized conditions match in every source.', fontsize=20, weight='bold')
fig.text(.09, .815, 'Rates are computed within each source; denominators differ across sources.', fontsize=11, color='#526477')
for j, arm in enumerate(arms):
    xs = [i+(j-1)*.24 for i in range(4)]
    vals = [100*int(s[(arm, src)]['correct'])/int(s[(arm, src)]['n']) for src, _ in sources]
    bars=ax.bar(xs, vals, width=.21, color=colors[j], zorder=3)
    ax.bar_label(bars, labels=[f'{v:.2f}' for v in vals], padding=5, fontsize=10, weight='bold')
clean(ax)
ax.set(ylim=(0, 70), ylabel='Answer accuracy (%)')
ax.set_xticks(range(4), [f"{label}\nn = {int(s[('old', src)]['n']):,}" for src,label in sources])
fig.legend(handles=[Line2D([0],[0],color=c,lw=8,label=l) for c,l in zip(colors,['Original','Normalized / 8,192 cap','Normalized / context bound only'])], loc='lower center', bbox_to_anchor=(.52,.095), frameon=False,ncol=3,fontsize=10)
fig.text(.09, .037, 'Each condition: 2,208 trajectories. Same question/sample keys, labels and scorer; failed and truncated outputs retained.', fontsize=10,color='#526477')
save(fig,'source_comparison')
