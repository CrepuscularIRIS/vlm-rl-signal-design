"""画 E1 分 benchmark 的配对差：每个来源一列，再加不加权平均一列；行按底座分组，不按分数排序。

用法：python plot_by_bench.py --data e1_by_bench.json --out DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = {'mme_realworld': 'MME-RealWorld', 'vstar': 'V*', 'worldbench': 'WorldBench', 'mmmu_pro': 'MMMU-Pro',
          'macro': 'Unweighted mean of 4'}
ORDER = [  # (name, group label) — grouped by base, never by score
    ('DAPO', 'optimizers vs GRPO'), ('GSPO', 'optimizers vs GRPO'), ('GiGPO', 'optimizers vs GRPO'),
    ('RLOO', 'optimizers vs GRPO'), ('SAPO', 'optimizers vs GRPO'),
    ('DTPO', 'GRPO-based vs GRPO'), ('G1', 'GRPO-based vs GRPO'), ('A+B', 'GRPO-based vs GRPO'),
    ('A+B+C', 'GRPO-based vs GRPO'), ('GRPO+A', 'GRPO-based vs GRPO'), ('strat', 'GRPO-based vs GRPO'),
    ('max2', 'GRPO-based vs GRPO'),
    ('Htau', 'DAPO-based vs DAPO'), ('DAPO-SB', 'DAPO-based vs DAPO'),
    ('RLOO+first-action', 'RLOO-based vs RLOO'),
]
GROUP_COLORS = {'optimizers vs GRPO': '#7f7f7f', 'GRPO-based vs GRPO': '#1f77b4',
                'DAPO-based vs DAPO': '#ff7f0e', 'RLOO-based vs RLOO': '#8c564b'}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    data = json.loads(args.data.read_text())
    cond = {c['name']: c for c in data['conditions']}
    rows = [(n, g) for n, g in ORDER if n in cond and 'vs_base' in cond[n]]
    cols = list(data['sources']) + ['macro']
    nq = data['questions_per_source']
    fig, axes = plt.subplots(1, len(cols), figsize=(22, 8.5), sharey=True, layout='constrained')
    ys = list(range(len(rows)))[::-1]
    for ax, col in zip(axes, cols):
        for y, (name, group) in zip(ys, rows):
            d = cond[name]['vs_base'][col]
            lo, hi = d['ci95']
            resolved = lo > 0 or hi < 0
            ax.errorbar(d['delta_pp'], y, xerr=[[d['delta_pp'] - lo], [hi - d['delta_pp']]], fmt='o',
                        color=GROUP_COLORS[group], ms=7 if resolved else 5, capsize=3,
                        mfc=GROUP_COLORS[group] if resolved else 'white', lw=1.4)
            ax.annotate(f"{d['delta_pp']:+.1f}", (d['delta_pp'], y), xytext=(0, 7), textcoords='offset points',
                        ha='center', fontsize=7, fontweight='bold' if resolved else 'normal')
        ax.axvline(0, color='black', lw=.8)
        title = LABELS[col] + (f'\n({nq[col]} questions x 4)' if col in nq else '\n(each source weighted 1/4)')
        ax.set(title=title, xlabel='accuracy difference vs own base (pp)')
        ax.grid(alpha=.2)
    axes[0].set_yticks(ys, [f'{n}  [{g.split(" vs ")[1]}]' for n, g in rows])
    handles = [axes[-1].plot([], [], 'o', color=c, label=g)[0] for g, c in GROUP_COLORS.items()]
    fig.legend(handles=handles, loc='outside lower center', ncol=len(handles), fontsize=9,
               title='filled marker = 95% CI excludes 0; open marker = not resolved')
    fig.suptitle('Fixed step-32 E1 by benchmark: each method vs its own optimizer base (GRPO-based vs GRPO, DAPO-based vs DAPO)\n'
                 'Paired by uid + sample_idx; 95% CI from 10,000 question resamples stratified by source; '
                 'no weighting by question count')
    for ext in ('png', 'pdf'):
        fig.savefig(args.out / f'e1_by_bench_vs_base.{ext}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
