"""从冻结长表重画训练 family、E1 family 及首裁/后续裁/处理 token 图。

python plot_delivery.py --data DATA_DIRECTORY
旧条件 E1 精度沿用已发布表；strat 与三个直接对照来自本轮统一重评分。
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save(fig: object, root: Path, name: str) -> None:
    for ext in ('png', 'pdf'):
        fig.savefig(root / f'{name}.{ext}', dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).parent)
    root = parser.parse_args().data
    data = defaultdict(dict)
    with (root / 'family_long.csv').open() as stream:
        for row in csv.DictReader(stream):
            key = (row['run'], row['metric'])
            step = int(row['step'])
            if step in data[key]:
                raise ValueError(f'重复run/metric/step: {key}/{step}')
            data[key][step] = float(row['value'])
    names = sorted({key[0] for key in data})
    colors = {name: plt.get_cmap('tab20')(i % 20) for i, name in enumerate(names)}
    colors.update({'strat': 'tab:red', 'GRPO': 'tab:blue', 'DAPO': 'tab:orange', 'GRPO+A': 'tab:green'})
    metrics = [('reward', 'Logged mean rollout reward (different reward / filtering definitions)', 'reward'),
               ('acc', 'Task accuracy on generated training trajectories', 'correct / trajectory'),
               ('entropy', 'Sampled token surprisal (actor/entropy_loss; not full-vocabulary entropy)', 'nats / token'),
               ('ppo_kl', 'Policy diff: update vs rollout policy (logged estimate, may be negative)', 'actor/ppo_kl'),
               ('grad_norm', 'Actor gradient norm before clipping (log reductions differ)', 'gradient norm'),
               ('crops', 'Executed crops per generated trajectory', 'crops / trajectory'),
               ('gen_tok', 'Generated policy tokens per trajectory', 'tokens / trajectory'),
               ('adv_nz', 'Response tokens with nonzero advantage in the update batch', 'share of response tokens')]
    for index, (metric, title, unit) in enumerate(metrics, 1):
        fig, ax = plt.subplots(figsize=(15, 7), layout='constrained')
        missing = []
        for name in names:
            values = data.get((name, metric), {})
            if not values:
                missing.append(name)
                continue
            xs = sorted(values)
            main = name in ('strat', 'GRPO', 'DAPO')
            ax.plot(xs, [values[x] for x in xs], color=colors[name], label=name,
                    lw=3 if name == 'strat' else 1.8 if main else 1.0,
                    alpha=1 if main else .4, zorder=5 if main else 1)
        ax.set(title=title, xlabel='RL update step (raw observations; no smoothing)', ylabel=unit)
        ax.grid(alpha=.2)
        ax.legend(loc='center left', bbox_to_anchor=(1.01, .5), fontsize=9)
        if missing:
            ax.text(.01, .02, 'Not measured: ' + ', '.join(missing), transform=ax.transAxes, fontsize=8)
        strat = data.get(('strat', metric), {})
        end = f"strat last = {strat[max(strat)]:.5g}" if strat else 'strat not measured'
        fig.suptitle(f'{end} | strat vs completed RL family\n'
                     'Qwen3-VL-4B / A-SFT | 32 steps, seed 42, lr 1e-6 | protocol/reward/filter differences retained')
        save(fig, root, f'family_{index:02d}_{metric}')
    comparison = json.loads((root / 'comparison.json').read_text())
    legacy = json.loads((root / 'family_e1.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(19, 8), layout='constrained')
    order = sorted(legacy, key=lambda x: x['acc'])
    for i, row in enumerate(order):
        name = row['name']
        color = colors.get(name, 'tab:gray')
        axes[0].barh(i, row['acc'] * 100, color=color, alpha=1 if name == 'strat' else .65)
        axes[0].text(row['acc'] * 100 + .12, i, f"{row['acc']*100:.2f}", va='center', fontsize=9)
        axes[1].scatter(row['tokens'], row['acc'] * 100, color=color, s=130 if name == 'strat' else 55)
        offsets = {'GSPO': (-35, -15), 'G1 R-read': (-30, 16), 'A+B+C': (8, -12),
                   'GRPO+A': (-65, 8), 'A+B': (-20, 10)}
        axes[1].annotate(name, (row['tokens'], row['acc']*100), xytext=offsets.get(name, (5, (i % 3-1)*10)),
                         textcoords='offset points', fontsize=8)
    axes[0].set(yticks=range(len(order)), yticklabels=[r['name'] for r in order],
                xlabel='E1 accuracy (%)', title='Fixed E1: failures stay in denominator', xlim=(55, 68))
    axes[1].set(xlabel='Processed input + output tokens / trajectory', ylabel='E1 accuracy (%)',
                title='Completed family: accuracy vs inference cost')
    for ax in axes:
        ax.grid(alpha=.2)
    headline = 'E1: ' + ' | '.join(f"{name} {row['correct']/row['n']*100:.2f}%" for name, row in comparison['summary'].items())
    fig.suptitle(headline + '\n552 questions x 4 samples; step 32 | Historical entries retain released rounding; no training-seed interval')
    save(fig, root, 'family_e1')
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), layout='constrained')
    summaries = comparison['summary']
    for ax, metric, unit in zip(axes, ('first', 'later', 'tokens'),
                                ('First crop indicator / trajectory', 'Later crops / trajectory', 'Processed tokens / trajectory')):
        for name, row in summaries.items():
            ax.scatter(row[metric], row['correct']/row['n']*100, color=colors[name], s=130, label=name)
            ax.annotate(name, (row[metric], row['correct']/row['n']*100), xytext=(4, 5), textcoords='offset points', fontsize=9)
        ax.set(xlabel=unit, ylabel='E1 accuracy (%)')
        ax.grid(alpha=.2)
    axes[0].legend(loc='lower left')
    focus = summaries[comparison['reference']]
    fig.suptitle(f"{comparison['reference']}: first crop {focus['first']:.4f} | later crops {focus['later']:.4f} | processed tokens {focus['tokens']:,.0f}\n"
                 'Same 2,208 paired E1 keys; costs sum all turns, including repeated history and image tokens')
    save(fig, root, 'direct_e1_cost')


if __name__ == '__main__':
    main()
