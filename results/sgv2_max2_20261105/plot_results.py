"""从冻结长表重画 max2 family 训练图及已完成的 E1；不插补缺测。"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


FOCUS = 'max2'
METRICS = [
    ('reward', 'Logged reward; reward / filtering definitions differ', 'reward / trajectory'),
    ('acc', 'Task correctness on generated training trajectories', 'correct / trajectory'),
    ('entropy', 'Sampled token surprisal; not full-vocabulary entropy', 'nats / token'),
    ('ppo_kl', 'Policy diff: update vs rollout policy; logged estimate', 'actor/ppo_kl'),
    ('grad_norm', 'Actor gradient norm before clipping', 'gradient norm'),
    ('crops', 'Executed crops on generated training trajectories', 'crops / trajectory'),
    ('gen_tok', 'Generated policy tokens', 'tokens / trajectory'),
    ('adv_nz', 'Nonzero advantage in actual update batch', 'share of response tokens'),
]


def save(fig: object, out: Path, name: str) -> None:
    """保存同源 PNG 与 PDF。"""
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'{name}.{ext}', dpi=180)
    plt.close(fig)


def training(root: Path) -> tuple[dict, dict]:
    """绘制历史 family 和直接对照，保留各条件的原始口径。"""
    values = defaultdict(dict)
    with (root / 'family_long.csv').open() as stream:
        for row in csv.DictReader(stream):
            key, step = (row['run'], row['metric']), int(row['step'])
            assert step not in values[key], (key, step)
            values[key][step] = float(row['value'])
    names = sorted({key[0] for key in values})
    colors = {name: plt.get_cmap('tab20')(i % 20) for i, name in enumerate(names)}
    colors.update({'max2': 'tab:red', 'strat': 'tab:purple', 'GRPO': 'tab:blue',
                   'DAPO': 'tab:orange', 'GRPO+A': 'tab:green'})
    for index, (metric, title, unit) in enumerate(METRICS, 1):
        fig, ax = plt.subplots(figsize=(15, 7), layout='constrained')
        missing = []
        for name in names:
            points = values.get((name, metric), {})
            if not points:
                missing.append(name)
                continue
            xs = sorted(points)
            ax.plot(xs, [points[x] for x in xs], label=name, color=colors[name],
                    lw=3 if name == FOCUS else 1.6, alpha=1 if name in [FOCUS, 'GRPO', 'DAPO', 'strat'] else .35)
        ax.set(title=title, xlabel='RL update step; no smoothing', ylabel=unit)
        ax.legend(loc='center left', bbox_to_anchor=(1.01, .5), fontsize=8)
        ax.grid(alpha=.2)
        if missing:
            ax.text(.01, .02, 'Not measured: ' + ', '.join(missing), transform=ax.transAxes, fontsize=8)
        end = values[FOCUS, metric][32]
        fig.suptitle(f'max2 step 32 = {end:.6g}\nA-SFT Qwen3-VL-4B; n=8, seed=42, lr=1e-6; historical protocol / reward differences retained')
        save(fig, root, f'family_{index:02d}_{metric}')
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), layout='constrained')
    panels = METRICS[:4] + [('dev', 'Logged dev validation; not E1', 'reward score'),
                          ('clip_high', 'Upper clipping fraction in actual actor updates', 'fraction')]
    for ax, (metric, title, unit) in zip(axes.flat, panels):
        for name in ['GRPO', 'DAPO', 'strat', FOCUS]:
            points = values.get((name, metric), {})
            if not points:
                ax.text(.02, .03 + .05 * ['GRPO', 'DAPO', 'strat', FOCUS].index(name),
                        name + ': not measured in this table', transform=ax.transAxes, fontsize=7)
                continue
            xs = sorted(points)
            ax.plot(xs, [points[x] for x in xs], label=name, color=colors[name], lw=2.6 if name == FOCUS else 1.5)
            ax.annotate(f'{points[xs[-1]]:.4g}', (xs[-1], points[xs[-1]]), fontsize=7)
        ax.set(title=title, xlabel='Logged update step', ylabel=unit)
        ax.grid(alpha=.2)
    axes.flat[0].legend()
    fig.suptitle('max2 training and optimization diagnostics\nMissing observations are not interpolated; dev points are not E1 scores')
    save(fig, root, 'max2_training_diagnostics')
    return values, colors


def evaluation(root: Path, colors: dict) -> None:
    """只在已有统一评分结果时绘制固定 E1。"""
    comparison = root / 'comparison.json'
    if not comparison.exists():
        print('E1尚无统一评分结果：未画E1图')
        return
    results = json.loads(comparison.read_text())
    summaries = results['summary']
    family = json.loads((root / 'family_e1.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(19, 8), layout='constrained')
    for i, row in enumerate(sorted(family, key=lambda r: r['acc'])):
        name = row['name']
        color = colors.get(name, 'tab:gray')
        axes[0].barh(i, row['acc'] * 100, color=color, alpha=1 if name == FOCUS else .65)
        axes[0].text(row['acc'] * 100 + .12, i, f"{row['acc']*100:.2f}", va='center', fontsize=8)
        axes[1].scatter(row['tokens'], row['acc'] * 100, color=color, s=130 if name == FOCUS else 50)
        axes[1].annotate(name, (row['tokens'], row['acc'] * 100), xytext=(4, (i % 3 - 1) * 10), textcoords='offset points', fontsize=8)
    order = sorted(family, key=lambda r: r['acc'])
    axes[0].set(yticks=range(len(order)), yticklabels=[r['name'] for r in order], xlabel='E1 accuracy (%)', xlim=(55, 69))
    axes[1].set(xlabel='Processed input + output tokens / trajectory', ylabel='E1 accuracy (%)')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle('Fixed step-32 E1: 552 questions x 4; failures retained\nHistorical family entries retain their published rounding; paired arms use one scorer')
    save(fig, root, 'family_e1')
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), layout='constrained')
    for ax, metric, label in zip(axes, ['first', 'later', 'tokens'], ['First crop indicator / trajectory', 'Later crops / trajectory', 'Processed tokens / trajectory']):
        for name, row in summaries.items():
            ax.scatter(row[metric], row['correct'] / row['n'] * 100, label=name, color=colors[name])
            ax.annotate(name, (row[metric], row['correct'] / row['n'] * 100), xytext=(4, 4), textcoords='offset points', fontsize=8)
        ax.set(xlabel=label, ylabel='E1 accuracy (%)')
        ax.grid(alpha=.2)
    axes[0].legend()
    focus = summaries[FOCUS]
    fig.suptitle(f"max2 {focus['correct']}/{focus['n']}; first crop {focus['first']:.4f}, later crops {focus['later']:.4f}\nSame paired keys; token cost includes repeated history and image tokens")
    save(fig, root, 'direct_e1_cost')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    root = ap.parse_args().data
    _, colors = training(root)
    evaluation(root, colors)


if __name__ == '__main__':
    main()
