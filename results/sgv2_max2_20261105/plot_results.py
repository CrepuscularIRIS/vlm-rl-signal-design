"""从冻结长表与 E1 聚合重画 max2 的全部图；只读发布目录内文件，不插补缺测。

用法：python plot_results.py --data DIR
输入：family_long.csv、comparison.json、family_e1.json、grpo_gap.json（后三者缺失时跳过对应图）。
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

FOCUS = 'max2'
KEY_RUNS = ('GRPO', 'GRPO+A', 'DAPO', 'strat', FOCUS)   # 加粗、着重色；其余为背景
COLORS = {  # 每个条件一个固定且互不相同的颜色
    'max2': '#d62728', 'GRPO': '#1f77b4', 'DAPO': '#ff7f0e', 'GRPO+A': '#2ca02c', 'strat': '#9467bd',
    'A+B': '#17becf', 'A+B+C': '#9edae5', 'DTPO': '#8c564b', 'Htau': '#e377c2', 'GSPO': '#bcbd22',
    'GiGPO': '#7f7f7f', 'G1 R-read': '#c49c94', 'RLOO': '#dbdb8d', 'RLOO-decision': '#f7b6d2',
    'SAPO': '#c7c7c7', 'DAPO-SB': '#ffbb78', 'A-SFT (start)': '#000000',
}
METRICS = (
    ('reward', 'Logged reward; reward / filtering definitions differ', 'reward / trajectory'),
    ('acc', 'Task correctness on generated training trajectories', 'correct / trajectory'),
    ('entropy', 'Sampled token surprisal; not full-vocabulary entropy', 'nats / token'),
    ('ppo_kl', 'Policy diff: update vs rollout policy; logged estimate', 'actor/ppo_kl'),
    ('grad_norm', 'Actor gradient norm before clipping', 'gradient norm'),
    ('crops', 'Executed crops on generated training trajectories', 'crops / trajectory'),
    ('gen_tok', 'Generated policy tokens', 'tokens / trajectory'),
    ('adv_nz', 'Nonzero advantage in actual update batch', 'share of response tokens'),
)
DIAGNOSTICS = METRICS[:4] + (('dev', 'Logged dev validation; not E1', 'reward score'),
                             ('clip_high', 'Upper clipping fraction in actual actor updates', 'fraction'))
SOURCE_LABELS = {'mme_realworld': 'MME-RealWorld', 'vstar': 'V*', 'worldbench': 'WorldBench', 'mmmu_pro': 'MMMU-Pro'}
SOURCE_COLORS = {'mme_realworld': '#440154', 'vstar': '#31688e', 'worldbench': '#35b779', 'mmmu_pro': '#fde725'}  # 与条件色不重
SUBTITLE = 'A-SFT Qwen3-VL-4B; n=8, seed=42, lr=1e-6; historical protocol / reward differences retained'


def save(fig: plt.Figure, out: Path, name: str) -> None:
    """保存同源 PNG 与 PDF。"""
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'{name}.{ext}', dpi=180)
    plt.close(fig)


def read_curves(root: Path) -> dict[tuple[str, str], dict[int, float]]:
    """长表 → {(run, metric): {step: value}}；重复 (run, metric, step) 直接报错。"""
    values: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    with (root / 'family_long.csv').open() as stream:
        for row in csv.DictReader(stream):
            key, step = (row['run'], row['metric']), int(row['step'])
            assert step not in values[key], (key, step)
            values[key][step] = float(row['value'])
    return values


def plot_line(ax: plt.Axes, points: dict[int, float], name: str) -> None:
    """一条原值曲线；重点条件加粗。"""
    xs = sorted(points)
    key = name in KEY_RUNS
    ax.plot(xs, [points[x] for x in xs], 'o--' if len(xs) < 10 else '-', label=name, color=COLORS[name],
            lw=3 if name == FOCUS else 2 if key else 1.1, alpha=1 if key else .5, zorder=3 if key else 1)


def family_figures(root: Path, values: dict) -> None:
    """每个指标一张 family 图：重点条件在前、背景条件在后。"""
    names = sorted({run for run, _ in values}, key=lambda n: (n not in KEY_RUNS, n))
    for index, (metric, title, unit) in enumerate(METRICS, 1):
        fig, ax = plt.subplots(figsize=(15, 7), layout='constrained')
        missing = [n for n in names if not values.get((n, metric))]
        for name in names:
            if values.get((name, metric)):
                plot_line(ax, values[name, metric], name)
        ax.set(title=title, xlabel='RL update step; raw values, no smoothing', ylabel=unit)
        ax.legend(loc='center left', bbox_to_anchor=(1.01, .5), fontsize=8, title='bold = direct comparisons')
        ax.grid(alpha=.2)
        if missing:
            ax.text(.01, .02, 'Not measured: ' + ', '.join(missing), transform=ax.transAxes, fontsize=8)
        fig.suptitle(f'max2 step 32 = {values[FOCUS, metric][32]:.6g}\n{SUBTITLE}')
        save(fig, root, f'family_{index:02d}_{metric}')


def diagnostics_figure(root: Path, values: dict) -> None:
    """max2 与直接对照的 2×3 诊断图；未测条件写在面板内。"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), layout='constrained')
    for ax, (metric, title, unit) in zip(axes.flat, DIAGNOSTICS):
        absent = []
        for name in KEY_RUNS:
            points = values.get((name, metric))
            if not points:
                absent.append(name)
                continue
            plot_line(ax, points, name)
            last = max(points)
            ax.annotate(f'{points[last]:.4g}', (last, points[last]), color=COLORS[name], fontsize=7)
        if absent:
            ax.text(.02, .03, 'Not measured: ' + ', '.join(absent), transform=ax.transAxes, fontsize=7)
        ax.set(title=title, xlabel='Logged update step', ylabel=unit)
        ax.grid(alpha=.2)
    axes.flat[0].legend()
    fig.suptitle('max2 training and optimization diagnostics vs direct comparisons\n'
                 'Raw logged values; missing observations are not interpolated; dev points are not E1 scores')
    save(fig, root, 'max2_training_diagnostics')


def family_e1_figure(root: Path) -> None:
    """全部条件的 E1 准确率条形图与准确率–token 散点。"""
    family = sorted(json.loads((root / 'family_e1.json').read_text()), key=lambda r: r['acc'])
    best = family[-1]['name']
    fig, axes = plt.subplots(1, 2, figsize=(19, 8), layout='constrained')
    for i, row in enumerate(family):
        name, acc = row['name'], row['acc'] * 100
        color = COLORS.get(name, 'tab:gray')
        axes[0].barh(i, acc, color=color, alpha=1 if name in KEY_RUNS else .55,
                     edgecolor='black' if name == FOCUS else 'none', lw=1.5)
        axes[0].text(acc + .12, i, f'{acc:.2f}' + (' (best)' if name == best else ''), va='center', fontsize=8,
                     fontweight='bold' if name in (best, FOCUS) else 'normal')
        axes[1].scatter(row['tokens'], acc, color=color, s=140 if name == FOCUS else 55,
                        edgecolor='black' if name == FOCUS else 'none', zorder=3)
        axes[1].annotate(name, (row['tokens'], acc), xytext=(5, (i % 3 - 1) * 9), textcoords='offset points',
                         fontsize=8, fontweight='bold' if name in KEY_RUNS else 'normal')
    axes[0].set(yticks=range(len(family)), yticklabels=[r['name'] for r in family], xlabel='E1 accuracy (%)',
                xlim=(55, 69), title='E1 accuracy, step 32 (black edge = max2)')
    axes[1].set(xlabel='Processed input + output tokens / trajectory', ylabel='E1 accuracy (%)',
                title='Accuracy vs processed-token cost')
    axes[1].margins(x=.08)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle('Fixed step-32 E1: 552 questions x 4; failures retained\n'
                 'Historical family entries retain their published rounding; paired arms use one scorer')
    save(fig, root, 'family_e1')


def cost_figure(root: Path, summaries: dict) -> None:
    """首裁、后续裁图、token 三项成本对 E1 准确率。"""
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), layout='constrained')
    panels = (('first', 'First crop indicator / trajectory'), ('later', 'Later crops / trajectory'),
              ('tokens', 'Processed tokens / trajectory'))
    for ax, (metric, label) in zip(axes, panels):
        for name, row in summaries.items():
            acc = row['correct'] / row['n'] * 100
            ax.scatter(row[metric], acc, label=name, color=COLORS[name], s=120 if name == FOCUS else 55, zorder=3)
            ax.annotate(name, (row[metric], acc), xytext=(4, 4), textcoords='offset points', fontsize=8)
        ax.set(xlabel=label, ylabel='E1 accuracy (%)')
        ax.grid(alpha=.2)
    axes[0].legend()
    focus = summaries[FOCUS]
    fig.suptitle(f"max2 {focus['correct']}/{focus['n']}; first crop {focus['first']:.4f}, "
                 f"later crops {focus['later']:.4f}\nSame paired keys; token cost includes repeated history and image tokens")
    save(fig, root, 'direct_e1_cost')


def gap_figure(root: Path, comparison: dict) -> None:
    """max2 对 GRPO：裁图数分布、按自身裁图数的准确率、按 GRPO 裁图数分档的分源净差。"""
    gap = json.loads((root / 'grpo_gap.json').read_text())
    ref, ctrl = gap['ref'], gap['ctrl']
    buckets = list(gap['crop_hist'][ref])
    x = range(len(buckets))
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), layout='constrained')
    for offset, name in ((-.2, ctrl), (.2, ref)):
        hist = gap['crop_hist'][name]
        axes[0].bar([i + offset for i in x], [hist[b] / gap['n'] * 100 for b in buckets], .4,
                    color=COLORS[name], label=name)
        acc = gap['acc_by_own_crops'][name]
        kept = [b for b in buckets if acc.get(b, {}).get('n', 0) >= 20]
        axes[1].plot([buckets.index(b) for b in kept], [acc[b]['correct'] / acc[b]['n'] * 100 for b in kept],
                     'o-', color=COLORS[name], lw=2.5 if name == ref else 2, label=name)
        for b in kept:
            axes[1].annotate(f"n={acc[b]['n']}", (buckets.index(b), acc[b]['correct'] / acc[b]['n'] * 100),
                             xytext=(4, -12 if name == ref else 6), textcoords='offset points', fontsize=7,
                             color=COLORS[name])
    net = gap['net_by_ctrl_crops']
    bottom_pos, bottom_neg = [0.0] * len(buckets), [0.0] * len(buckets)
    for source, label in SOURCE_LABELS.items():
        vals = [net[b]['net_by_source'][source] for b in buckets]
        base = [p if v >= 0 else n for v, p, n in zip(vals, bottom_pos, bottom_neg)]
        axes[2].bar(x, vals, .6, bottom=base, label=label, color=SOURCE_COLORS[source], edgecolor='white')
        bottom_pos = [p + max(v, 0) for v, p in zip(vals, bottom_pos)]
        bottom_neg = [n + min(v, 0) for v, n in zip(vals, bottom_neg)]
    for i, b in enumerate(buckets):
        axes[2].annotate(f"net {net[b]['net']:+d}\n(n={net[b]['n']})", (i, max(bottom_pos[i], 0) + .5),
                         ha='center', fontsize=8)
    axes[2].axhline(0, color='black', lw=.8)
    axes[2].margins(y=.2)
    axes[0].set(title='Executed crops per trajectory (share of 2,208)', xlabel='crops', ylabel='% of trajectories')
    axes[1].set(title='E1 accuracy by own crop count (buckets with n >= 20)', xlabel='crops', ylabel='accuracy (%)')
    axes[2].set(title=f'Net correct ({ref} - {ctrl}), same keys, by {ctrl} crop count',
                xlabel=f'{ctrl} crops on that key', ylabel='net correct trajectories')
    for ax in axes:
        ax.set_xticks(list(x), buckets)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    paired = comparison['paired'][ctrl]
    lo, hi = paired['question_bootstrap_95_pp']
    fig.suptitle(f"{ref} vs {ctrl} on E1: {gap['correct'][ref]} vs {gap['correct'][ctrl]} / {gap['n']} "
                 f"({paired['delta_pp']:+.2f} pp, question-bootstrap 95% CI [{lo:+.2f}, {hi:+.2f}])\n"
                 'Same uid + sample_idx keys; crop buckets are descriptive counts, not causal attribution')
    save(fig, root, 'max2_vs_grpo_gap')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, required=True)
    root = ap.parse_args().data
    values = read_curves(root)
    family_figures(root, values)
    diagnostics_figure(root, values)
    comparison_path = root / 'comparison.json'
    if not comparison_path.exists():
        print('E1 尚无统一评分结果：未画 E1 图')
        return
    comparison = json.loads(comparison_path.read_text())
    family_e1_figure(root)
    cost_figure(root, comparison['summary'])
    if (root / 'grpo_gap.json').exists():
        gap_figure(root, comparison)


if __name__ == '__main__':
    main()
