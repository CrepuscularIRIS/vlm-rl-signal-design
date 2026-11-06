"""R/P 发布包：训练长表（与 family_long.csv 同口径，追加 R、P）、训练对照图、E1 分 benchmark 图。

用法：python build_publication.py --data DATA_DIR --family FAMILY_LONG_CSV --e1 E1_BY_BENCH_JSON --parser TRAIN_CURVES_PY --out OUT
- 训练正确率、裁图、生成 token：训练账本 rollouts.jsonl 全部生成轨迹的均值（P 含追加组）。
- reward：日志 reward/overall（本步新采样）；entropy、ppo_kl、grad_norm：日志 actor/*；adv_nz：updates.jsonl。
- update_mixed：每步进入更新的混合组数（R/P 取 entry.jsonl；GRPO 取 updates.jsonl；DAPO 每步 128）。
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS = {'R': 'replay32', 'P': 'pilot32'}
COLORS = {'GRPO': '#1f77b4', 'DAPO': '#ff7f0e', 'R': '#d62728', 'P': '#2ca02c'}
LABELS = {'GRPO': 'GRPO (1x rollout)', 'DAPO': 'DAPO (1.92x)', 'R': 'R: replay fill (1x)', 'P': 'P: extra group (1.56x)'}
METRICS = [('update_mixed', 'Mixed groups entering each update (of 128)', 'groups / step'),
           ('reward', 'Logged reward on freshly sampled batch', 'reward'),
           ('acc', 'Task correctness on generated trajectories', 'correct / trajectory'),
           ('entropy', 'Sampled token surprisal (not full-vocab entropy)', 'nats / token'),
           ('ppo_kl', 'Policy diff: actor/ppo_kl (logged estimate)', 'ppo_kl'),
           ('grad_norm', 'Gradient norm before clipping (clip 1.0)', 'grad norm'),
           ('crops', 'Executed crops on generated trajectories', 'crops / trajectory'),
           ('gen_tok', 'Generated tokens on generated trajectories', 'tokens / trajectory')]
SOURCES = {'mme_realworld': 'MME-RealWorld', 'vstar': 'V*', 'worldbench': 'WorldBench', 'mmmu_pro': 'MMMU-Pro'}


def ledger_rows(run: str, d: Path, parse_log) -> list[dict]:
    """rollouts.jsonl 的 step 是奖励调用序号：R 每步 1 次；P 每步 2 次（第一轮、追加组），映射回训练步。"""
    calls_per_step = 2 if run == 'P' else 1
    per = defaultdict(lambda: defaultdict(list))
    for r in map(json.loads, (d / 'rollouts.jsonl').open()):
        s = per[(r['step'] + calls_per_step - 1) // calls_per_step]
        s['acc'].append(float(r['terminal'] == 'answer' and r['correct'] >= 1))
        s['crops'].append(r['crops'])
        s['gen_tok'].append(r['gen_tokens'])
    rows = [{'run': run, 'step': st, 'metric': m, 'value': sum(v) / len(v), 'source': 'rollouts.jsonl'}
            for st, s in per.items() for m, v in s.items()]
    for i, u in enumerate(map(json.loads, (d / 'updates.jsonl').open()), 1):
        rows.append({'run': run, 'step': i, 'metric': 'adv_nz', 'value': u['adv_nonzero_token_share'], 'source': 'updates.jsonl'})
    for e in map(json.loads, (d / 'entry.jsonl').open()):
        rows.append({'run': run, 'step': e['step'], 'metric': 'update_mixed', 'value': e['update_mixed'], 'source': 'entry.jsonl'})
    log = parse_log(str(next(d.glob('*.log'))))
    keys = {'reward': 'reward/overall', 'entropy': 'actor/entropy_loss', 'ppo_kl': 'actor/ppo_kl',
            'grad_norm': 'actor/grad_norm', 'clip_high': 'actor/pg_clipfrac_higher'}
    for st, m in log.items():
        for name, k in keys.items():
            if st >= 1 and k in m:
                rows.append({'run': run, 'step': st, 'metric': name, 'value': m[k], 'source': 'stdout'})
        if 'val/reward_score' in m:
            rows.append({'run': run, 'step': st, 'metric': 'dev', 'value': m['val/reward_score'], 'source': 'stdout'})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('data', 'family', 'e1', 'parser', 'out'):
        ap.add_argument(f'--{k}', type=Path, required=True)
    ap.add_argument('--grpo-updates', type=Path, required=True)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location('tc', a.parser)
    tc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tc)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(a.family.open()))
    assert not any(r['run'] in RUNS for r in rows)
    for name, sub in RUNS.items():
        rows += ledger_rows(name, a.data / sub, tc.parse_log)
    for i, u in enumerate(map(json.loads, a.grpo_updates.open()), 1):
        rows.append({'run': 'GRPO', 'step': i, 'metric': 'update_mixed', 'value': u['groups']['mixed'], 'source': 'updates.jsonl'})
        rows.append({'run': 'DAPO', 'step': i, 'metric': 'update_mixed', 'value': 128, 'source': 'filter rule (all kept groups mixed)'})
    keys = [(r['run'], r['metric'], int(r['step'])) for r in rows]
    assert len(keys) == len(set(keys)), 'duplicate (run, metric, step)'
    with (a.out / 'family_long.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['run', 'step', 'metric', 'value', 'source'])
        w.writeheader()
        w.writerows(rows)
    vals = defaultdict(dict)
    for r in rows:
        vals[r['run'], r['metric']][int(r['step'])] = float(r['value'])

    fig, axes = plt.subplots(2, 4, figsize=(24, 10), layout='constrained')
    for ax, (m, title, unit) in zip(axes.flat, METRICS):
        for run in ('GRPO', 'DAPO', 'R', 'P'):
            pts = vals.get((run, m))
            if not pts:
                ax.text(.02, .03, f'{run}: not measured', transform=ax.transAxes, fontsize=7)
                continue
            xs = sorted(x for x in pts if x >= 1)
            ax.plot(xs, [pts[x] for x in xs], color=COLORS[run], lw=2.4 if run in ('R', 'P') else 1.6, label=LABELS[run])
        ax.set(title=title, xlabel='RL update step (raw)', ylabel=unit)
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Update-entry ablation on GRPO: R replays last step\'s mixed groups (1x rollout); P samples one extra group for mixed questions (1.56x)\n'
                 'Same start, pool, seed, steps, lr, clip and reward as GRPO; advantage unchanged (group z-score)')
    for ext in ('png', 'pdf'):
        fig.savefig(a.out / f'entry_training.{ext}', dpi=180)
    plt.close(fig)

    e1 = json.loads(a.e1.read_text())
    cond = {c['name']: c for c in e1['conditions']}
    names = [('GRPO', 'GRPO'), ('DAPO', 'DAPO'), ('R', 'R-replay'), ('P', 'P-pilot')]
    fig, axes = plt.subplots(1, 2, figsize=(20, 6.5), layout='constrained', gridspec_kw={'width_ratios': [3, 1.2]})
    width = .2
    for i, (short, key) in enumerate(names):
        c = cond[key]
        ys = [c['by_source'][s] * 100 for s in SOURCES] + [c['macro_mean'] * 100]
        xs = [j + (i - 1.5) * width for j in range(len(ys))]
        axes[0].bar(xs, ys, width, color=COLORS[short], label=LABELS[short])
        for x, y in zip(xs, ys):
            axes[0].text(x, y + .4, f'{y:.1f}', ha='center', fontsize=7)
        pooled = c['pooled_correct'] / c['n'] * 100
        axes[1].bar(i, pooled, .6, color=COLORS[short])
        axes[1].text(i, pooled + .1, f'{pooled:.2f}', ha='center', fontsize=9)
    axes[0].set_xticks(range(5), list(SOURCES.values()) + ['Unweighted mean of 4'])
    axes[0].set(ylabel='E1 accuracy (%)', ylim=(25, 95), title='E1 by benchmark (552 questions x 4)')
    axes[0].legend(fontsize=8, loc='upper right')
    axes[1].set_xticks(range(4), [s for s, _ in names])
    axes[1].set(ylim=(62, 67.5), ylabel='E1 accuracy (%)', title='Pooled over 2,208 (question-count weighted)')
    for ax in axes:
        ax.grid(alpha=.2, axis='y')
    fig.suptitle('Fixed step-32 E1: GRPO 1411, R 1450, P 1438, DAPO 1465 of 2,208; failures retained\n'
                 'Paired 95% CIs (question bootstrap) are in REPORT.zh.md; R-GRPO pooled +1.77 [-0.23, +3.81]')
    for ext in ('png', 'pdf'):
        fig.savefig(a.out / f'entry_e1.{ext}', dpi=180)
    plt.close(fig)
    print('rows', len(rows), 'R', sum(r['run'] == 'R' for r in rows), 'P', sum(r['run'] == 'P' for r in rows))


if __name__ == '__main__':
    main()
