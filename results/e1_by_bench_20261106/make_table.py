"""从 e1_by_bench.json 生成 README 的分 benchmark 表（行按底座分组，不按分数排序）。"""
import json
import sys

ORDER = ['A-SFT', 'GRPO', 'DAPO', 'GSPO', 'GiGPO', 'RLOO', 'SAPO',
         'DTPO', 'G1', 'A+B', 'A+B+C', 'GRPO+A', 'strat', 'max2', 'Htau', 'DAPO-SB', 'RLOO+first-action']
NAMES = {'Htau': 'Hτ', 'RLOO+first-action': 'RLOO + 首动作分叉', 'A-SFT': 'A-SFT 起点'}
GROUP = {'A-SFT': '起点', 'GRPO': '优化器', 'DAPO': '优化器', 'GSPO': '优化器', 'GiGPO': '优化器', 'RLOO': '优化器',
         'SAPO': '优化器', 'Htau': 'DAPO 底座', 'DAPO-SB': 'DAPO 底座', 'RLOO+first-action': 'RLOO 底座'}

d = json.load(open(sys.argv[1]))
S = d['sources']
c = {e['name']: e for e in d['conditions']}
best = {s: max(c[n]['by_source'][s] for n in ORDER if n != 'A-SFT') for s in S}
best['macro'] = max(c[n]['macro_mean'] for n in ORDER if n != 'A-SFT')
nq = d['questions_per_source']
print(f"| 方法 | 类别 | 对照底座 | MME-RealWorld（{nq['mme_realworld']} 题） | V\\*（{nq['vstar']}） | WorldBench（{nq['worldbench']}） "
      f"| MMMU-Pro（{nq['mmmu_pro']}） | **四个 bench 不加权平均** | 平均相对底座 [95% 区间] |")
print('|---|---|---|---:|---:|---:|---:|---:|---|')
for n in ORDER:
    e = c[n]; vb = e.get('vs_base')
    cells = []
    for s in S:
        v = e['by_source'][s] * 100
        txt = f"**{v:.1f}**" if abs(e['by_source'][s] - best[s]) < 1e-12 else f"{v:.1f}"
        if vb:
            dd = vb[s]; star = '\\*' if dd['ci95'][0] > 0 or dd['ci95'][1] < 0 else ''
            txt += f"（{dd['delta_pp']:+.1f}{star}）"
        cells.append(txt)
    m = e['macro_mean'] * 100
    mt = f"**{m:.2f}**" if abs(e['macro_mean'] - best['macro']) < 1e-12 else f"{m:.2f}"
    delta = (f"{vb['macro']['delta_pp']:+.2f} [{vb['macro']['ci95'][0]:+.2f}, {vb['macro']['ci95'][1]:+.2f}]"
             + ('\\*' if vb['macro']['ci95'][0] > 0 or vb['macro']['ci95'][1] < 0 else '')) if vb else '—'
    print(f"| {NAMES.get(n, n)} | {GROUP.get(n, 'GRPO 底座')} | {e['base'] or '—'} | " + ' | '.join(cells) + f" | {mt} | {delta} |")
