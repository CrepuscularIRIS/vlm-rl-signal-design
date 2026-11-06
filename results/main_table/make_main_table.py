"""Generate the README main table: every E1 condition, per-benchmark accuracy, unweighted mean of 4, pooled score,
difference vs its own optimizer base (unweighted, 95% CI), and cost columns. No hand-copied numbers.

usage: python make_main_table.py ../entry_ab_20261106/e1_by_bench_RP.json costs.csv > main_table.md
- accuracy and paired deltas: e1_by_bench_RP.json (unified scorer; question bootstrap stratified by source, 10,000
  draws, seed 20261105); '*' = 95% CI excludes 0.
- costs.csv: rollout multiple vs GRPO and E1 crops / generated tokens / processed tokens per trajectory (from the
  per-run reports; see the source column).
Bold = best in that accuracy column among trained conditions (A-SFT excluded). Rows grouped by base, not sorted.
"""
import csv
import json
import sys

SOURCES = ('mme_realworld', 'vstar', 'worldbench', 'mmmu_pro')


def main(e1_path: str, costs_path: str) -> None:
    e1 = {c['name']: c for c in json.load(open(e1_path))['conditions']}
    costs = list(csv.DictReader(open(costs_path)))
    trained = [r['key'] for r in costs if r['key'] != 'A-SFT']
    best = {s: max(e1[k]['by_source'][s] for k in trained) for s in SOURCES}
    best['macro'] = max(e1[k]['macro_mean'] for k in trained)
    best['pooled'] = max(e1[k]['pooled_correct'] for k in trained)
    b = lambda v, ref, txt: f'**{txt}**' if abs(v - ref) < 1e-12 else txt
    print('| 方法 | 底座 / 改动层 | rollout | MME-RW | V\\* | WorldBench | MMMU-Pro | **4 bench 不加权** | 相对底座（不加权）[95% 区间] '
          '| 合并 /2,208 | 裁图 / 条 | 生成 token / 条 | 处理 token / 条 |')
    print('|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|')
    for r in costs:
        c = e1[r['key']]
        cells = [b(c['by_source'][s], best[s], f"{c['by_source'][s] * 100:.1f}") for s in SOURCES]
        macro = b(c['macro_mean'], best['macro'], f"{c['macro_mean'] * 100:.2f}")
        vb = c.get('vs_base', {}).get('macro')
        delta = '—' if not vb else (f"{vb['delta_pp']:+.2f} [{vb['ci95'][0]:+.2f}, {vb['ci95'][1]:+.2f}]"
                                    + ('\\*' if vb['ci95'][0] > 0 or vb['ci95'][1] < 0 else ''))
        pooled = str(c['pooled_correct']) if r['key'] == 'A-SFT' else b(c['pooled_correct'], best['pooled'], str(c['pooled_correct']))
        print(f"| {r['label']} | {r['layer']} | {r['rollout']} | " + ' | '.join(cells) +
              f" | {macro} | {delta} | {pooled} | {r['crops']} | {r['gen']} | {r['proc']} |")


if __name__ == '__main__':
    main(*sys.argv[1:3])
