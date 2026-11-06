"""E1 按 benchmark 分开评：每个来源单独的准确率、四个来源的不加权平均，以及相对各自优化器底座的配对差。

依据导师 2026-11-05：“算平均分（评估的时候）可以不用加权”；“假设我们的optimizer用的是grpo……那我们超过grpo就可以，对于dapo也是同理”。

- 正确判定与统一评分器 summarize.py 相同（从 final_text 抽答案，error/截断记错，失败留分母）。
- 配对按 (uid, sample_idx)；区间：每个来源内按题重采样（分层），10,000 次，seed 20261105；
  不加权平均的差 = 四个来源差值的平均，与分来源差值用同一组重采样。
- 输出只含聚合数（不含 uid、题面、图像）。

用法：python e1_by_bench.py --scorer SUMMARIZE_PY --conditions CONDITIONS.json --out OUT.json
CONDITIONS.json: [{"name":..., "base":..., "family":..., "traj":..., "published_correct": int|null}, ...]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SOURCES = ('mme_realworld', 'vstar', 'worldbench', 'mmmu_pro')
DRAWS, SEED = 10_000, 20261105


def load_scorer(path: Path):
    spec = importlib.util.spec_from_file_location('summarize', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load(path: Path, scorer) -> dict[tuple[str, int], tuple[str, int]]:
    """(uid, sample_idx) → (source, correct)，判定与 summarize.summarize 一致。"""
    rows = {}
    for line in path.open():
        r = json.loads(line)
        pred, _ = scorer.extract(r.get('final_text') or '')
        ok = int(pred is not None and pred == r['gt'] and not r.get('error') and not scorer.truncated(r))
        key = (r['uid'], r['sample_idx'])
        assert key not in rows, key
        rows[key] = (r['source'], ok)
    return rows


def per_question(rows: dict) -> dict[str, dict[str, float]]:
    """source → uid → 该题 4 次采样的正确率。"""
    acc = defaultdict(lambda: defaultdict(list))
    for (uid, _), (src, ok) in rows.items():
        acc[src][uid].append(ok)
    return {s: {u: float(np.mean(v)) for u, v in acc[s].items()} for s in SOURCES}


def summary(rows: dict) -> dict:
    q = per_question(rows)
    by = {s: float(np.mean(list(q[s].values()))) for s in SOURCES}
    counts = {s: {'correct': int(sum(ok for (src, ok) in rows.values() if src == s)),
                  'n': sum(1 for (src, _) in rows.values() if src == s)} for s in SOURCES}
    return {'by_source': by, 'counts': counts, 'macro_mean': float(np.mean([by[s] for s in SOURCES])),
            'pooled_correct': int(sum(ok for _, ok in rows.values())), 'n': len(rows)}


def paired(new: dict, base: dict, rng_draws: dict) -> dict:
    """分来源配对差与不加权平均差（百分点），按题分层重采样区间。"""
    assert new.keys() == base.keys(), 'paired keys differ'
    qn, qb = per_question(new), per_question(base)
    out, macro_draws = {}, np.zeros(DRAWS)
    for s in SOURCES:
        uids = sorted(qn[s])
        d = np.array([qn[s][u] - qb[s][u] for u in uids]) * 100
        draws = d[rng_draws[s]].mean(axis=1)
        macro_draws += draws / len(SOURCES)
        lo, hi = np.quantile(draws, [.025, .975])
        out[s] = {'delta_pp': float(d.mean()), 'ci95': [float(lo), float(hi)], 'questions': len(uids)}
    lo, hi = np.quantile(macro_draws, [.025, .975])
    out['macro'] = {'delta_pp': float(np.mean([out[s]['delta_pp'] for s in SOURCES])), 'ci95': [float(lo), float(hi)]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scorer', type=Path, required=True)
    ap.add_argument('--conditions', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    scorer = load_scorer(args.scorer)
    conds = json.loads(args.conditions.read_text())
    rows = {c['name']: load(Path(c['traj']), scorer) for c in conds}
    ref = next(iter(rows.values()))
    q_per_source = {s: len({u for (u, _), (src, _) in ref.items() if src == s}) for s in SOURCES}
    rng = np.random.default_rng(SEED)
    rng_draws = {s: rng.integers(0, q_per_source[s], size=(DRAWS, q_per_source[s])) for s in SOURCES}
    result = {'sources': SOURCES, 'questions_per_source': q_per_source, 'bootstrap': {'draws': DRAWS, 'seed': SEED,
              'unit': 'question, stratified by source'}, 'conditions': []}
    for c in conds:
        s = summary(rows[c['name']])
        if c.get('published_correct') is not None:
            assert s['pooled_correct'] == c['published_correct'], (c['name'], s['pooled_correct'], c['published_correct'])
        entry = {k: c[k] for k in ('name', 'base', 'family')} | s
        if c['base'] and c['base'] in rows:
            entry['vs_base'] = paired(rows[c['name']], rows[c['base']], rng_draws)
        result['conditions'].append(entry)
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    for e in result['conditions']:
        vb = e.get('vs_base', {})
        print(f"{e['name']:16s} macro {e['macro_mean']*100:6.2f} | " +
              ' '.join(f"{s[:5]} {e['by_source'][s]*100:5.1f}" for s in SOURCES) +
              (f" | vs {e['base']}: macro {vb['macro']['delta_pp']:+.2f} [{vb['macro']['ci95'][0]:+.2f},{vb['macro']['ci95'][1]:+.2f}] " +
               ' '.join(f"{s[:5]} {vb[s]['delta_pp']:+.1f}{'*' if vb[s]['ci95'][0] > 0 or vb[s]['ci95'][1] < 0 else ''}" for s in SOURCES)
               if vb else ''))


if __name__ == '__main__':
    main()
