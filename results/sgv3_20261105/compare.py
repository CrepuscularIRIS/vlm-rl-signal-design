"""用既有统一评分器复算；输出匿名逐题结果、配对四格与按题 bootstrap。

用法：python compare.py --scorer summarize.py --out output --arm NAME traj.jsonl header.json [...]
输入原始轨迹留在私有目录；公开输出不含题面、图片、原始 uid 或机器路径。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np

HEADER_KEYS = ('perturb', 'samples', 'seed_base', 'n_questions', 'temperature', 'top_p',
               'max_new_tokens', 'max_gen_total', 'max_model_len', 'min_px', 'max_px',
               'max_crops', 'proto_module')


def four_cells(new: list[int], control: list[int]) -> dict:
    if len(new) != len(control):
        raise ValueError('配对长度不等')
    counts = Counter(zip(new, control))
    return dict(both_correct=counts[1, 1], new_only=counts[1, 0],
                control_only=counts[0, 1], both_wrong=counts[0, 0],
                net=sum(new) - sum(control))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scorer', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm', nargs=3, action='append', required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('fixed_scorer', args.scorer)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    summaries, headers, scores, provenance = {}, {}, {}, {}
    identity = None
    for name, raw, head in args.arm:
        if name in scores:
            raise ValueError(f'重复条件 {name}')
        path, header_path = Path(raw), Path(head)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        keyed = {(r['uid'], r['sample_idx']): r for r in rows}
        assert len(rows) == len(keyed) == 2208
        assert len({k[0] for k in keyed}) == 552
        assert set(Counter(k[0] for k in keyed).values()) == {4}
        assert {k[1] for k in keyed} == {0, 1, 2, 3}
        ids = {k: (r['gt'], r['source']) for k, r in keyed.items()}
        if identity is None:
            identity = ids
        assert ids == identity, f'{name}: 题目身份/GT/来源不一致'
        header = json.loads(header_path.read_text())
        headers[name] = {k: header.get(k) for k in HEADER_KEYS} | {'sha256': header['sha256']}
        values = {}
        for key, row in keyed.items():
            pred, fmt = scorer.extract(row.get('final_text') or '')
            values[key] = int(pred is not None and pred == row['gt'] and not row.get('error') and not scorer.truncated(row))
        scores[name] = values
        result = scorer.summarize(str(path))
        result.pop('file')
        result['correct'] = sum(values.values())
        result['by_source_counts'] = {s: {'correct': sum(values[k] for k in values if ids[k][1] == s),
                                                  'n': sum(ids[k][1] == s for k in values)}
                                      for s in sorted({r['source'] for r in rows})}
        result['counts'] = dict(error=sum(bool(r.get('error')) for r in rows),
                               truncated=sum(scorer.truncated(r) for r in rows),
                               zero_crop=sum(r['n_crops'] == 0 for r in rows),
                               missing_answer=sum(scorer.extract(r.get('final_text') or '')[0] is None for r in rows),
                               missing_answer_tag=sum(not scorer.extract(r.get('final_text') or '')[1] for r in rows))
        result['crops_exact'] = sum(r['n_crops'] for r in rows) / len(rows)
        result['tokens_total'] = sum(r['prompt_tokens'] + r['gen_tokens'] for r in rows)
        result['generated_tokens_total'] = sum(r['gen_tokens'] for r in rows)
        summaries[name] = result
        provenance[name] = {'trajectory_sha256': digest(path), 'header_sha256': digest(header_path)}
    ref = next(iter(headers))
    checks = {}
    for name, head in headers.items():
        settings = {k: {'new': headers[ref][k], 'control': head[k]} for k in HEADER_KEYS if head[k] != headers[ref][k]}
        hashes = {k: {'new': headers[ref]['sha256'].get(k), 'control': head['sha256'].get(k)}
                  for k in ('val', 'vstar_holdout', 'caption_env', 'script')
                  if head['sha256'].get(k) != headers[ref]['sha256'].get(k)}
        checks[name] = {'setting_differences': settings, 'data_code_hash_differences': hashes}
    uids = sorted({k[0] for k in identity})
    keys = [(u, s) for u in uids for s in range(4)]
    paired = {}
    rng = np.random.default_rng(20261105)
    # 题为重采样单位；这是统计重采样，不是新增训练 seed。
    draws = rng.integers(0, len(uids), size=(10000, len(uids)))
    for name in list(scores)[1:]:
        a, b = [scores[ref][k] for k in keys], [scores[name][k] for k in keys]
        delta = (np.array(a) - np.array(b)).reshape(552, 4).mean(axis=1)
        ci = np.quantile(delta[draws].mean(axis=1) * 100, [.025, .975])
        paired[name] = four_cells(a, b) | {'delta_pp': (sum(a)-sum(b))/2208*100,
                                           'question_bootstrap_95_pp': ci.tolist(),
                                           'by_source': {s: four_cells([scores[ref][k] for k in keys if identity[k][1] == s],
                                                                      [scores[name][k] for k in keys if identity[k][1] == s])
                                                         for s in sorted({v[1] for v in identity.values()})}}
    anonymous = [{'item_id': f'e1_{i:04d}', 'source': identity[u, 0][1],
                  'correct': {name: [values[u, s] for s in range(4)] for name, values in scores.items()}}
                 for i, u in enumerate(uids)]
    args.out.mkdir(parents=True, exist_ok=True)
    output = {'reference': ref, 'summary': summaries, 'paired': paired, 'header_checks': checks,
              'headers': headers, 'provenance': provenance, 'scorer_sha256': digest(args.scorer),
              'bootstrap': {'unit': 'question', 'draws': 10000, 'seed': 20261105},
              'acceptance': {'rows_per_arm': 2208, 'questions': 552, 'samples': [0, 1, 2, 3], 'identities_equal': True}}
    (args.out / 'comparison.json').write_text(json.dumps(output, indent=2) + '\n')
    (args.out / 'paired_questions.json').write_text(json.dumps(anonymous, indent=2) + '\n')
    print(json.dumps({'summary': summaries, 'paired': paired, 'header_checks': checks}, indent=2))


if __name__ == '__main__':
    main()
