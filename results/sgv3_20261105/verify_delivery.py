"""核对公开聚合、匿名逐题配对、bootstrap及发布文件SHA；不声称重放私有原始响应。"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parent
    result = json.loads((root / 'comparison.json').read_text())
    questions = json.loads((root / 'paired_questions.json').read_text())
    assert len(questions) == len({q['item_id'] for q in questions}) == 552
    assert result['acceptance']['identities_equal']
    names = list(result['summary'])
    ref = result['reference']
    draws = np.random.default_rng(result['bootstrap']['seed']).integers(0, 552, size=(10000, 552))
    for name in names:
        rows = [x for q in questions for x in q['correct'][name]]
        assert len(rows) == 2208 and set(rows) <= {0, 1}
        assert all(len(q['correct'][name]) == 4 for q in questions)
        summary = result['summary'][name]
        assert sum(rows) == summary['correct']
        assert round(sum(rows)/2208, 4) == summary['acc']
        for source, count in summary['by_source_counts'].items():
            values = [x for q in questions if q['source'] == source for x in q['correct'][name]]
            assert len(values) == count['n'] and sum(values) == count['correct']
        assert not result['header_checks'][name]['setting_differences']
        assert not result['header_checks'][name]['data_code_hash_differences']
        if name == ref:
            continue
        cells = {key: 0 for key in ('both_correct', 'new_only', 'control_only', 'both_wrong')}
        delta = []
        for q in questions:
            a, b = q['correct'][ref], q['correct'][name]
            delta.append((sum(a)-sum(b))/4)
            for x, y in zip(a, b):
                key = 'both_correct' if x and y else 'new_only' if x else 'control_only' if y else 'both_wrong'
                cells[key] += 1
        for key, value in cells.items():
            assert result['paired'][name][key] == value
        net = cells['new_only'] - cells['control_only']
        assert net == result['paired'][name]['net']
        assert abs(result['paired'][name]['delta_pp'] - net/2208*100) < 1e-12
        for source, expected in result['paired'][name]['by_source'].items():
            pairs = [(x, y) for q in questions if q['source'] == source
                     for x, y in zip(q['correct'][ref], q['correct'][name])]
            actual = {'both_correct': sum(x == y == 1 for x, y in pairs),
                      'new_only': sum(x == 1 and y == 0 for x, y in pairs),
                      'control_only': sum(x == 0 and y == 1 for x, y in pairs),
                      'both_wrong': sum(x == y == 0 for x, y in pairs),
                      'net': sum(x-y for x, y in pairs)}
            assert actual == expected
        ci = np.quantile(np.asarray(delta)[draws].mean(axis=1)*100, [.025, .975])
        assert np.allclose(ci, result['paired'][name]['question_bootstrap_95_pp'], atol=1e-12)
    with (root / 'family_long.csv').open() as stream:
        curves = list(csv.DictReader(stream))
    assert len(curves) == len({(r['run'], r['metric'], r['step']) for r in curves})
    assert all(math.isfinite(float(r['value'])) for r in curves)
    strat = [r for r in curves if r['run'] == 'strat']
    assert len(strat) == 256
    assert all(sorted(int(r['step']) for r in strat if r['metric'] == metric) == list(range(1, 33))
               for metric in {r['metric'] for r in strat})
    for row in json.loads((root / 'FILES.json').read_text()):
        assert hashlib.sha256((root/row['file']).read_bytes()).hexdigest() == row['sha256'], row['file']
    print(json.dumps({'status': 'pass', 'paired_conditions': len(names), 'trajectories_per_condition': 2208,
                      'questions': 552, 'strat_training_metric_rows': len(strat),
                      'scope': 'published aggregates and hashes; not raw-response or image replay'}, indent=2))


if __name__ == '__main__':
    main()
