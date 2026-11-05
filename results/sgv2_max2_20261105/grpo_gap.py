"""同键（uid + sample_idx）拆分两个条件的 E1 差距：按裁图数、来源和缺答案计数，只输出聚合数。

用法：
  python grpo_gap.py --ref max2 REF_TRAJ.jsonl --ctrl GRPO CTRL_TRAJ.jsonl \
      --comparison comparison.json --out grpo_gap.json

正确判定沿用评测记录的 pred 字段：pred 非空、等于 gt、无 error、未截断；
并断言两边正确数与 comparison.json（统一评分器）一致。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

BUCKETS = ('0', '1', '2', '3-4', '5+')
SOURCES = ('mme_realworld', 'vstar', 'worldbench', 'mmmu_pro')
EXTRA_CROP_BINS = ('<=0', '(0,1]', '(1,3]', '>3')


def bucket(n_crops: int) -> str:
    """裁图次数分档。"""
    if n_crops <= 2:
        return str(n_crops)
    return '3-4' if n_crops <= 4 else '5+'


def extra_bin(diff: float) -> str:
    """每题平均多裁图数（参考 − 对照）分档。"""
    if diff <= 0:
        return '<=0'
    return '(0,1]' if diff <= 1 else '(1,3]' if diff <= 3 else '>3'


def load(path: Path) -> dict[tuple[str, int], dict]:
    """读取轨迹，键为 (uid, sample_idx)。"""
    rows = {}
    for line in path.open():
        r = json.loads(line)
        key = (r['uid'], r['sample_idx'])
        assert key not in rows, key
        truncated = bool(r.get('overlong')) or (bool(r.get('turns')) and r['turns'][-1].get('finish') == 'length')
        ok = r['pred'] is not None and r['pred'] == r['gt'] and not r.get('error') and not truncated
        rows[key] = {'ok': int(ok), 'crops': int(r['n_crops']), 'source': r['source'],
                     'missing': r['pred'] is None}
    return rows


def crop_stats(rows: dict) -> tuple[dict, dict]:
    """裁图数分布，以及按自身裁图数分档的正确数。"""
    hist, acc = Counter(), defaultdict(lambda: {'n': 0, 'correct': 0})
    for r in rows.values():
        b = bucket(r['crops'])
        hist[b] += 1
        acc[b]['n'] += 1
        acc[b]['correct'] += r['ok']
    return ({b: hist[b] for b in BUCKETS}, {b: acc[b] for b in BUCKETS if acc[b]['n']})


def net_by_ctrl_crops(ref: dict, ctrl: dict) -> dict:
    """同键上按对照条件的裁图数分档：条数、两边正确数、分源净差。"""
    out = {b: {'n': 0, 'ctrl_correct': 0, 'ref_correct': 0, 'net_by_source': dict.fromkeys(SOURCES, 0)}
           for b in BUCKETS}
    for key, c in ctrl.items():
        cell = out[bucket(c['crops'])]
        cell['n'] += 1
        cell['ctrl_correct'] += c['ok']
        cell['ref_correct'] += ref[key]['ok']
        cell['net_by_source'][c['source']] += ref[key]['ok'] - c['ok']
    for cell in out.values():
        cell['net'] = cell['ref_correct'] - cell['ctrl_correct']
    return out


def question_level(ref: dict, ctrl: dict) -> dict:
    """按题汇总 4 次采样：平均多裁图数分档下的题数与净差。"""
    per_q = defaultdict(lambda: [0, 0, 0])  # [ref_ok − ctrl_ok, ref_crops − ctrl_crops, samples]
    for (uid, _), c in ctrl.items():
        r = ref[(uid, _)]
        q = per_q[uid]
        q[0] += r['ok'] - c['ok']
        q[1] += r['crops'] - c['crops']
        q[2] += 1
    out = {b: {'questions': 0, 'net': 0} for b in EXTRA_CROP_BINS}
    for net, extra, samples in per_q.values():
        cell = out[extra_bin(extra / samples)]
        cell['questions'] += 1
        cell['net'] += net
    return out


def analyse(ref_name: str, ref: dict, ctrl_name: str, ctrl: dict) -> dict:
    """组装全部聚合结果。"""
    assert ref.keys() == ctrl.keys(), 'paired keys differ'
    hist_ref, acc_ref = crop_stats(ref)
    hist_ctrl, acc_ctrl = crop_stats(ctrl)
    missing_ref = [k for k, r in ref.items() if r['missing']]
    return {
        'ref': ref_name, 'ctrl': ctrl_name, 'n': len(ref),
        'correct': {ref_name: sum(r['ok'] for r in ref.values()), ctrl_name: sum(c['ok'] for c in ctrl.values())},
        'crop_hist': {ref_name: hist_ref, ctrl_name: hist_ctrl},
        'acc_by_own_crops': {ref_name: acc_ref, ctrl_name: acc_ctrl},
        'net_by_ctrl_crops': net_by_ctrl_crops(ref, ctrl),
        'question_level_by_extra_crops': question_level(ref, ctrl),
        'missing_answer': {ref_name: len(missing_ref), ctrl_name: sum(c['missing'] for c in ctrl.values())},
        'ref_missing_where_ctrl_correct': sum(ctrl[k]['ok'] for k in missing_ref),
        'bucket_note': 'n_crops counts executed crops; the original image is not a crop',
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ref', nargs=2, metavar=('NAME', 'TRAJ'), required=True)
    ap.add_argument('--ctrl', nargs=2, metavar=('NAME', 'TRAJ'), required=True)
    ap.add_argument('--comparison', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    result = analyse(args.ref[0], load(Path(args.ref[1])), args.ctrl[0], load(Path(args.ctrl[1])))
    summary = json.loads(args.comparison.read_text())['summary']
    for name, correct in result['correct'].items():
        assert correct == summary[name]['correct'], (name, correct, summary[name]['correct'])
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({'out': str(args.out.name), **result['correct'],
                      'net': result['correct'][args.ref[0]] - result['correct'][args.ctrl[0]]}))


if __name__ == '__main__':
    main()
