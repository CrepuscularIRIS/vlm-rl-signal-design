# R-read 信号准入统计（冻结协议 §信号准入）
# 输入: score_traj 输出的 scores.jsonl (uid, sample_idx, ok, K, phi, z)
# 输出: ν_allwrong / ν_allcorrect / ν_fixedview / 有效读取差异率 vs 10%/5% 准入线
"""用法: python admission_stats.py <scores.jsonl>"""
import json, sys
from collections import defaultdict


def main(path: str):
    rows = [json.loads(x) for x in open(path)]
    groups = defaultdict(list)
    for r in rows:
        groups[r["uid"]].append(r)
    n_groups = len(groups)
    n_rows = len(rows)

    allwrong = [g for g in groups.values() if all(r.get("vy", r["ok"]) == 0 for r in g)]
    allcorrect = [g for g in groups.values() if all(r.get("vy", r["ok"]) == 1 for r in g)]
    mixed = [g for g in groups.values() if len({r.get("vy", r["ok"]) for r in g}) > 1]

    def var(g):
        return len({round(r["phi"], 6) for r in g}) > 1

    nu_aw = sum(map(var, allwrong)) / max(1, len(allwrong))
    nu_ac = sum(map(var, allcorrect)) / max(1, len(allcorrect))
    no_task_signal = allwrong + allcorrect
    eff_rate = sum(map(var, no_task_signal)) / max(1, len(no_task_signal))

    # ν_fixedview: 字段级部分正确 (0 < 组内该字段 z=1 数 < 8)
    tot_fields = 0
    partial_fields = 0
    zero_fields = 0
    for g in groups.values():
        K = g[0]["K"]
        for j in range(K):
            c = sum(r["z"][j] for r in g)
            tot_fields += 1
            if 0 < c < len(g):
                partial_fields += 1
            elif c == 0:
                zero_fields += 1

    mean_phi = sum(r["phi"] for r in rows) / max(1, n_rows)
    stated = sum(1 for r in rows if sum(r["z"]) > 0) / max(1, n_rows)
    mean_K = sum(r["K"] for r in rows) / max(1, n_rows)

    print(f"groups={n_groups} rows={n_rows} mean_K={mean_K:.2f} mean_phi={mean_phi:.4f}")
    print(f"组构成: allwrong={len(allwrong)} allcorrect={len(allcorrect)} mixed={len(mixed)}")
    print(f"ν_allwrong  = {nu_aw:.3f}  ({sum(map(var, allwrong))}/{len(allwrong)})")
    print(f"ν_allcorrect= {nu_ac:.3f}  ({sum(map(var, allcorrect))}/{len(allcorrect)})")
    print(f"ν_fixedview = {partial_fields}/{tot_fields} = {partial_fields / max(1, tot_fields):.3f}"
          f"  (全零字段 {zero_fields}/{tot_fields})")
    print(f"有效读取差异率 = {eff_rate:.3f}  ({sum(map(var, no_task_signal))}/{len(no_task_signal)} 无任务信号组)")
    verdict = "≥10% → 完整训练" if eff_rate >= 0.10 else ("5–10% → 有限预算探索" if eff_rate >= 0.05 else "<5% → 暂停检查")
    print(f"准入判定: {verdict}")
    print(f"明确写出≥1个正确读数的轨迹比例 = {stated:.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
