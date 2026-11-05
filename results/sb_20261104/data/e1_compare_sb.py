"""E1 comparison for DAPO-SB: paired 4-cells (uid+sample_idx) vs DAPO (direct control) and the other 4k arms,
per-source counts, cost. Also summarises the training-side probe log (sb_log.jsonl).
usage: python e1_compare_sb.py   (run from anywhere; writes e1_compare_sb.json next to this file)"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

H = Path(__file__).parent
FULL = H.parents[1] / "baselines_v52_20261024" / "full"
SG = H.parents[1] / "sg_20261104" / "data"
CONDS = {"DAPO-SB": H / "traj_sb_dapo_sb.jsonl", "DAPO": FULL / "traj_k8_dapo.jsonl",
         "GRPO": FULL / "traj_k8_grpo.jsonl", "GRPO+A": SG / "traj_sg_a.jsonl", "GRPO+A+B": SG / "traj_sg_ab.jsonl"}


def load(p):
    return {(r["uid"], r["sample_idx"]): r for r in map(json.loads, open(p))}


T = {k: load(v) for k, v in CONDS.items()}
keys = sorted(set.intersection(*(set(t) for t in T.values())))
assert len(keys) == 2208 and all(len(t) == 2208 for t in T.values()), len(keys)
crops = lambda r: r["n_crops"] or 0  # noqa: E731
out = {"n": len(keys), "acc": {}, "by_source": {}, "cost": {}, "paired": {}}
for c, t in T.items():
    out["acc"][c] = round(st.mean(t[k]["ok"] for k in keys), 4)
    src = defaultdict(list)
    for k in keys:
        src[t[k]["source"]].append(t[k]["ok"])
    out["by_source"][c] = {s: [sum(v), len(v), round(sum(v) / len(v), 4)] for s, v in sorted(src.items())}
    out["cost"][c] = {"crops": round(st.mean(crops(t[k]) for k in keys), 3),
                      "zero_crop": round(st.mean(crops(t[k]) == 0 for k in keys), 4),
                      "gen_tokens": round(st.mean(t[k]["gen_tokens"] or 0 for k in keys), 1),
                      "proc_tokens": round(st.mean(t[k]["prompt_tokens"] or 0 for k in keys), 1),
                      "no_answer": round(st.mean(t[k]["pred"] is None for k in keys), 4)}


def paired(a, b):
    cells = {"both": 0, "only_" + a: 0, "only_" + b: 0, "neither": 0}
    by_src = defaultdict(lambda: [0, 0])
    for k in keys:
        x, y = T[a][k]["ok"], T[b][k]["ok"]
        if x and y:
            cells["both"] += 1
        elif x:
            cells["only_" + a] += 1
            by_src[T[a][k]["source"]][0] += 1
        elif y:
            cells["only_" + b] += 1
            by_src[T[a][k]["source"]][1] += 1
        else:
            cells["neither"] += 1
    cells["net_" + b] = cells["only_" + b] - cells["only_" + a]
    cells["by_source_only_a_only_b"] = dict(sorted(by_src.items()))
    return cells


for b in ("DAPO", "GRPO", "GRPO+A", "GRPO+A+B"):
    out["paired"][f"DAPO-SB vs {b}"] = paired("DAPO-SB", b)

# zero-crop vs cropped accuracy (DAPO-SB vs DAPO), same keys
zc = {}
for c in ("DAPO-SB", "DAPO"):
    t = T[c]
    z = [t[k]["ok"] for k in keys if crops(t[k]) == 0]
    nz = [t[k]["ok"] for k in keys if crops(t[k]) > 0]
    zc[c] = {"zero_crop_n": len(z), "zero_crop_acc": round(st.mean(z), 4) if z else None,
             "cropped_n": len(nz), "cropped_acc": round(st.mean(nz), 4) if nz else None}
out["zero_crop_split"] = zc

# identical-count check flagged during scoring (mmmu_pro 49/156 in several arms)
mp = [k for k in keys if T["DAPO-SB"][k]["source"] == "mmmu_pro"]
out["mmmu_pro_same_pred_vs_GRPO"] = sum(T["DAPO-SB"][k]["pred"] == T["GRPO"][k]["pred"] for k in mp)

# training-side probe log
L = [json.loads(x) for x in open(H / "sb_log.jsonl")]
pick = lambda f: [round(r[f], 4) for r in L]  # noqa: E731
out["train_probe"] = {f: pick(f) for f in ("mean_b0", "mean_acc", "mean_groupacc_minus_b0", "mean_letter_mass",
                                            "share_raw_adv_pos", "mean_raw_adv", "turns_per_traj",
                                            "adv_abs_mean_sb", "adv_abs_mean_dapo")}
json.dump(out, open(H / "e1_compare_sb.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "train_probe"}, indent=1))
print({f: (v[0], v[-1], round(st.mean(v), 4)) for f, v in out["train_probe"].items()})
