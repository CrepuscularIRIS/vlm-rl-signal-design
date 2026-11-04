"""E1 comparison GRPO vs A+B vs A+B+C: paired 4-cells (uid+sample_idx), per-source, cost, V* object scale.
usage: python e1_compare.py   (run from this dir; writes e1_compare.json)"""
import ast
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

H = Path(__file__).parent
sys.path.insert(0, str(H.parents[1] / ".." / "notes" / "rereview_20261104"))
FULL = H.parents[1] / "baselines_v52_20261024" / "full"
CONDS = {"GRPO": FULL / "traj_k8_grpo.jsonl", "A+B": H / "traj_sg_ab.jsonl", "A+B+C": H / "traj_sg_abc.jsonl"}


def load(p):
    return {(r["uid"], r["sample_idx"]): r for r in map(json.loads, open(p))}


T = {k: load(v) for k, v in CONDS.items()}
keys = sorted(set.intersection(*(set(t) for t in T.values())))
assert len(keys) == 2208, len(keys)
out = {"n": len(keys), "acc": {}, "by_source": {}, "cost": {}, "paired": {}}
for c, t in T.items():
    out["acc"][c] = round(st.mean(t[k]["ok"] for k in keys), 4)
    src = defaultdict(list)
    for k in keys:
        src[t[k]["source"]].append(t[k]["ok"])
    out["by_source"][c] = {s: [sum(v), len(v), round(sum(v) / len(v), 4)] for s, v in sorted(src.items())}
    out["cost"][c] = {"crops": round(st.mean(t[k]["n_crops"] or 0 for k in keys), 3),
                      "zero_crop": round(st.mean((t[k]["n_crops"] or 0) == 0 for k in keys), 4),
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
    cells["by_source_only_a_only_b"] = dict(by_src)
    return cells


for a, b in (("GRPO", "A+B"), ("GRPO", "A+B+C"), ("A+B", "A+B+C")):
    out["paired"][f"{a} vs {b}"] = paired(a, b)

# V* object scale (rereview_20261104 metric): GT long side / smallest covering crop long side
from vstar_recheck import cover, find_item  # noqa: E402
vs = {}
for c, t in T.items():
    res = {"right": [], "wrong": [], "wrong_n": 0}
    for k in keys:
        r = t[k]
        if r["source"] != "vstar":
            continue
        it = find_item(r["uid"])
        b = r["boxes_abs"]
        crops = (ast.literal_eval(b) if isinstance(b, str) else (b or []))[1:]
        gts = [[x, y, x + w, y + h] for x, y, w, h in it["bbox"]]
        res["wrong_n"] += not r["ok"]
        fr = []
        for g in gts:
            cs = [cc for cc in crops if cover(cc, g) >= 0.9]
            if not cs:
                fr = None
                break
            cc = min(cs, key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))
            fr.append(max(g[2] - g[0], g[3] - g[1]) / max(cc[2] - cc[0], cc[3] - cc[1]))
        if fr:
            res["right" if r["ok"] else "wrong"].append(min(fr))
    vs[c] = {"wrong_n": res["wrong_n"], "wrong_hit": len(res["wrong"]),
             "wrong_hit_median_scale": round(st.median(res["wrong"]), 3) if res["wrong"] else None,
             "wrong_hit_lt5pct": sum(x < .05 for x in res["wrong"]),
             "right_hit": len(res["right"]), "right_hit_median_scale": round(st.median(res["right"]), 3)}
out["vstar_scale"] = vs
json.dump(out, open(H / "e1_compare.json", "w"), indent=1)
print(json.dumps(out, indent=1))
