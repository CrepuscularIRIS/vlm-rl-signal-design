"""What do successive crops do? For each crop k>=2: relation to the previous crop (zoom_in: >=90% inside previous and
area <= 0.5x; widen: contains previous; overlap; move: IoU<0.05 and not containing). Accuracy by first-transition type.
Also checks the 1,024 single-crop coincidence (GRPO vs DAPO keys).
python crop_relations.py"""
import ast
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

R = Path(__file__).parents[2]
E1 = {"GRPO": R / "baselines_v52_20261024/full/traj_k8_grpo.jsonl", "DAPO": R / "baselines_v52_20261024/full/traj_k8_dapo.jsonl",
      "A+B": R / "sg_20261104/data/traj_sg_ab.jsonl", "A+B+C": R / "sg_20261104/data/traj_sg_abc.jsonl"}


def bx(r):
    b = r["boxes_abs"]
    b = ast.literal_eval(b) if isinstance(b, str) else (b or [])
    return [tuple(map(float, x)) for x in b[1:]]


def inter(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return iw * ih


def area(a):
    return (a[2] - a[0]) * (a[3] - a[1])


def rel(prev, cur):
    i = inter(prev, cur)
    if i >= 0.9 * area(cur) and area(cur) <= 0.5 * area(prev):
        return "zoom_in"
    if i >= 0.9 * area(prev):
        return "widen_or_same"
    u = area(prev) + area(cur) - i
    return "move" if i / u < 0.05 else "shift_overlap"


T = {c: {(r["uid"], r["sample_idx"]): r for r in map(json.loads, open(p))} for c, p in E1.items()}
one = {c: {k for k, r in t.items() if (r["n_crops"] or 0) == 1} for c, t in T.items()}
print("single-crop keys: GRPO", len(one["GRPO"]), "DAPO", len(one["DAPO"]), "overlap", len(one["GRPO"] & one["DAPO"]))
out = {}
for c, t in T.items():
    trans = Counter()
    by_src = defaultdict(Counter)
    acc2 = defaultdict(list)
    for k, r in t.items():
        b = bx(r)
        for j in range(1, len(b)):
            x = rel(b[j - 1], b[j])
            trans[x] += 1
            by_src[r["source"]][x] += 1
        if len(b) >= 2:
            acc2[rel(b[0], b[1])].append(r["ok"])
    tot = sum(trans.values())
    out[c] = {"later_crops": tot, "share": {k: round(v / tot, 3) for k, v in trans.most_common()},
              "acc_by_crop1->crop2": {k: [round(st.mean(v), 3), len(v)] for k, v in sorted(acc2.items())},
              "mme_share": {k: round(v / sum(by_src["mme_realworld"].values()), 3) for k, v in by_src["mme_realworld"].most_common()}}
print(json.dumps(out, indent=1))
json.dump(out, open(Path(__file__).parent / "crop_relations.json", "w"), indent=1)
