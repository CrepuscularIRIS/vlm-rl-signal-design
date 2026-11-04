"""Over-magnification check: crop size in original pixels and the environment upscale it implies
(short side < 448 is upscaled, capped at 4x). Tiny = long side <= 128 original px (min crop is 56).
Reports share of crops / last crops that are tiny, and accuracy by last-crop size bucket, overall and by source.
python tiny_crops.py"""
import ast
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

R = Path(__file__).parents[2]
E1 = {"GRPO": R / "baselines_v52_20261024/full/traj_k8_grpo.jsonl", "DAPO": R / "baselines_v52_20261024/full/traj_k8_dapo.jsonl",
      "A+B": R / "sg_20261104/data/traj_sg_ab.jsonl", "A+B+C": R / "sg_20261104/data/traj_sg_abc.jsonl"}
BUCKETS = [(0, 128, "<=128px"), (128, 256, "129-256"), (256, 512, "257-512"), (512, 10 ** 9, ">512")]


def bx(r):
    b = r["boxes_abs"]
    b = ast.literal_eval(b) if isinstance(b, str) else (b or [])
    return [tuple(map(float, x)) for x in b[1:]]


def bucket(b):
    L = max(b[2] - b[0], b[3] - b[1])
    return next(n for lo, hi, n in BUCKETS if lo < L <= hi or (lo == 0 and L <= hi))


out = {}
for c, p in E1.items():
    rows = [json.loads(l) for l in open(p)]
    allc = [b for r in rows for b in bx(r)]
    last = defaultdict(list)
    last_src = defaultdict(lambda: defaultdict(list))
    for r in rows:
        b = bx(r)
        if b:
            k = bucket(b[-1])
            last[k].append(r["ok"])
            last_src[r["source"]][k].append(r["ok"])
    tiny_any = [r["ok"] for r in rows if any(max(x[2] - x[0], x[3] - x[1]) <= 128 for x in bx(r))]
    out[c] = {
        "crops": len(allc),
        "share_crops_tiny(<=128px)": round(sum(max(b[2] - b[0], b[3] - b[1]) <= 128 for b in allc) / len(allc), 3),
        "share_crops_at_min(<=64px)": round(sum(max(b[2] - b[0], b[3] - b[1]) <= 64 for b in allc) / len(allc), 3),
        "trajs_with_any_tiny": len(tiny_any), "acc_trajs_with_any_tiny": round(st.mean(tiny_any), 3) if tiny_any else None,
        "last_crop_bucket": {n: [len(last[n]), round(st.mean(last[n]), 3) if last[n] else None] for _, _, n in BUCKETS},
        "last_crop_tiny_by_source": {s: [len(v["<=128px"]), round(st.mean(v["<=128px"]), 3) if v["<=128px"] else None]
                                     for s, v in sorted(last_src.items())},
    }
print(json.dumps(out, indent=1))
json.dump(out, open(Path(__file__).parent / "tiny_crops.json", "w"), indent=1)
