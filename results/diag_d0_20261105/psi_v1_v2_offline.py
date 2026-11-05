"""Offline v1 vs v2 S on the GRPO+A training ledger (same executed boxes, same qids). CPU, facts only.
v1 = ledger field s (obs/src/grpo_sg/reward_sg.py); v2 = obs/src/grpo_sgv2/reward_sg.py recomputed from boxes.
python psi_v1_v2_offline.py -> psi_v1_v2.json"""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

H = Path(__file__).parent
sys.path.insert(0, str(H.parents[1] / "src/grpo_sgv2"))
import reward_sg as V2  # noqa: E402

R = [json.loads(l) for l in open(H.parent / "sg_20261104/data/sg_a_rollouts.jsonl")]
R = [r for r in R if r.get("m") == 1 and not r.get("cont") and r["terminal"] == "answer"]
for r in R:
    _, r["s2"], d = V2.process_score(r["qid"], tuple(r["wh"]), r["boxes"])
    r.update({"prox": d["prox"], "gate2": d["gate"], "explore2": d["explore"]})
    W, Hh = r["wh"]
    r["fullish"] = int(any((b[2] - b[0]) * (b[3] - b[1]) >= 0.9 * W * Hh for b in r["boxes"]))


def auc(groups, key):
    num = den = 0.0
    for g in groups:
        c = [r[key] for r in g if r["correct"] >= 1]
        w = [r[key] for r in g if r["correct"] < 1]
        for a in c:
            for b in w:
                num += 1 if a > b else .5 if a == b else 0
                den += 1
    return round(num / den, 4) if den else None


out = {}
for ph, steps in (("step1-2", range(1, 3)), ("step25-32", range(25, 33))):
    X = [r for r in R if r["step"] in steps]
    gby = defaultdict(list)
    for r in X:
        gby[(r["step"], r["uid"])].append(r)
    mixed = [g for g in gby.values() if 0 < sum(r["correct"] >= 1 for r in g) < len(g)]
    bycrops = lambda key: {k: round(st.mean(r[key] for r in X if min(r["crops"], 6) == k), 3)  # noqa: E731
                           for k in range(7) if any(min(r["crops"], 6) == k for r in X)}
    s2tot = sum(r["s2"] for r in X)
    out[ph] = {"n": len(X), "mean_S_v1": round(st.mean(r["s"] for r in X), 4), "mean_S_v2": round(st.mean(r["s2"] for r in X), 4),
               "share_S_v2_mass_gate_closed": round(sum(r["s2"] for r in X if not r["gate2"]) / s2tot, 4) if s2tot else None,
               "mean_explore_v2": round(st.mean(r["explore2"] for r in X), 4),
               "fullish_crop_rate": round(st.mean(r["fullish"] for r in X), 4),
               "mean_S_by_crops_v1": bycrops("s"), "mean_S_by_crops_v2": bycrops("s2"),
               "within_group_auc_vs_Y": {"S_v1": auc(mixed, "s"), "S_v2": auc(mixed, "s2"), "prox": auc(mixed, "prox")},
               "mean_S_v2_correct_vs_wrong": [round(st.mean(r["s2"] for r in X if r["correct"] >= 1), 4),
                                              round(st.mean(r["s2"] for r in X if r["correct"] < 1), 4)],
               "gate0_share": round(st.mean(V2.process_score(r["qid"], tuple(r["wh"]), [])[2]["gate0"] for r in X), 4)}
json.dump(out, open(H / "psi_v1_v2.json", "w"), indent=1)
print(json.dumps(out, indent=1))
