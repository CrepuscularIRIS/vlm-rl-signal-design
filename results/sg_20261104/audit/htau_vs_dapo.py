"""DAPO vs DAPO+Htau.
(1) Sign flips of a lambda*Phi term in Y-mixed groups under standard GRPO normalisation, using the deployed Htau reward
    (reward_d2.py dc8fbbb4 + index.json 592c4184) on executed boxes from A+B training rollouts (same pool; Htau's own
    ledger has no boxes). Compared with A+B's task-scaled advantage. Steps 1-4 (policy ~ A-SFT) and all steps.
(2) E1 paired DAPO vs Htau: per source, crops on lost / gained keys, absent-target questions, V* evidence path.
python htau_vs_dapo.py"""
import ast
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
RUNS = HERE.parents[1]
OBS = RUNS.parent  # private layout; public copy: point OBS at the code/ + data locations
sys.path.insert(0, str(OBS / "src/grpo_sg/_remote"))
sys.path.insert(0, str(OBS / "notes/rereview_20261104"))
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("reward_d2", OBS / "src/grpo_sg/_remote/grpo_d2_reward_d2.py")
D2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D2)
D2._INDEX = json.loads((OBS / "src/grpo_d2/index.json").read_text())
from vstar_recheck import cover, find_item  # noqa: E402

out = {}

# ---- (1) flips ----
groups = defaultdict(list)
for l in open(RUNS / "sg_20261104/data/sg_ab_rollouts.jsonl"):
    r = json.loads(l)
    groups[r["uid"]].append(r)


def grpo_adv(R):
    m = st.mean(R)
    sd = st.stdev(R)
    return [(x - m) / (sd + 1e-6) for x in R]


def flips(steps, lam):
    c = Counter()
    for g in groups.values():
        if g[0]["step"] not in steps:
            continue
        V = [1.0 if r["terminal"] == "answer" else 0.0 for r in g]
        Y = [r["correct"] * v for r, v in zip(g, V)]
        if not (0 < sum(Y) < len(Y)):
            continue
        phi = [v * D2.process_score(r["qid"], tuple(r["wh"]), r["boxes"])[1] * D2.process_score(r["qid"], tuple(r["wh"]), r["boxes"])[0]
               for r, v in zip(g, V)]
        A = grpo_adv([y + lam * p for y, p in zip(Y, phi)])
        c["mixed_groups"] += 1
        c["correct"] += sum(Y)
        c["wrong"] += len(Y) - sum(Y)
        bad_c = sum(1 for y, a in zip(Y, A) if y == 1 and a <= 0)
        bad_w = sum(1 for y, a in zip(Y, A) if y == 0 and a >= 0)
        c["correct_with_nonpos_adv"] += bad_c
        c["wrong_with_nonneg_adv"] += bad_w
        c["groups_with_any_flip"] += (bad_c + bad_w) > 0
        c["phi_correct_sum"] += sum(p for y, p in zip(Y, phi) if y)
        c["phi_wrong_sum"] += sum(p for y, p in zip(Y, phi) if not y)
    c = dict(c)
    c["phi_mean_correct"] = round(c.pop("phi_correct_sum") / c["correct"], 3)
    c["phi_mean_wrong"] = round(c.pop("phi_wrong_sum") / c["wrong"], 3)
    return c


out["htau_flips_steps1-4_lam0.25"] = flips(range(1, 5), 0.25)
out["htau_flips_all_lam0.25"] = flips(range(1, 33), 0.25)
out["htau_flips_all_lam0.1"] = flips(range(1, 33), 0.1)

# ---- (2) E1 paired ----
F = RUNS / "baselines_v52_20261024/full"
T = {c: {(r["uid"], r["sample_idx"]): r for r in map(json.loads, open(p))}
     for c, p in (("DAPO", F / "traj_k8_dapo.jsonl"), ("Htau", F / "traj_ht532.jsonl"), ("GRPO", F / "traj_k8_grpo.jsonl"))}
keys = sorted(set(T["DAPO"]) & set(T["Htau"]))
assert len(keys) == 2208
src = defaultdict(lambda: [0, 0])
lost, gained = [], []
for k in keys:
    a, b = T["DAPO"][k], T["Htau"][k]
    if a["ok"] != b["ok"]:
        src[a["source"]][0 if a["ok"] else 1] += 1
        (lost if a["ok"] else gained).append(k)
out["paired_DAPO_vs_Htau"] = {s: {"only_DAPO": v[0], "only_Htau": v[1], "net_Htau": v[1] - v[0]} for s, v in sorted(src.items())}


def crops(c, ks):
    return round(st.mean(T[c][k]["n_crops"] or 0 for k in ks), 2)


out["crops_on_lost_keys(DAPO right, Htau wrong)"] = {"n": len(lost), "DAPO": crops("DAPO", lost), "Htau": crops("Htau", lost),
                                                     "Htau_more_crops": sum(T["Htau"][k]["n_crops"] > T["DAPO"][k]["n_crops"] for k in lost),
                                                     "Htau_fewer_crops": sum(T["Htau"][k]["n_crops"] < T["DAPO"][k]["n_crops"] for k in lost)}
out["crops_on_gained_keys(Htau right, DAPO wrong)"] = {"n": len(gained), "DAPO": crops("DAPO", gained), "Htau": crops("Htau", gained),
                                                       "Htau_more_crops": sum(T["Htau"][k]["n_crops"] > T["DAPO"][k]["n_crops"] for k in gained),
                                                       "Htau_fewer_crops": sum(T["Htau"][k]["n_crops"] < T["DAPO"][k]["n_crops"] for k in gained)}
out["crops_all"] = {c: crops(c, keys) for c in T}
out["acc_by_correctness_crops"] = {c: {"crops_when_right": round(st.mean(T[c][k]["n_crops"] or 0 for k in keys if T[c][k]["ok"]), 2),
                                       "crops_when_wrong": round(st.mean(T[c][k]["n_crops"] or 0 for k in keys if not T[c][k]["ok"]), 2)} for c in T}

# absent-target questions
N = OBS / "notes/rereview_20261104"
qd = pd.concat([pd.read_parquet(N / "e1_val.parquet"), pd.read_parquet(N / "vstar_holdout.parquet")])
q = {r["uid"]: str(r["env_head"]).split("\nOriginal image")[0] for _, r in qd.iterrows()}
NEG = re.compile(r"\(([A-J])\)\s*(This|The) image (doesn'?t|does not) feature", re.I)
ab = {}
for c in T:
    neg = [k for k in keys if (m := NEG.search(q.get(k[0], ""))) and T[c][k]["gt"] == m.group(1)]
    ab[c] = {"n": len(neg), "acc": round(st.mean(T[c][k]["ok"] for k in neg), 3)}
out["absent_target_questions"] = ab


# V* path
def bx(r):
    b = r["boxes_abs"]
    b = ast.literal_eval(b) if isinstance(b, str) else (b or [])
    return [tuple(map(float, x)) for x in b[1:]]


vs = {}
for c in T:
    after, stop, more = [], [], []
    for k in keys:
        r = T[c][k]
        if r["source"] != "vstar":
            continue
        it = find_item(r["uid"])
        g = [[x, y, x + w, y + h] for x, y, w, h in it["bbox"]]
        bs = bx(r)
        f = next((j for j, b in enumerate(bs) if all(cover(b, gg) >= .9 for gg in g)), None)
        if f is not None:
            n_after = len(bs) - f - 1
            after.append(n_after)
            (stop if n_after == 0 else more).append(r["ok"])
    vs[c] = {"crops_after_first_cover": round(st.mean(after), 2), "stop_at_cover": [len(stop), round(st.mean(stop), 3)],
             "more_after_cover": [len(more), round(st.mean(more), 3)]}
out["vstar_after_cover"] = vs
json.dump(out, open(HERE / "htau_vs_dapo.json", "w"), indent=1)
print(json.dumps(out, indent=1))
