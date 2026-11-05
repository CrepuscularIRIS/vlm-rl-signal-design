"""How DAPO-SB's advantage was distributed in training, recomputed offline from the rollout ledger (b per turn) and
the update ledger (qids kept per update). Same formula as stop_relative_advantage.build_advantages with lambda=1:
A_{i,k} = (Y_i - b_{i,k}) / (sigma_q + 1e-6), sigma_q = sample std of Y in the group (ddof=1). DAPO on the same
groups: A_i = (Y_i - mean_q Y) / (sigma_q + 1e-6) (EasyR1 GRPO-style z-score uses the same ddof=1 std).
Approximate token weight per turn = gen_tokens / n_turns (per-turn token counts are not in the ledger).
python credit_facts.py -> credit_facts.json"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

H = Path(__file__).parent
EPS = 1e-6
upd = [set(json.loads(l)["qids"]) for l in open(H / "updates.jsonl")]
calls = defaultdict(list)
for l in open(H / "rollouts.jsonl"):
    r = json.loads(l)
    calls[r["step"]].append(r)

# map each reward call to the update that kept its groups (same rule as stage_report build_family.map_calls)
u, call_upd, done = 0, {}, defaultdict(set)
for c in sorted(calls):
    q = {r["qid"] for r in calls[c]}
    while u < len(upd) - 1 and not (q & upd[u]) and any(q & x for x in upd[u + 1:]):
        u += 1
    call_upd[c] = u
    done[u] |= q
    if q & upd[u] and not (upd[u] - done[u]):
        u += 1

groups = []  # (update_step, [rows]) for kept mixed groups
for c, rows in calls.items():
    s = call_upd[c]
    byq = defaultdict(list)
    for r in rows:
        byq[r["qid"]].append(r)
    for qid, g in byq.items():
        ys = [int(r["correct"] >= 1) for r in g]
        if qid in upd[s] and len(g) == 8 and 0 < sum(ys) < 8:
            groups.append((s + 1, g))
seen, kept = set(), []
for s, g in sorted(groups, key=lambda x: x[0]):
    if (s, g[0]["qid"]) not in seen:
        seen.add((s, g[0]["qid"]))
        kept.append((s, g))

turns = []  # one record per (trajectory, turn)
traj = []
for s, g in kept:
    ys = [int(r["correct"] >= 1) for r in g]
    ybar = sum(ys) / 8
    sig = st.stdev(ys)
    for r, y in zip(g, ys):
        b = r["b"]
        n = len(b)
        w = r["gen_tokens"] / n
        a_dapo = (y - ybar) / (sig + EPS)
        a_sb = [(y - bk) / (sig + EPS) for bk in b]
        traj.append({"s": s, "y": y, "crops": r["crops"], "a_dapo": a_dapo, "a_sb_mean": st.mean(a_sb), "b0": b[0],
                     "bK": b[-1], "n": n})
        for k, a in enumerate(a_sb):
            turns.append({"s": s, "y": y, "k": k, "last": k == n - 1, "a_sb": a, "a_dapo": a_dapo, "w": w,
                          "delta": (b[k + 1] - b[k]) if k < n - 1 else None, "crops": r["crops"]})


def mass(rows, key, weighted):
    pos = sum(r[key] * (r["w"] if weighted else 1) for r in rows if r[key] > 0)
    neg = -sum(r[key] * (r["w"] if weighted else 1) for r in rows if r[key] < 0)
    return round(pos, 1), round(neg, 1), round(pos / neg, 3) if neg else None


out = {"kept_groups": len(kept), "trajectories": len(traj), "turns": len(turns),
       "kept_groups_per_update_min_max": [min(sum(1 for s, _ in kept if s == t) for t in range(1, 33)),
                                          max(sum(1 for s, _ in kept if s == t) for t in range(1, 33))]}
out["mass_pos_neg_ratio"] = {
    "turns_unweighted": {"SB": mass(turns, "a_sb", False), "DAPO": mass(turns, "a_dapo", False)},
    "turns_token_weighted_approx": {"SB": mass(turns, "a_sb", True), "DAPO": mass(turns, "a_dapo", True)}}
out["mean_A_per_turn"] = {"SB": round(st.mean(t["a_sb"] for t in turns), 4),
                          "DAPO": round(st.mean(t["a_dapo"] for t in turns), 4)}
cell = {}
for y in (1, 0):
    for zc in (True, False):
        R = [t for t in traj if t["y"] == y and (t["crops"] == 0) == zc]
        cell[f"{'correct' if y else 'wrong'}_{'zero_crop' if zc else 'cropped'}"] = {
            "n": len(R), "A_SB_mean_over_turns": round(st.mean(t["a_sb_mean"] for t in R), 3) if R else None,
            "A_DAPO": round(st.mean(t["a_dapo"] for t in R), 3) if R else None,
            "b0": round(st.mean(t["b0"] for t in R), 3) if R else None}
out["by_correctness_and_crop"] = cell
early = [t for t in traj if t["s"] <= 4]
late = [t for t in traj if t["s"] >= 29]
out["zero_crop_share_kept"] = {"steps_1_4": round(st.mean(t["crops"] == 0 for t in early), 3),
                               "steps_29_32": round(st.mean(t["crops"] == 0 for t in late), 3)}
ct = [t for t in turns if t["delta"] is not None]  # turns followed by another turn = crop turns
out["crop_turns"] = {"n": len(ct), "share_b_rises_after": round(st.mean(t["delta"] > 0 for t in ct), 3),
                     "share_b_falls_after": round(st.mean(t["delta"] < 0 for t in ct), 3),
                     "mean_delta": round(st.mean(t["delta"] for t in ct), 4),
                     "mean_delta_in_correct": round(st.mean(t["delta"] for t in ct if t["y"]), 4),
                     "mean_delta_in_wrong": round(st.mean(t["delta"] for t in ct if not t["y"]), 4)}
cc = [t for t in traj if t["y"] == 1 and t["crops"] > 0]
out["correct_cropped_first_vs_last_turn"] = {"n": len(cc), "A_turn0": round(st.mean((1 - t["b0"]) for t in cc), 3),
                                             "A_last_turn_raw": round(st.mean((1 - t["bK"]) for t in cc), 3),
                                             "note": "raw (before / sigma)"}
gm = defaultdict(list)
for t in traj:
    gm[t["s"]].append(t)
out["per_update_mean_A_SB"] = [round(st.mean(t["a_sb_mean"] for t in gm[s]), 3) for s in range(1, 33)]
json.dump(out, open(H / "credit_facts.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "per_update_mean_A_SB"}, indent=1))
print("per-update mean A_SB:", out["per_update_mean_A_SB"])
