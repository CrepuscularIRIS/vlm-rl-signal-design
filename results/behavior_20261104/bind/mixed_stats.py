"""Phi_bind on real DAPO rollouts: within MIXED groups (what DAPO keeps), does Phi_bind differ among same-outcome
trajectories? Stratified by crop count and output length. Also writes per-trajectory scores for manual audit.
usage: python mixed_stats.py <bind_index.json> <out_scores.jsonl> <traj.jsonl> [...]"""
import ast
import json
import statistics as st
import sys
from collections import defaultdict

from bind_scorer import score

idx = json.load(open(sys.argv[1]))
rows = [json.loads(l) for p in sys.argv[3:] for l in open(p)]


def full(r):
    t = r["turns"]
    t = ast.literal_eval(t) if isinstance(t, str) else (t or [])
    return "".join(x.get("text") or "" for x in t) or (r.get("final_text") or "")


def crops(r):
    b = r["boxes_abs"]
    b = ast.literal_eval(b) if isinstance(b, str) else (b or [])
    return [tuple(map(float, x)) for x in b[1:]]


G = defaultdict(list)
with open(sys.argv[2], "w") as f:
    for r in rows:
        e = idx.get(r["uid"])
        if not e:
            continue
        phi, us = score(e, tuple(r["image_wh"]), crops(r), full(r))
        rec = {"uid": r["uid"], "s": r["sample_idx"], "ok": r["ok"], "phi": phi, "u": us, "n": len(crops(r)),
               "gen": r.get("gen_tokens") or 0}
        G[r["uid"]].append(rec)
        f.write(json.dumps(rec) + "\n")
T = [t for g in G.values() for t in g]
print(f"eligible questions {len(G)}, trajectories {len(T)}, slots {sum(len(idx[u]['slots']) for u in G)}")
print(f"Phi_bind mean: correct {st.mean(t['phi'] for t in T if t['ok']):.3f}  wrong {st.mean(t['phi'] for t in T if not t['ok']):.3f}"
      f"  | share >0: correct {st.mean(t['phi'] > 0 for t in T if t['ok']):.3f} wrong {st.mean(t['phi'] > 0 for t in T if not t['ok']):.3f}")
kinds = defaultdict(int)
mix_c = mix_w = 0
var_c = var_w = 0
for g in G.values():
    y = sum(t["ok"] for t in g)
    k = "all_correct" if y == len(g) else "all_wrong" if y == 0 else "mixed"
    kinds[k] += 1
    if k != "mixed":
        continue
    c = [t["phi"] for t in g if t["ok"]]
    w = [t["phi"] for t in g if not t["ok"]]
    if len(c) >= 2:
        mix_c += 1
        var_c += max(c) - min(c) > 1e-9
    if len(w) >= 2:
        mix_w += 1
        var_w += max(w) - min(w) > 1e-9
print("groups", dict(kinds))
print(f"MIXED groups: Phi_bind varies among the correct trajectories in {var_c}/{mix_c} groups (>=2 correct); "
      f"among the wrong trajectories in {var_w}/{mix_w} groups (>=2 wrong)")
for lab, key, cuts in (("crops", "n", [(0, 0), (1, 1), (2, 2), (3, 99)]), ("gen tokens", "gen", [(0, 150), (151, 300), (301, 10 ** 9)])):
    out = []
    for a, b in cuts:
        s = [t for t in T if a <= t[key] <= b]
        if s:
            out.append(f"{a}-{b}: n={len(s)} phi|correct {st.mean(t['phi'] for t in s if t['ok']) if any(t['ok'] for t in s) else float('nan'):.3f} "
                       f"phi|wrong {st.mean(t['phi'] for t in s if not t['ok']) if any(not t['ok'] for t in s) else float('nan'):.3f}")
    print(f"by {lab}: " + " | ".join(out))
