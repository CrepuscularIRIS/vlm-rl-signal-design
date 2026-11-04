"""Real-trajectory checks for the evidence-fact scorer.
usage: python gate_real.py <evfact_index.json> <traj.jsonl> [label]
Reports: eligible coverage, E by answer correctness within equal crop counts, crop-count dependence of E,
all-wrong groups with >=1 slot split (some trajectories u=1, others u=0) = teachable groups."""
import ast
import json
import statistics as st
import sys
from collections import defaultdict

from evfact import score

idx = json.load(open(sys.argv[1]))
rows = [json.loads(l) for l in open(sys.argv[2])]
label = sys.argv[3] if len(sys.argv) > 3 else sys.argv[2]


def full_text(r):
    t = r["turns"]
    t = ast.literal_eval(t) if isinstance(t, str) else (t or [])
    return "".join(x.get("text") or "" for x in t) or (r.get("final_text") or "")


def crops(r):
    b = r["boxes_abs"]
    b = ast.literal_eval(b) if isinstance(b, str) else (b or [])
    return [tuple(map(float, x)) for x in b[1:]]


G = defaultdict(list)
for r in rows:
    e = idx.get(r["uid"])
    if not e:
        continue
    E, u, _ = score(e, tuple(r["image_wh"]), crops(r), full_text(r))
    G[r["uid"]].append({"ok": r["ok"], "E": E, "u": u, "n": len(crops(r))})
traj = [t for g in G.values() for t in g]
print(f"[{label}] eligible questions {len(G)} / {len({r['uid'] for r in rows})}; trajectories {len(traj)}")
print(f"  E mean: correct {st.mean(t['E'] for t in traj if t['ok']):.3f}  wrong {st.mean(t['E'] for t in traj if not t['ok']):.3f}")
for n in range(0, 4):
    a = [t["E"] for t in traj if t["n"] == n and t["ok"]]
    b = [t["E"] for t in traj if t["n"] == n and not t["ok"]]
    if a and b:
        print(f"  crops={n}: E correct {st.mean(a):.3f} (n={len(a)})  wrong {st.mean(b):.3f} (n={len(b)})")
print("  E by crop count:", {n: round(st.mean(t["E"] for t in traj if t["n"] == n), 3)
                             for n in sorted({t["n"] for t in traj}) if sum(t["n"] == n for t in traj) >= 20})
kinds = defaultdict(int)
teach = splits = slots = 0
for g in G.values():
    y = sum(t["ok"] for t in g)
    k = "all_wrong" if y == 0 else "all_correct" if y == len(g) else "mixed"
    kinds[k] += 1
    if k == "all_wrong":
        K = len(g[0]["u"])
        s = sum(1 for j in range(K) if 0 < sum(t["u"][j] for t in g) < len(g))
        splits += s
        slots += K
        teach += s > 0
print(f"  groups {dict(kinds)}; all-wrong groups with >=1 slot split: {teach}/{kinds['all_wrong']} (slot splits {splits}/{slots})")
