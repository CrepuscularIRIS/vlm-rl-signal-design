"""Offline admission for reward_sg on real A-SFT trajectories (rl4k pool, 256 q x 8, temp 0.7).

usage: python sg_admission.py <traj_rread_adm.jsonl>
Reports: eligibility, S distribution, within-group S variance in all-wrong / all-correct groups (no-task-signal
groups: the only place S adds information under GRPO), mixed-group discrimination P(S_correct > S_wrong),
attack deltas (append full-image crop; 8-crop tiling), and the scale proxy p at the best covering view.
"""
import itertools
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "grpo_sg"))
import reward_sg as R  # noqa: E402

rows = [json.loads(l) for l in open(sys.argv[1])]
groups = defaultdict(list)
for r in rows:
    W, H = r["image_wh"]
    crops = (r.get("boxes_abs") or [])[1:]           # [0] is the initial full view
    valid = r.get("pred") is not None
    m, s, d = R.process_score(r["uid"], (W, H), crops) if valid else (0, 0.0, {})
    groups[r["uid"]].append({"y": r["ok"], "m": m, "s": s, "d": d, "crops": crops, "wh": (W, H)})

elig = {u: g for u, g in groups.items() if any(t["m"] for t in g)}
traj = [t for g in elig.values() for t in g]
print(f"questions {len(groups)} eligible {len(elig)} ({len(elig)/len(groups):.1%}); eligible traj {len(traj)}")
print(f"S mean {st.mean(t['s'] for t in traj):.3f}  S>0 {sum(t['s']>0 for t in traj)/len(traj):.1%}  "
      f"gate {sum(t['d'].get('gate',0) for t in traj)/len(traj):.1%}  gate0 {sum(t['d'].get('gate0',0) for t in traj)/len(traj):.1%}")
for y in (0, 1):
    ss = [t["s"] for t in traj if t["y"] == y]
    print(f"  Y={y}: n {len(ss)} S mean {st.mean(ss):.3f}")

allw = [g for g in elig.values() if not any(t["y"] for t in g)]
allc = [g for g in elig.values() if all(t["y"] for t in g)]
mixed = [g for g in elig.values() if 0 < sum(t["y"] for t in g) < len(g)]


def var_share(gs):
    return sum(1 for g in gs if max(t["s"] for t in g) - min(t["s"] for t in g) > 1e-6), len(gs)


a, n = var_share(allw); b, k = var_share(allc)
print(f"no-task-signal groups with S variance: all-wrong {a}/{n}, all-correct {b}/{k}; "
      f"combined {(a+b)}/{(n+k)} = {(a+b)/max(1,n+k):.1%} (admission line 10%)")
win = tie = tot = 0
for g in mixed:
    for p, q in itertools.product([t for t in g if t["y"]], [t for t in g if not t["y"]]):
        tot += 1
        win += p["s"] > q["s"]
        tie += p["s"] == q["s"]
print(f"mixed groups {len(mixed)}: P(S_correct > S_wrong) {win/tot:.3f}, ties {tie/tot:.3f}, "
      f"P(>) + P(=)/2 = {(win + tie/2)/tot:.3f}")

# attacks
full_gain, tile_gain = [], []
for g in elig.values():
    for t in g:
        W, H = t["wh"]
        base = R.process_score_cached = None
        s_full = R.process_score(t_uid := None or next(u for u, gg in elig.items() if gg is g), (W, H),
                                 t["crops"] + [(0, 0, W, H)])[1]
        full_gain.append(s_full - t["s"])
        tiles = [(i * W / 4, j * H / 2, (i + 1) * W / 4, (j + 1) * H / 2) for i in range(4) for j in range(2)]
        tile_gain.append(R.process_score(t_uid, (W, H), tiles)[1] - t["s"])
print(f"attack append-full-image: mean dS {st.mean(full_gain):+.4f}, max {max(full_gain):+.4f}")
print(f"attack 8-tile grid (replaces trajectory): mean dS {st.mean(tile_gain):+.3f}, "
      f"share dS>0 {sum(x>0 for x in tile_gain)/len(tile_gain):.1%}")
print(f"crops: Y=1 {st.mean(len(t['crops']) for t in traj if t['y']):.2f}  Y=0 {st.mean(len(t['crops']) for t in traj if not t['y']):.2f}")
