# SG two-arm training curves — benchmark style (rl-three-part-report): English labels, constrained layout,
# dpi 180, one legend, per-panel metric captions, 2-line suptitle.
# Inputs (this dir): sg_curves.json (parsed from sg_{ab,abc}.log), sg_{ab,abc}_rollouts.jsonl, k8_grpo_rollouts.jsonl,
#                    ../../k8_20261028/curves_k8.json.  Output: sg_curves.{png,pdf}, sg_curves_long.csv
# python plot_sg_curves.py
import csv
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = Path(__file__).parent
AB, ABC, K8 = "#d62728", "#9467bd", "#1f77b4"
logs = json.load(open(H / "sg_curves.json"))
k8log = json.load(open(H.parents[1] / "k8_20261028" / "curves_k8.json"))["k8_grpo"]


def ledger(path):
    R = [json.loads(l) for l in open(path)]
    out = {}
    for s in range(1, len(R) // 1024 + 1):
        B = R[(s - 1) * 1024:s * 1024]
        o = [r for r in B if not r.get("cont")]
        m = [r for r in o if r.get("m")]
        out[s] = {"acc": st.mean(r["correct"] for r in o), "crops": st.mean(r["crops"] for r in o),
                  "S": st.mean(r["s"] for r in m) if m else None,
                  "gate": st.mean(r.get("gate", 0) for r in m) if m else None}
    return out


L = {"sg_ab": ledger(H / "sg_ab_rollouts.jsonl"), "sg_abc": ledger(H / "sg_abc_rollouts.jsonl"),
     "k8_grpo": ledger(H / "k8_grpo_rollouts.jsonl")}
G = {"sg_ab": logs["sg_ab"], "sg_abc": logs["sg_abc"], "k8_grpo": k8log}
ARMS = [("sg_ab", AB, "A+B: SG reward + task-scaled adv"), ("sg_abc", ABC, "A+B+C: + evidence continuation"),
        ("k8_grpo", K8, "GRPO (k8, V*Y)")]

rows = []
for arm, _, _ in ARMS:
    for s in range(1, 33):
        g = G[arm].get(str(s), {})
        l = L[arm].get(s, {})
        rows.append({"arm": arm, "step": s, "reward": g.get("critic/rewards/mean"),
                     "entropy": g.get("actor/entropy_loss"),  # EasyR1 logs mean token entropy under this name
                     "ppo_kl": g.get("actor/ppo_kl"), "acc_orig": l.get("acc"), "crops": l.get("crops"),
                     "S_m1": l.get("S"), "gate_m1": l.get("gate")})
with open(H / "sg_curves_long.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)


def series(arm, key):
    pts = [(r["step"], r[key]) for r in rows if r["arm"] == arm and r[key] is not None]
    return [p[0] for p in pts], [p[1] for p in pts]


fig, axs = plt.subplots(2, 3, figsize=(18, 8.6), layout="constrained")
panels = [("reward", "Mean reward per rollout (logged; SG arms include 0.1*m*S)", "reward"),
          ("entropy", "Policy entropy (logged actor/entropy_loss = mean token entropy)", "entropy (nats)"),
          ("ppo_kl", "Policy diff vs rollout policy (actor ppo_kl)", "KL"),
          ("S_m1", "Scale-gated acquisition score S (eligible rows, m=1)", "S"),
          ("gate_m1", "Gate open: key evidence at usable scale (m=1)", "share of rollouts"),
          ("crops", "Crops per trajectory (original-question rows)", "crops / trajectory")]
for ax, (key, title, ylab) in zip(axs.flat, panels):
    ax.axvspan(30.5, 32.5, color=ABC, alpha=0.08)  # A+B+C steps 31-32: composition shift (see reward panel)
    for k, (arm, c, lab) in enumerate(ARMS):
        x, y = series(arm, key)
        if not x:
            continue
        lw = 2.4 if arm == "sg_ab" else 2.0
        ax.plot(x, y, color=c, lw=lw, marker="o", ms=2.5, label=lab)
        ax.annotate(f"{y[-1]:.3g}", (x[-1], y[-1]), xytext=(4, 9 * (1 - k)), textcoords="offset points",
                    fontsize=8, color=c, va="center")
    if key in ("S_m1", "gate_m1"):
        ax.text(0.02, 0.97, "GRPO: not computed", transform=ax.transAxes, fontsize=8, color=K8, va="top")
    ax.margins(y=0.08)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("RL step")
    ax.set_ylabel(ylab)
    ax.grid(alpha=.2)
    ax.set_xlim(0, 34.5)
axs[0, 0].text(30.3, 0.36, "A+B+C steps 31-32:\n49/128 and 109/128 ineligible Qs\n(GRPO acc on same Qs .566 / .318)",
               fontsize=7.5, ha="right", color=ABC)
axs[0, 0].legend(fontsize=8.5, loc="lower left")

ab_c, k8_c = L["sg_ab"], L["k8_grpo"]
late = range(25, 33)
fig.suptitle(
    f"SG arms raise S {L['sg_ab'][1]['S']:.2f}->{st.mean(L['sg_ab'][s]['S'] for s in late):.2f} (A+B, steps 25-32 mean) "
    f"and crops {st.mean(ab_c[s]['crops'] for s in late):.2f} vs GRPO {st.mean(k8_c[s]['crops'] for s in late):.2f} per trajectory\n"
    "Qwen3-VL-4B A-SFT, 4k pool x 8 rollouts x 32 steps, seed 42, same question order as GRPO; "
    "A+B vs GRPO: reward S term + advantage scaling; A+B+C vs A+B: 384 continuation rows (steps 1-16)",
    fontsize=11.5)
fig.savefig(H / "sg_curves.png", dpi=180)
fig.savefig(H / "sg_curves.pdf")
print("saved", H / "sg_curves.png")
