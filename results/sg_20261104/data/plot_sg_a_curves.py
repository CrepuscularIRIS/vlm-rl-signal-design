"""GRPO+A (sg_a) training-log figure: sg_a vs GRPO (direct control) vs A+B (reference).
Inputs (this dir): curves_sg_a.json (parse_log of sg_a.log), sg_a_ledger_steps.json (per reward call from
sg_a/rollouts.jsonl; GRPO has no resampling so call = update step), sg_curves_long.csv (GRPO, A+B).
python plot_sg_a_curves.py -> sg_a_curves.png/.pdf, sg_a_curves_long.csv"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

H = Path(__file__).parent
STEPS = list(range(1, 33))
log = json.load(open(H / "curves_sg_a.json"))["sg_a"]
led = json.load(open(H / "sg_a_ledger_steps.json"))
M = ["reward", "entropy", "ppo_kl", "acc_orig", "crops", "S_m1"]
data = {"sg_a": {s: {"reward": log[str(s)]["critic/score/mean"], "entropy": log[str(s)]["actor/entropy_loss"],
                     "ppo_kl": log[str(s)]["actor/ppo_kl"], "acc_orig": led[str(s)]["acc"],
                     "crops": led[str(s)]["crops"], "S_m1": led[str(s)]["S_m1"]} for s in STEPS}}
for r in csv.DictReader(open(H / "sg_curves_long.csv")):
    if r["arm"] in ("k8_grpo", "sg_ab"):
        data.setdefault(r["arm"], {})[int(r["step"])] = {m: (float(r[m]) if r.get(m) not in (None, "") else None)
                                                          for m in M}
with open(H / "sg_a_curves_long.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["arm", "step"] + M)
    for arm, d in data.items():
        for s in STEPS:
            w.writerow([arm, s] + [d.get(s, {}).get(m) for m in M])
NAME = {"sg_a": "GRPO+A (S reward)", "k8_grpo": "GRPO", "sg_ab": "A+B"}
STYLE = {"sg_a": dict(color="tab:red", lw=2.6, zorder=3), "k8_grpo": dict(color="tab:blue", lw=1.6),
         "sg_ab": dict(color="tab:gray", lw=1.1, alpha=.55)}
panels = [("reward", "Mean reward per rollout (logged; S arms include 0.1*m*S)", "reward"),
          ("entropy", "Policy entropy (logged actor/entropy_loss)", "entropy (nats)"),
          ("ppo_kl", "Policy diff vs rollout policy (actor ppo_kl)", "KL"),
          ("acc_orig", "Rollout task accuracy Y (ledger)", "accuracy"),
          ("crops", "Crops per trajectory (ledger)", "crops / trajectory"),
          ("S_m1", "Scale-gated acquisition S on eligible questions (m=1)", "S")]
fig, ax = plt.subplots(2, 3, figsize=(17, 9), layout="constrained", dpi=180)
for a, (m, title, ylab) in zip(ax.flat, panels):
    for arm in ("sg_ab", "k8_grpo", "sg_a"):
        xs = [s for s in STEPS if data[arm].get(s, {}).get(m) is not None]
        if not xs:
            continue
        ys = [data[arm][s][m] for s in xs]
        a.plot(xs, ys, label=NAME[arm], **STYLE[arm])
        a.annotate(f"{ys[-1]:.3g}", (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8,
                   color=STYLE[arm]["color"])
    a.set_title(title, fontsize=10)
    a.set_ylabel(ylab)
    a.set_xlabel("update step")
    a.grid(alpha=.2)
ax.flat[0].legend(fontsize=9)
fig.suptitle("GRPO+A E1 .6436 vs GRPO .6390 (paired net +10 / 2208); crops/trajectory on E1 4.02 vs 1.91, "
             "tokens 18,034 vs 8,125\nOnly variable vs GRPO: reward V*(Y + 0.1*m*S), standard GRPO normalisation. "
             "A-SFT start, 4k pool, seed 42, n = 8, 32 steps, one seed. S not computed for GRPO (missing).",
             fontsize=11)
fig.savefig(H / "sg_a_curves.png")
fig.savefig(H / "sg_a_curves.pdf")
