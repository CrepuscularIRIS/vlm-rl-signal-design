"""DAPO-SB training-log figure (three-part report, part 3) + long CSV.
Inputs (this dir): curves_sb.json (parse_log of sb_dapo_sb.log), sb_log.jsonl; ../../k8_20261028/curves_k8.json.
python plot_sb_curves.py  -> sb_curves.png/.pdf, sb_curves_long.csv"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

H = Path(__file__).parent
sb = json.load(open(H / "curves_sb.json"))["sb_dapo_sb"]
k8 = json.load(open(H.parents[1] / "k8_20261028" / "curves_k8.json"))
probe = [json.loads(x) for x in open(H / "sb_log.jsonl")]
STEPS = list(range(1, 33))
RUNS = {"DAPO-SB": sb, "DAPO": k8["k8_dapo"], "GRPO": k8["k8_grpo"]}
STYLE = {"DAPO-SB": dict(color="tab:red", lw=2.6, zorder=3), "DAPO": dict(color="tab:blue", lw=1.6),
         "GRPO": dict(color="tab:gray", lw=1.0, alpha=.45)}
METRICS = {"score": "critic/score/mean", "entropy": "actor/entropy_loss", "ppo_kl": "actor/ppo_kl",
           "grad_norm": "actor/grad_norm"}

rows = []
for run, g in RUNS.items():
    for s in STEPS:
        for m, key in METRICS.items():
            v = g.get(str(s), {}).get(key)
            if v is not None:
                rows.append({"run": run, "step": s, "metric": m, "value": v, "source": "train log"})
for s, r in zip(STEPS, probe):
    assert r["call"] == s
    for m in ("mean_b0", "mean_acc", "mean_groupacc_minus_b0", "mean_letter_mass", "turns_per_traj",
              "share_raw_adv_pos", "adv_abs_mean_sb", "adv_abs_mean_dapo"):
        rows.append({"run": "DAPO-SB", "step": s, "metric": m, "value": r[m], "source": "sb_log.jsonl"})
with open(H / "sb_curves_long.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["run", "step", "metric", "value", "source"])
    w.writeheader()
    w.writerows(rows)


def series(run, m):
    d = {r["step"]: r["value"] for r in rows if r["run"] == run and r["metric"] == m}
    return [s for s in STEPS if s in d], [d[s] for s in STEPS if s in d]


fig, ax = plt.subplots(2, 3, figsize=(17, 9), layout="constrained", dpi=180)
panels = [("score", "Rollout accuracy on retained mixed groups (logged score)", "accuracy", ("DAPO-SB", "DAPO")),
          ("entropy", "Policy entropy (logged actor/entropy_loss = mean token entropy)", "entropy (nats)",
           ("DAPO-SB", "DAPO", "GRPO")),
          ("ppo_kl", "Policy diff vs rollout policy (actor ppo_kl, log printed at 3 d.p.)", "KL",
           ("DAPO-SB", "DAPO", "GRPO")),
          ("grad_norm", "Pre-clip grad norm (max_grad_norm = 1.0)", "L2 norm", ("DAPO-SB", "DAPO", "GRPO"))]
for a, (m, title, ylab, runs) in zip(ax.flat, panels):
    for run in runs:
        x, y = series(run, m)
        a.plot(x, y, label=run, **STYLE[run])
        a.annotate(f"{y[-1]:.3g}", (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8,
                   color=STYLE[run]["color"])
    a.set_title(title, fontsize=10)
    a.set_ylabel(ylab)
a = ax.flat[4]
for m, lab, c in (("mean_acc", "retained-batch accuracy Y", "tab:red"), ("mean_b0", "probe b0 = P(gt) before turn 0",
                                                                          "tab:orange"),
                  ("mean_groupacc_minus_b0", "group acc - b0 (mean over groups)", "tab:brown")):
    x, y = series("DAPO-SB", m)
    a.plot(x, y, label=lab, color=c, lw=2)
    a.annotate(f"{y[-1]:.3f}", (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=c)
a.set_title("DAPO-SB stop probe on retained batch (DAPO has no probe)", fontsize=10)
a.set_ylabel("probability / accuracy")
a.legend(fontsize=8, loc="center right")
a = ax.flat[5]
x, y = series("DAPO-SB", "turns_per_traj")
a.plot(x, y, color="tab:red", lw=2, label="assistant turns / trajectory")
a.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, color="tab:red")
a.set_ylabel("turns / trajectory")
b = a.twinx()
x, y = series("DAPO-SB", "mean_letter_mass")
b.plot(x, y, color="tab:purple", lw=1.4, ls="--", label="probe option-letter mass")
b.set_ylabel("letter mass")
a.set_title("DAPO-SB retained batch: turns and probe letter mass", fontsize=10)
a.legend(handles=a.get_lines() + b.get_lines(), fontsize=8, loc="lower right")
for a in ax.flat:
    a.set_xlabel("update step")
    a.grid(alpha=.2)
ax.flat[0].legend(fontsize=9)
ax.flat[1].legend(fontsize=9)
fig.suptitle("DAPO-SB E1 .6413 vs DAPO .6635 (paired net -49 / 2208); entropy .448 -> .341 (DAPO .446 at step 32)\n"
             "Only variable vs DAPO: advantage (Y - b_k)/sigma_q with pre-turn stop probe b_k, lambda = 1. "
             "A-SFT start, 4k pool, seed 42, n = 8, 32 steps, one seed. GRPO (grey) has no filtering.", fontsize=11)
fig.savefig(H / "sb_curves.png")
fig.savefig(H / "sb_curves.pdf")
print("rows", len(rows))
