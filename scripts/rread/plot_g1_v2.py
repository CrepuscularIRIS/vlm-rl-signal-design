# G1 (R-read) main figure — benchmark style (algorithm_comparison.png): all-English labels,
# constrained layout, dpi 180, one legend, per-panel metric captions, 2-line suptitle.
# Panels: reward / entropy / policy diff (KL) / reading consistency.
# python plot_g1_v2.py
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/lingxufeng/icml/obs/runs"
G1C, G0C = "#d62728", "#1f77b4"

g1 = json.load(open(f"{BASE}/rread_adm_20261103/data/curves/g1_curves.json"))["g1_rread"]
g0 = json.load(open(f"{BASE}/k8_20261028/curves_k8.json"))["k8_grpo"]
phi = {int(r["step"]): float(r["phi_mean_m1"]) for r in csv.DictReader(open(f"{BASE}/rread_adm_20261103/data/curves/phi_read_by_step.csv"))}

def series(d, key, sign=1):
    pts = sorted((int(s), sign * m[key]) for s, m in d.items() if key in m)
    return [p[0] for p in pts], [p[1] for p in pts]

fig, axs = plt.subplots(1, 4, figsize=(19, 4.6), layout="constrained")

# (a) Reward
axs[0].plot(*series(g1, "critic/rewards/mean"), color=G1C, lw=2.0, label="G1 (R-read): V*(Y+0.25*m*Phi)")
axs[0].plot(*series(g0, "critic/rewards/mean"), color=G0C, lw=2.0, label="G0 (GRPO): V*Y")
axs[0].set_title("Mean reward per rollout", fontsize=11)
axs[0].legend(fontsize=8.5, loc="lower right")

# (b) Entropy
axs[1].plot(*series(g1, "actor/entropy_loss", -1), color=G1C, lw=2.0)
axs[1].plot(*series(g0, "actor/entropy_loss", -1), color=G0C, lw=2.0)
axs[1].set_title("Policy entropy (-entropy_loss, logged actor metric)", fontsize=11)

# (c) Policy diff (KL)
axs[2].plot(*series(g1, "actor/ppo_kl"), color=G1C, lw=2.0)
axs[2].plot(*series(g0, "actor/ppo_kl"), color=G0C, lw=2.0)
axs[2].set_title("Policy diff vs reference (actor ppo_kl)", fontsize=11)

# (d) Reading consistency
steps = sorted(phi)
ys = [phi[s] for s in steps]
axs[3].scatter(steps, ys, s=14, color=G1C, alpha=0.45, zorder=3)
k = 5
sm = [sum(ys[max(0, i - k + 1):i + 1]) / len(ys[max(0, i - k + 1):i + 1]) for i in range(len(ys))]
axs[3].plot(steps, sm, color=G1C, lw=2.4, label="Phi_read per step (m=1) + 5-step avg")
axs[3].axhline(0.5518, color="#7f7f7f", ls="--", lw=1.4)
axs[3].axhline(0.6911, color="#2ca02c", ls=":", lw=1.6)
axs[3].text(31.6, 0.5518, "A-SFT .552", fontsize=8.5, ha="right", va="bottom", color="#7f7f7f")
axs[3].text(31.6, 0.6911, "G1-32 same-q probe .691", fontsize=8.5, ha="right", va="bottom", color="#2ca02c")
axs[3].set_ylim(0.44, 0.74)
axs[3].set_title("Reading consistency Phi_read (exact match vs L2 index)", fontsize=11)

for ax in axs:
    ax.grid(alpha=.2)
    ax.set_xlabel("RL step")
axs[0].set_ylabel("reward")
axs[1].set_ylabel("entropy (nats)")
axs[2].set_ylabel("KL")
axs[3].set_ylabel("Phi_read")

# endpoint annotations (sparse)
for ax, key, sign in [(axs[0], "critic/rewards/mean", 1), (axs[1], "actor/entropy_loss", -1), (axs[2], "actor/ppo_kl", 1)]:
    for d, c in [(g1, G1C), (g0, G0C)]:
        last = sorted((int(s), sign * m[key]) for s, m in d.items() if key in m)[-1]
        ax.annotate(f"{last[1]:.3f}", xy=last, xytext=(4, 0), textcoords="offset points",
                    fontsize=8.5, color=c, va="center")

fig.suptitle("G1 (R-read): reading reward is learned (Phi .55 -> .65, same-question probe .69) "
             "but end-task accuracy is flat (E1 .6395 = G0 .6390)",
             fontsize=13, fontweight="bold")
fig.text(0, -0.04, "Qwen3-VL-4B A-SFT start | no-filter GRPO | 4k pool x 8 rollouts | 32 steps | single variable vs G0: +0.25*m*Phi_read\n"
         "Probe = same 256 training-pool questions x 8, options included (de-optioned transfer test not run); E1 = 552 questions x 4",
         fontsize=9, color="#555")

out = f"{BASE}/rread_adm_20261103/data/curves/g1_main.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"))
print("saved", out)
