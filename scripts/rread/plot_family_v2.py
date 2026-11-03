# Ours vs optimizer baselines — replaces README algorithm_comparison: PO/SP/CA/RLOO removed,
# our methods (G1 R-read / Htau / DTPO) highlighted. All-English, benchmark style, 2x3.
# python plot_family_v2.py
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/lingxufeng/icml/obs/runs"
REPO = "/home/lingxufeng/vlm-rl-signal-design"

# baselines from frozen curves.csv
brows = list(csv.DictReader(open(f"{REPO}/data/algorithm_curves/curves.csv")))
BL = {"grpo": "GRPO", "dapo": "DAPO", "gspo": "GSPO", "gigpo": "GiGPO"}
BLC = {"grpo": "#7fa8c9", "dapo": "#f0a878", "gspo": "#93c493", "gigpo": "#b39ddb"}
bl = {m: {int(r["step"]): {k: float(r[k]) for k in
         ("generated_accuracy", "entropy", "crops", "adv_nonzero_token_share", "cumulative_processed_tokens")}
         for r in brows if r["method"] == m} for m in BL}

# ours
g1c = json.load(open(f"{BASE}/rread_adm_20261103/data/curves/g1_curves.json"))["g1_rread"]
g1l = json.load(open(f"{BASE}/rread_adm_20261103/data/curves/g1_ledger.json"))
g1_acc = {int(s): m["reward/accuracy"] for s, m in g1c.items() if "reward/accuracy" in m}
g1_ent = {int(s): -m["actor/entropy_loss"] for s, m in g1c.items() if "actor/entropy_loss" in m}
dtpo = {int(r["step"]): float(r["accuracy"]) for r in csv.DictReader(open(f"{REPO}/results/dtpo_e1_20261103/training_points.csv")) if r["method"] == "DTPO" and r["series"] == "train"}
ht5 = {int(r["sampling_step"]): float(r["y"]) for r in csv.DictReader(open(f"{BASE}/d2_ht5_20261102/curves/ht5_signals.csv"))}
ht5_crops = {int(r["sampling_step"]): float(r["crops"]) for r in csv.DictReader(open(f"{BASE}/d2_ht5_20261102/curves/ht5_signals.csv"))}

OURS = {"G1 R-read": "#d62728", "Htau": "#e377c2", "DTPO": "#8c564b"}

fig, axs = plt.subplots(2, 3, figsize=(16, 8.4), layout="constrained")

def plot_kv(ax, data, color, lw, label=None, ms=0):
    xs = sorted(data)
    ax.plot(xs, [data[x] for x in xs], color=color, lw=lw, label=label,
            marker="o" if ms else None, ms=ms)

# (a) accuracy
for m, name in BL.items():
    plot_kv(axs[0][0], {s: v["generated_accuracy"] for s, v in bl[m].items()}, BLC[m], 1.3, name)
plot_kv(axs[0][0], g1_acc, OURS["G1 R-read"], 2.6, "G1 R-read (ours)")
plot_kv(axs[0][0], ht5, OURS["Htau"], 1.8, "Htau (ours)", ms=3)
plot_kv(axs[0][0], dtpo, OURS["DTPO"], 1.8, "DTPO (ours)")
axs[0][0].set_title("Accuracy of ALL generated training rollouts"); axs[0][0].legend(fontsize=8, ncol=2)

# (b) entropy
for m, name in BL.items():
    plot_kv(axs[0][1], {s: v["entropy"] for s, v in bl[m].items()}, BLC[m], 1.3)
plot_kv(axs[0][1], g1_ent, OURS["G1 R-read"], 2.6)
axs[0][1].set_title("Logged actor entropy (mask / reduction differ)\nHtau / DTPO not logged -> missing")

# (c) crops
for m, name in BL.items():
    plot_kv(axs[0][2], {s: v["crops"] for s, v in bl[m].items()}, BLC[m], 1.3)
plot_kv(axs[0][2], {int(k): v for k, v in g1l["crops"].items()}, OURS["G1 R-read"], 2.6)
plot_kv(axs[0][2], ht5_crops, OURS["Htau"], 1.8, ms=3)
axs[0][2].set_title("Executed crops / generated trajectory")

# (d) adv nonzero share
for m, name in BL.items():
    plot_kv(axs[1][0], {s: v["adv_nonzero_token_share"] for s, v in bl[m].items()}, BLC[m], 1.3)
plot_kv(axs[1][0], {int(k): v for k, v in g1l["adv_nonzero"].items()}, OURS["G1 R-read"], 2.6)
axs[1][0].set_title("Advantage-nonzero token share in update\nG1: reading reward keeps all-correct/all-wrong groups learnable")

# (e) cumulative tokens
for m, name in BL.items():
    plot_kv(axs[1][1], {s: v["cumulative_processed_tokens"] / 1e6 for s, v in bl[m].items()}, BLC[m], 1.3)
plot_kv(axs[1][1], {int(k): v for k, v in g1l["cum_tokens_m"].items()}, OURS["G1 R-read"], 2.6)
axs[1][1].set_title("Cumulative processed rollout tokens (M)")

# (f) E1 bars
E1 = [("A-SFT", 0.5670, "#aaaaaa"), ("GiGPO", 0.6209, BLC["gigpo"]), ("GRPO (G0)", 0.6390, BLC["grpo"]),
      ("GSPO", 0.6395, BLC["gspo"]), ("G1 R-read", 0.6395, OURS["G1 R-read"]),
      ("Htau", 0.6508, OURS["Htau"]), ("DTPO", 0.6553, OURS["DTPO"]), ("DAPO", 0.6635, BLC["dapo"])]
ax = axs[1][2]; ax.grid(axis="x", alpha=.2)
ax.barh(range(len(E1)), [e[1] for e in E1], color=[e[2] for e in E1], height=0.62)
ax.set_yticks(range(len(E1))); ax.set_yticklabels([e[0] for e in E1], fontsize=10)
for i, e in enumerate(E1):
    ax.text(e[1] + 0.002, i, f"{e[1]:.4f}", va="center", fontsize=8.5)
ax.set_xlim(0.54, 0.685)
ax.set_title("E1 accuracy, 552 questions x 4\nPO/SP/CA and RLOO family removed per direction")
ax.get_yticklabels()[4].set_fontweight("bold")

for ax in axs.flat:
    ax.grid(alpha=.2)
    ax.set_xlabel("RL step")
axs[0][0].set_ylabel("accuracy"); axs[1][0].set_ylabel("share")

fig.suptitle("Our reward-signal methods (G1 R-read / Htau / DTPO) vs optimizer baselines — 32 steps, 4k pool x 8, single seed\n"
             "Same start (A-SFT) for G1/Htau/G0; DTPO/Htau published earlier; G1 single variable vs G0: +0.25*m*Phi_read",
             fontsize=12)

out = f"{BASE}/rread_adm_20261103/data/curves/ours_vs_optimizers.png"
fig.savefig(out, dpi=180); fig.savefig(out.replace(".png", ".pdf"))
print("saved", out)
