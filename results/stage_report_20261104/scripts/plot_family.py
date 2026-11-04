# Stage-report figures, one metric per figure (large). Benchmark style: English text, constrained layout, dpi 180,
# legend in one place, per-figure metric caption, 2-line suptitle. Emphasis: A+B, A+B+C (bold), GRPO / DAPO (medium);
# all other same-protocol runs faint. Lines = 3-step centered moving average; dots = raw per-step values (emphasised runs).
# Inputs: family_long.csv (build_family.py), ../data/sg_curves_long.csv. Output: figs/*.png|pdf
# python plot_family.py
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = Path(__file__).parent
OUT = H / "figs"
OUT.mkdir(exist_ok=True)
D = defaultdict(dict)
for r in csv.DictReader(open(H / "family_long.csv")):
    D[(r["run"], r["metric"])][int(r["step"])] = float(r["value"])
for r in csv.DictReader(open(H.parent / "data" / "sg_curves_long.csv")):
    run = {"sg_ab": "A+B", "sg_abc": "A+B+C"}.get(r["arm"])
    if run:
        for m in ("S_m1", "gate_m1"):
            if r[m]:
                D[(run, m)][int(r["step"])] = float(r[m])

STYLE = {  # run: (label, color, lw, alpha, ls, z)
    "A+B": ("GRPO + scale-gated evidence reward + task-scaled advantage (A+B)", "#d62728", 3.0, 1.0, "-", 10),
    "A+B+C": ("A+B + evidence-continuation rows, steps 1-16 (A+B+C)", "#9467bd", 3.0, 1.0, "-", 9),
    "GRPO": ("GRPO (outcome reward V*Y, no filtering)", "#1f77b4", 2.3, 0.95, "-", 8),
    "DAPO": ("DAPO (dynamic-sampling filter, clip-higher)", "#ff7f0e", 2.3, 0.95, "-", 7),
    "GSPO": ("GSPO", "#2ca02c", 1.3, 0.35, "-", 2),
    "GiGPO": ("GiGPO", "#8c564b", 1.3, 0.35, "-", 2),
    "RLOO": ("RLOO", "#7f7f7f", 1.3, 0.35, "-", 2),
    "RLOO-decision": ("RLOO + first-action fork", "#bcbd22", 1.3, 0.35, "-", 2),
    "DTPO": ("DTPO", "#e377c2", 1.3, 0.35, "-", 2),
    "SAPO": ("SAPO (1 rollout / question)", "#17becf", 1.3, 0.35, "-", 2),
    "G1 R-read": ("GRPO + reading-consistency reward (G1)", "#000000", 1.4, 0.40, "--", 3),
    "Htau": ("DAPO + tau-gated box-coverage reward (Htau)", "#c51b7d", 1.4, 0.40, ":", 3),
}
EMPH = ("A+B", "A+B+C", "GRPO", "DAPO")
SUB = ("Qwen3-VL-4B from A-SFT | 4k pool (4,096 questions, one pass) x 8 rollouts x 32 steps | seed 42, lr 1e-6, "
       "clip .2/.3, no KL | caption v5.1 crop protocol")


def ma(xs, ys, k=3):
    out = []
    for i in range(len(ys)):
        w = ys[max(0, i - k // 2):i + k // 2 + 1]
        out.append(sum(w) / len(w))
    return out


def figure(metric, title, ylabel, headline, fname, runs=None, raw=False, shade=True, scale=1.0, extra=None):
    fig, ax = plt.subplots(figsize=(13, 7), layout="constrained")
    missing, ends = [], []
    for run, (lab, c, lw, a, ls, z) in STYLE.items():
        if runs and run not in runs:
            continue
        d = D.get((run, metric))
        if not d:
            missing.append(lab.split(" (")[0] if run not in ("A+B", "A+B+C") else run)
            continue
        xs = sorted(d)
        ys = [d[x] * scale for x in xs]
        line = ys if raw else ma(xs, ys)
        ax.plot(xs, line, color=c, lw=lw, alpha=a, ls=ls, label=lab, zorder=z)
        if run in EMPH:
            if not raw:
                ax.scatter(xs, ys, color=c, s=12, alpha=0.30, zorder=z)
            ends.append([line[-1], f"{ys[-1]:.3g}" + ("" if raw else " (last raw)" if abs(ys[-1] - line[-1]) > 0.02 * (max(ys) - min(ys) + 1e-9) * 5 else ""), c, run])
    if shade and (runs is None or "A+B+C" in runs):
        ax.axvspan(30.5, 32.5, color="#9467bd", alpha=0.07)
        ax.text(32.4, 0.02, "A+B+C steps 31-32:\ndifferent question mix", transform=ax.get_xaxis_transform(),
                fontsize=8.5, color="#9467bd", ha="right")
    if extra:
        extra(ax)
    ax.set_xlim(0.5, 36.5)
    lo, hi = ax.get_ylim()
    gap = 0.045 * (hi - lo)
    ends.sort(key=lambda e: e[0])
    pos = []
    for e in ends:  # dodge end labels vertically
        y = e[0] if not pos or e[0] - pos[-1] >= gap else pos[-1] + gap
        pos.append(y)
        ax.annotate(e[1], (32, e[0]), xytext=(32.6, y), textcoords="data", fontsize=10, color=e[2], va="center",
                    fontweight="bold" if e[3] in ("A+B", "A+B+C") else None,
                    arrowprops=dict(arrowstyle="-", color=e[2], lw=0.6, alpha=0.6))
    ax.set_xlabel("RL update step", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12.5)
    ax.grid(alpha=.2)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=9.5, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    if missing:
        ax.text(1.01, 0.02, "not logged: " + ", ".join(missing), transform=ax.transAxes, fontsize=8.5, color="#7f7f7f")
    fig.suptitle(headline + "\n" + SUB, fontsize=12)
    fig.savefig(OUT / f"{fname}.png", dpi=180)
    fig.savefig(OUT / f"{fname}.pdf")
    plt.close(fig)


def late(run, metric, a=25, b=32):
    d = D[(run, metric)]
    return st.mean(d[s] for s in range(a, b + 1) if s in d)


figure("acc", "Rollout task accuracy Y on training questions (continuation rows excluded)",
       "accuracy (fraction of rollouts)",
       f"Training accuracy, steps 25-32 mean: A+B {late('A+B','acc'):.3f}, A+B+C {late('A+B+C','acc',25,30):.3f} "
       f"(steps 25-30), GRPO {late('GRPO','acc'):.3f}, DAPO {late('DAPO','acc'):.3f}", "fig01_train_accuracy")
figure("reward", "Mean reward per rollout (logged; A+B arms include 0.1*m*S, G1 0.25*m*Phi)",
       "reward", f"Logged reward follows accuracy; A+B ends {D[('A+B','reward')][32]:.3f} vs GRPO "
       f"{D[('GRPO','reward')][32]:.3f}; A+B+C step 32 = {D[('A+B+C','reward')][32]:.3f} (question mix, see shading)",
       "fig02_reward")
figure("entropy", "Policy entropy (logged actor/entropy_loss = mean token entropy)",
       "entropy (nats / token)",
       f"Entropy, steps 25-32 mean: A+B {late('A+B','entropy'):.3f} > GRPO {late('GRPO','entropy'):.3f} > "
       f"A+B+C {late('A+B+C','entropy'):.3f}; DAPO {late('DAPO','entropy'):.3f}", "fig03_entropy")
figure("ppo_kl", "Policy diff: actor/ppo_kl, update vs rollout policy (k1 estimate, raw per step)",
       "KL x 1e-4", "Policy diff stays within +-1e-4 for all logged runs except one A+B step (1.0e-3 at step 5)",
       "fig04_policy_diff", raw=True, scale=1e4, extra=lambda ax: (ax.set_ylim(-1.2, 3.0), ax.annotate("A+B step 5: 10.0 (clipped)", (5, 3.0), xytext=(7, 2.6), fontsize=10, color="#d62728", arrowprops=dict(arrowstyle="->", color="#d62728"))))
figure("grad_norm", "Actor grad norm before clipping (DTPO / SAPO: mean over updates)",
       "grad norm", f"Gradient norm, steps 25-32 mean: A+B {late('A+B','grad_norm'):.2f}, A+B+C "
       f"{late('A+B+C','grad_norm'):.2f}, GRPO {late('GRPO','grad_norm'):.2f}, DAPO {late('DAPO','grad_norm'):.2f}",
       "fig05_grad_norm")
figure("crops", "Executed crops per trajectory during training (original-question rows)", "crops / trajectory",
       f"Crops per trajectory, steps 25-32 mean: A+B {late('A+B','crops'):.2f}, A+B+C {late('A+B+C','crops'):.2f}, "
       f"GRPO {late('GRPO','crops'):.2f}, DAPO {late('DAPO','crops'):.2f}", "fig06_crops")
figure("gen_tok", "Generated (policy) tokens per trajectory during training", "tokens / trajectory",
       f"Generated tokens, steps 25-32 mean: A+B {late('A+B','gen_tok'):.0f}, A+B+C {late('A+B+C','gen_tok'):.0f}, "
       f"GRPO {late('GRPO','gen_tok'):.0f}, DAPO {late('DAPO','gen_tok'):.0f}", "fig07_gen_tokens")
figure("adv_nz", "Share of response tokens with nonzero advantage in the update batch",
       "share of tokens", f"Nonzero-advantage share, steps 1-32 mean: A+B {late('A+B','adv_nz',1,32):.2f}, A+B+C "
       f"{late('A+B+C','adv_nz',1,32):.2f}, GRPO {late('GRPO','adv_nz',1,32):.2f}, DAPO "
       f"{late('DAPO','adv_nz',1,32):.2f} (DAPO drops all-same groups before the update)", "fig08_adv_nonzero")
figure("S_m1", "Scale-gated evidence acquisition score S on eligible questions (m=1)", "S (0-1)",
       f"S rises from {D[('A+B','S_m1')][1]:.2f} to {late('A+B','S_m1'):.2f} (A+B, steps 25-32 mean); A+B+C "
       f"{late('A+B+C','S_m1',25,30):.2f} (steps 25-30)", "fig09_S", runs=("A+B", "A+B+C"))
figure("gate_m1", "Gate open: some key evidence box reached usable input scale (m=1)", "share of rollouts",
       f"Gate-open share rises from {D[('A+B','gate_m1')][1]:.2f} to {late('A+B','gate_m1'):.2f} (A+B, steps 25-32 mean)",
       "fig10_gate", runs=("A+B", "A+B+C"))

# E1 bar + cost scatter
E1 = [  # condition, acc, crops/traj, processed tokens/traj, generated tokens/traj
    ("A-SFT (start)", .5670, 0.994, 5015, None), ("GRPO", .6390, 1.905, 8125, 261), ("DAPO", .6635, 2.092, 8930, 287),
    ("GSPO", .6395, 2.263, 9756, 278), ("GiGPO", .6209, 1.998, 8506, 275), ("RLOO", .6318, 1.992, 8535, 278),
    ("RLOO-decision", .5919, 1.372, 6266, 203), ("DTPO", .6553, 3.357, 15579, 395), ("SAPO", .5770, 1.063, 5292, 171),
    ("G1 R-read", .6395, 2.308, 10063, 374), ("Htau", .6508, 2.546, 10872, 305),
    ("A+B", .6472, 2.404, 10047, 342), ("A+B+C", .6395, 2.337, 10181, 326)]
COL = {k: v[1] for k, v in STYLE.items()}
COL["A-SFT (start)"] = "#bbbbbb"
fig, ax = plt.subplots(figsize=(13, 6.5), layout="constrained")
order = sorted(E1, key=lambda r: r[1])
for i, (c, a, *_rest) in enumerate(order):
    em = c in EMPH
    ax.barh(i, a * 100, color=COL[c], alpha=1.0 if em else 0.35, edgecolor="black" if c in ("A+B", "A+B+C") else None,
            lw=1.5)
    ax.text(a * 100 + 0.15, i, f"{a*100:.2f}", va="center", fontsize=10.5, fontweight="bold" if em else None)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([r[0] for r in order], fontsize=11)
ax.set_xlim(55, 68)
ax.axvline(63.90, color="#1f77b4", ls="--", lw=1)
ax.set_xlabel("E1 accuracy (%), 552 questions x 4 samples = 2,208; failures kept in denominator", fontsize=11.5)
ax.set_title("Fixed-endpoint E1 (step 32 checkpoints; same scorer); dashed line = GRPO", fontsize=12.5)
ax.grid(alpha=.2, axis="x")
fig.suptitle("E1: DAPO 66.35 > DTPO 65.53 > Htau 65.08 > A+B 64.72 > GRPO 63.90 = A+B+C 63.95; "
             "A+B vs GRPO paired net +18 / 2,208\n" + SUB, fontsize=12)
fig.savefig(OUT / "fig11_e1_scores.png", dpi=180)
fig.savefig(OUT / "fig11_e1_scores.pdf")
plt.close(fig)

OFF = {"GSPO": (-60, -20), "G1 R-read": (8, 14), "A+B+C": (10, -16), "DTPO": (-170, 8)}
fig, ax = plt.subplots(figsize=(13.5, 7), layout="constrained")
for c, a, cr, tok, g in E1:
    em = c in EMPH
    ax.scatter(tok, a * 100, s=220 if em else 110, color=COL[c], alpha=1 if em else 0.45,
               edgecolor="black" if c in ("A+B", "A+B+C") else "none", lw=1.5, zorder=3)
    ax.annotate(f"{c} ({cr:.2f} crops)", (tok, a * 100), xytext=OFF.get(c, (8, 4)), textcoords="offset points", fontsize=10,
                fontweight="bold" if em else None, color="black" if em else "#555555")
ax.set_xlabel("processed tokens per trajectory on E1 (sum over turns of input + output)", fontsize=11.5)
ax.set_ylabel("E1 accuracy (%)", fontsize=11.5)
ax.set_title("E1 accuracy vs inference cost (labels: mean crops per trajectory)", fontsize=12.5)
ax.grid(alpha=.2)
fig.suptitle("Cost: A+B uses 10.0k tokens and 2.40 crops per trajectory vs GRPO 8.1k / 1.91 and DAPO 8.9k / 2.09\n"
             + SUB, fontsize=12)
fig.savefig(OUT / "fig12_e1_cost.png", dpi=180)
fig.savefig(OUT / "fig12_e1_cost.pdf")
plt.close(fig)

# per-source grouped bars (emphasised runs + G1 / Htau)
SRC = ["mme_realworld", "vstar", "worldbench", "mmmu_pro"]
PS = {"GRPO": [.6645, .8438, .3347, .3141], "DAPO": [.6881, .8728, .3468, .3526],
      "G1 R-read": [.6645, .8527, .3266, .3077], "Htau": [.6792, .8571, .3306, .3205],
      "A+B": [.6659, .8371, .3952, .3397], "A+B+C": [.6563, .8728, .3387, .3013]}
fig, ax = plt.subplots(figsize=(13, 6.5), layout="constrained")
w = 0.13
for j, (run, vals) in enumerate(PS.items()):
    em = run in EMPH
    for i, v in enumerate(vals):
        ax.bar(i + (j - 2.5) * w, v * 100, w, color=COL[run], alpha=1 if em else 0.35,
               edgecolor="black" if run in ("A+B", "A+B+C") else None, label=STYLE[run][0] if i == 0 else None)
        ax.text(i + (j - 2.5) * w, v * 100 + 0.6, f"{v*100:.1f}", ha="center", fontsize=8.5, rotation=90)
ax.set_xticks(range(4))
ax.set_xticklabels(["MME-RealWorld (1,356)", "V* (448)", "WorldBench (248)", "MMMU-Pro (156)"], fontsize=11.5)
ax.set_ylabel("accuracy (%)", fontsize=11.5)
ax.set_ylim(25, 95)
ax.set_title("E1 by source (n = rollouts per source)", fontsize=12.5)
ax.grid(alpha=.2, axis="y")
ax.legend(fontsize=9.5, loc="upper right")
fig.suptitle("By source: A+B gains on WorldBench (+6.1 vs GRPO), A+B+C on V* (+2.9); MME-RealWorld within +-1\n" + SUB,
             fontsize=12)
fig.savefig(OUT / "fig13_e1_by_source.png", dpi=180)
fig.savefig(OUT / "fig13_e1_by_source.pdf")
plt.close(fig)
print("saved", sorted(p.name for p in OUT.glob("*.png")))
