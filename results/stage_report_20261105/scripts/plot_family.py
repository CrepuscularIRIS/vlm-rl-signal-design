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
for r in csv.DictReader(open(H.parent / "sg_20261104" / "data" / "sg_curves_long.csv")):
    run = {"sg_ab": "A+B", "sg_abc": "A+B+C"}.get(r["arm"])
    if run:
        for m in ("S_m1", "gate_m1"):
            if r[m]:
                D[(run, m)][int(r["step"])] = float(r[m])

STYLE = {  # run: (label, color, lw, alpha, ls, z)
    "DAPO-SB": ("DAPO + stop-relative advantage (Y - b_k)/sigma, pre-turn stop probe b_k (DAPO-SB)", "#d62728", 3.2, 1.0, "-", 12),
    "GRPO+A": ("GRPO + scale-gated evidence reward S only (GRPO+A)", "#00876c", 3.2, 1.0, "-", 11),
    "A+B": ("GRPO + scale-gated evidence reward + task-scaled advantage (A+B)", "#9467bd", 2.0, 0.85, "-", 9),
    "A+B+C": ("A+B + evidence-continuation rows, steps 1-16 (A+B+C)", "#c5b0d5", 1.6, 0.8, "-", 6),
    "GRPO": ("GRPO (outcome reward V*Y, no filtering)", "#1f77b4", 2.3, 0.95, "-", 8),
    "DAPO": ("DAPO (dynamic-sampling filter, clip-higher)", "#ff7f0e", 2.3, 0.95, "-", 7),
    "GSPO": ("GSPO", "#2ca02c", 1.3, 0.3, "-", 2),
    "GiGPO": ("GiGPO", "#8c564b", 1.3, 0.3, "-", 2),
    "RLOO": ("RLOO", "#7f7f7f", 1.3, 0.3, "-", 2),
    "RLOO-decision": ("RLOO + first-action fork", "#bcbd22", 1.3, 0.3, "-", 2),
    "DTPO": ("DTPO", "#e377c2", 1.3, 0.3, "-", 2),
    "SAPO": ("SAPO (1 rollout / question)", "#17becf", 1.3, 0.3, "-", 2),
    "G1 R-read": ("GRPO + reading-consistency reward (G1)", "#000000", 1.4, 0.35, "--", 3),
    "Htau": ("DAPO + tau-gated box-coverage reward (Htau)", "#c51b7d", 1.4, 0.35, ":", 3),
}
EMPH = ("DAPO-SB", "GRPO+A", "A+B", "GRPO", "DAPO")
BOLD = ("DAPO-SB", "GRPO+A")
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
            missing.append(run)
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
        ax.text(32.4, 0.02, "A+B+C steps 31-32:\ndifferent question mix (A+B+C only)", transform=ax.get_xaxis_transform(),
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
                    fontweight="bold" if e[3] in BOLD else None,
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


SUB = ("Qwen3-VL-4B from A-SFT | 4k pool (one pass) x 8 rollouts x 32 steps | seed 42, lr 1e-6, no KL | "
       "caption v5.1 crop protocol | one seed per run")
L = lambda run, m, a=25, b=32: late(run, m, a, b)  # noqa: E731

figure("acc", "Rollout task accuracy Y in training (all generated rollouts; no continuation rows)",
       "accuracy (fraction of rollouts)",
       f"Training accuracy, steps 25-32 mean: DAPO-SB {L('DAPO-SB','acc'):.3f} vs DAPO {L('DAPO','acc'):.3f}; "
       f"GRPO+A {L('GRPO+A','acc'):.3f} vs GRPO {L('GRPO','acc'):.3f}; A+B {L('A+B','acc'):.3f}", "fig01_train_accuracy")
figure("reward", "Logged mean reward (S arms incl. 0.1*m*S; G1 0.25*m*Phi; DAPO arms: kept groups)",
       "reward", f"Logged reward at step 32: DAPO-SB {D[('DAPO-SB','reward')][32]:.3f}, DAPO {D[('DAPO','reward')][32]:.3f}, "
       f"GRPO+A {D[('GRPO+A','reward')][32]:.3f}, GRPO {D[('GRPO','reward')][32]:.3f}", "fig02_reward")
figure("entropy", "Policy entropy (logged actor/entropy_loss = mean token entropy)", "entropy (nats / token)",
       f"Entropy at step 32: DAPO-SB {D[('DAPO-SB','entropy')][32]:.3f} (from .448) vs DAPO {D[('DAPO','entropy')][32]:.3f}; "
       f"GRPO+A {D[('GRPO+A','entropy')][32]:.3f}, GRPO {D[('GRPO','entropy')][32]:.3f}, A+B {D[('A+B','entropy')][32]:.3f}",
       "fig03_entropy")
figure("ppo_kl", "Policy diff: actor/ppo_kl, update vs rollout policy (k1 estimate, raw per step)", "KL x 1e-4",
       "Policy diff within +-1e-4 for all logged runs except single steps of A+B (step 5) and GRPO+A (step 3), 1.0e-3 each",
       "fig04_policy_diff", raw=True, scale=1e4,
       extra=lambda ax: (ax.set_ylim(-1.2, 3.0),
                         ax.annotate("A+B step 5 and GRPO+A step 3: 10.0 (clipped)", (4, 3.0), xytext=(7, 2.6),
                                     fontsize=10, color="#00876c", arrowprops=dict(arrowstyle="->", color="#00876c"))))
figure("grad_norm", "Grad norm before clipping (max 1.0; DTPO / SAPO: mean over updates)", "grad norm",
       f"Gradient norm, steps 25-32 mean: DAPO-SB {L('DAPO-SB','grad_norm'):.2f}, DAPO {L('DAPO','grad_norm'):.2f}, "
       f"GRPO+A {L('GRPO+A','grad_norm'):.2f}, GRPO {L('GRPO','grad_norm'):.2f}", "fig05_grad_norm")
figure("crops", "Executed crops per trajectory in training (all generated rollouts)",
       "crops / trajectory",
       f"Crops per trajectory, steps 25-32 mean: GRPO+A {L('GRPO+A','crops'):.2f}, A+B {L('A+B','crops'):.2f}, "
       f"DAPO {L('DAPO','crops'):.2f}, GRPO {L('GRPO','crops'):.2f}, DAPO-SB {L('DAPO-SB','crops'):.2f}", "fig06_crops")
figure("gen_tok", "Generated (policy) tokens per trajectory during training", "tokens / trajectory",
       f"Generated tokens, steps 25-32 mean: GRPO+A {L('GRPO+A','gen_tok'):.0f}, A+B {L('A+B','gen_tok'):.0f}, "
       f"DAPO {L('DAPO','gen_tok'):.0f}, GRPO {L('GRPO','gen_tok'):.0f}, DAPO-SB {L('DAPO-SB','gen_tok'):.0f}",
       "fig07_gen_tokens")
figure("adv_nz", "Share of response tokens with nonzero advantage in the update batch", "share of tokens",
       f"Nonzero-advantage share, steps 1-32 mean: DAPO-SB {L('DAPO-SB','adv_nz',1,32):.2f}, DAPO "
       f"{L('DAPO','adv_nz',1,32):.2f}, GRPO+A {L('GRPO+A','adv_nz',1,32):.2f}, GRPO {L('GRPO','adv_nz',1,32):.2f}",
       "fig08_adv_nonzero")
figure("S_m1", "Scale-gated evidence acquisition score S on eligible questions (m=1)", "S (0-1)",
       f"S, steps 25-32 mean: GRPO+A {L('GRPO+A','S_m1'):.2f}, A+B {L('A+B','S_m1'):.2f}; A+B+C "
       f"{L('A+B+C','S_m1',25,30):.2f} (steps 25-30); S not computed for other runs", "fig09_S",
       runs=("GRPO+A", "A+B", "A+B+C"))
figure("gate_m1", "Gate open: some key evidence box reached usable input scale (m=1)", "share of rollouts",
       f"Gate-open share, steps 25-32 mean: GRPO+A {L('GRPO+A','gate_m1'):.2f}, A+B {L('A+B','gate_m1'):.2f}",
       "fig10_gate", runs=("GRPO+A", "A+B", "A+B+C"))

# DAPO-SB probe panel (DAPO-SB only)
fig, ax = plt.subplots(figsize=(13, 7), layout="constrained")
for m, lab, c in (("mean_b0", "probe b0 = P(gt letter) before turn 0, kept groups", "#d62728"),
                  ("mean_groupacc_minus_b0", "group accuracy - b0 (mean over kept groups)", "#8c2d04"),
                  ("mean_letter_mass", "probe option-letter mass", "#6a51a3"),
                  ("share_raw_adv_pos", "share of turns with raw advantage > 0", "#525252"),
                  ("crops_kept", "crops / trajectory in kept groups, plotted / 4 (end label undivided)", "#fd8d3c")):
    d = D[("DAPO-SB", m)]
    xs = sorted(d)
    ys = [d[x] / (4 if m == "crops_kept" else 1) for x in xs]
    ax.plot(xs, ys, color=c, lw=2.4, label=lab)
    ax.annotate(f"{d[xs[-1]]:.3f}", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", fontsize=10, color=c)
ax.set_xlim(0.5, 34.5)
ax.set_xlabel("RL update step", fontsize=12)
ax.set_ylabel("probability / share (crops divided by 4)", fontsize=12)
ax.set_title("DAPO-SB stop probe on the kept (mixed) groups; DAPO has no probe", fontsize=12.5)
ax.grid(alpha=.2)
ax.legend(fontsize=10, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
fig.suptitle(f"Probe b0 {D[('DAPO-SB','mean_b0')][1]:.3f} -> {D[('DAPO-SB','mean_b0')][32]:.3f}; group accuracy - b0 "
             f"stays {min(D[('DAPO-SB','mean_groupacc_minus_b0')].values()):.2f}-"
             f"{max(D[('DAPO-SB','mean_groupacc_minus_b0')].values()):.2f}; raw advantage > 0 on "
             f"{D[('DAPO-SB','share_raw_adv_pos')][1]:.2f} -> {D[('DAPO-SB','share_raw_adv_pos')][32]:.2f} of turns\n" + SUB,
             fontsize=12)
fig.savefig(OUT / "fig14_sb_probe.png", dpi=180)
fig.savefig(OUT / "fig14_sb_probe.pdf")
plt.close(fig)

# E1 bar + cost scatter
E1 = [  # condition, acc, crops/traj, processed tokens/traj, generated tokens/traj
    ("A-SFT (start)", .5670, 0.994, 5015, None), ("GRPO", .6390, 1.905, 8125, 261), ("DAPO", .6635, 2.092, 8930, 287),
    ("GSPO", .6395, 2.263, 9756, 278), ("GiGPO", .6209, 1.998, 8506, 275), ("RLOO", .6318, 1.992, 8535, 278),
    ("RLOO-decision", .5919, 1.372, 6266, 203), ("DTPO", .6553, 3.357, 15579, 395), ("SAPO", .5770, 1.063, 5292, 171),
    ("G1 R-read", .6395, 2.308, 10063, 374), ("Htau", .6508, 2.546, 10872, 305),
    ("A+B", .6472, 2.404, 10047, 342), ("A+B+C", .6395, 2.337, 10181, 326),
    ("GRPO+A", .6436, 4.020, 18034, 440), ("DAPO-SB", .6413, 1.631, 7055, 222)]
COL = {k: v[1] for k, v in STYLE.items()}
COL["A-SFT (start)"] = "#bbbbbb"
fig, ax = plt.subplots(figsize=(13, 7), layout="constrained")
order = sorted(E1, key=lambda r: r[1])
for i, (c, a, *_rest) in enumerate(order):
    em = c in EMPH
    ax.barh(i, a * 100, color=COL[c], alpha=1.0 if em else 0.35, edgecolor="black" if c in BOLD else None, lw=1.5)
    ax.text(a * 100 + 0.15, i, f"{a*100:.2f}", va="center", fontsize=10.5, fontweight="bold" if c in BOLD else None)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([r[0] for r in order], fontsize=11)
ax.set_xlim(55, 68)
ax.axvline(63.90, color="#1f77b4", ls="--", lw=1)
ax.axvline(66.35, color="#ff7f0e", ls="--", lw=1)
ax.set_xlabel("E1 accuracy (%), 552 questions x 4 samples = 2,208; failures kept in denominator", fontsize=11.5)
ax.set_title("Fixed-endpoint E1 (step 32 checkpoints; same scorer); dashed lines = GRPO (blue), DAPO (orange)",
             fontsize=12.5)
ax.grid(alpha=.2, axis="x")
fig.suptitle("E1: DAPO 66.35, A+B 64.72, GRPO+A 64.36, DAPO-SB 64.13, GRPO 63.90 | paired net (of 2,208): "
             "DAPO over DAPO-SB +49, GRPO+A over GRPO +10\n" + SUB, fontsize=11.5)
fig.savefig(OUT / "fig11_e1_scores.png", dpi=180)
fig.savefig(OUT / "fig11_e1_scores.pdf")
plt.close(fig)

OFF = {"GSPO": (-60, -20), "G1 R-read": (8, 14), "A+B+C": (10, -16), "DTPO": (-170, 8), "GRPO+A": (-190, 8),
       "DAPO-SB": (-70, 14), "RLOO": (8, -14)}
fig, ax = plt.subplots(figsize=(13.5, 7), layout="constrained")
for c, a, cr, tok, g in E1:
    em = c in EMPH
    ax.scatter(tok, a * 100, s=220 if em else 110, color=COL[c], alpha=1 if em else 0.45,
               edgecolor="black" if c in BOLD else "none", lw=1.5, zorder=3)
    ax.annotate(f"{c} ({cr:.2f} crops)", (tok, a * 100), xytext=OFF.get(c, (8, 4)), textcoords="offset points",
                fontsize=10, fontweight="bold" if c in BOLD else None, color="black" if em else "#555555")
ax.set_xlabel("processed tokens per trajectory on E1 (sum over turns of input + output)", fontsize=11.5)
ax.set_ylabel("E1 accuracy (%)", fontsize=11.5)
ax.set_title("E1 accuracy vs inference cost (labels: mean crops per trajectory)", fontsize=12.5)
ax.grid(alpha=.2)
fig.suptitle("Cost: DAPO-SB 7.1k tokens / 1.63 crops (DAPO 8.9k / 2.09); GRPO+A 18.0k / 4.02 (GRPO 8.1k / 1.91)\n" + SUB,
             fontsize=12)
fig.savefig(OUT / "fig12_e1_cost.png", dpi=180)
fig.savefig(OUT / "fig12_e1_cost.pdf")
plt.close(fig)

SRC = ["mme_realworld", "vstar", "worldbench", "mmmu_pro"]
PS = {"GRPO": [.6645, .8438, .3347, .3141], "GRPO+A": [.6689, .8527, .3347, .3141], "A+B": [.6659, .8371, .3952, .3397],
      "DAPO": [.6881, .8728, .3468, .3526], "DAPO-SB": [.6799, .8326, .2903, .3141],
      "Htau": [.6792, .8571, .3306, .3205], "A+B+C": [.6563, .8728, .3387, .3013]}
fig, ax = plt.subplots(figsize=(14, 6.8), layout="constrained")
w = 0.115
for j, (run, vals) in enumerate(PS.items()):
    em = run in EMPH
    for i, v in enumerate(vals):
        x = i + (j - (len(PS) - 1) / 2) * w
        ax.bar(x, v * 100, w, color=COL[run], alpha=1 if em else 0.35, edgecolor="black" if run in BOLD else None,
               label=STYLE[run][0] if i == 0 else None)
        ax.text(x, v * 100 + 0.6, f"{v*100:.1f}", ha="center", fontsize=8.5, rotation=90)
ax.set_xticks(range(4))
ax.set_xticklabels(["MME-RealWorld (1,356)", "V* (448)", "WorldBench (248)", "MMMU-Pro (156)"], fontsize=11.5)
ax.set_ylabel("accuracy (%)", fontsize=11.5)
ax.set_ylim(25, 97)
ax.set_title("E1 by source (n = rollouts per source)", fontsize=12.5)
ax.grid(alpha=.2, axis="y")
ax.legend(fontsize=9, loc="upper right")
fig.suptitle("DAPO-SB minus DAPO (pts): MME -0.8, V* -4.0, WB -5.6, MMMU -3.8 | GRPO+A minus GRPO: "
             "MME +0.4, V* +0.9, WB 0, MMMU 0\n" + SUB, fontsize=11.5)
fig.savefig(OUT / "fig13_e1_by_source.png", dpi=180)
fig.savefig(OUT / "fig13_e1_by_source.pdf")
plt.close(fig)
print("saved", sorted(p.name for p in OUT.glob("*.png")))
