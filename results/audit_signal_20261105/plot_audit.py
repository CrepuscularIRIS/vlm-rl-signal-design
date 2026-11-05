"""Figures for the 2026-11-05 cross-method signal audit (English text). Input: audit_signal.json.
python plot_audit.py -> figs/a1..a6 (png + pdf)"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

H = Path(__file__).parent
OUT = H / "figs"
OUT.mkdir(exist_ok=True)
d = json.load(open(H / "audit_signal.json"))
T, PS, DD, E = d["train"], d["per_step"], d["dapo_detail"], d["e1"]
S = list(range(1, 33))
COL = {"GRPO": "#1f77b4", "DAPO": "#ff7f0e", "DAPO-SB": "#d62728", "A+B": "#9467bd", "GRPO+A": "#00876c",
       "GSPO": "#2ca02c", "GiGPO": "#8c564b", "DTPO": "#e377c2", "G1": "#000000", "A+B+C": "#c5b0d5",
       "Htau": "#c51b7d", "A-SFT": "#bbbbbb"}
SUB = "Qwen3-VL-4B from A-SFT | 4k pool x 8 rollouts x 32 updates x 128 groups | one seed per run"


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=180)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


# a1: what enters each update
fig, ax = plt.subplots(1, 2, figsize=(16, 6.5), layout="constrained", sharey=True)
mixed = [round(PS["GRPO"][str(s)]["share_mixed"] * 128) for s in S]
k8 = {"all_correct": [], "all_wrong": []}
calls = {}
import collections  # noqa: E402
grows = collections.defaultdict(collections.Counter)
for l in open(H.parent / "k8_20261028/k8_grpo/rollouts.jsonl"):
    r = json.loads(l)
    grows[(r["step"], r["uid"])][int(r["correct"] >= 1 and r["terminal"] == "answer")] += 1
for s in S:
    ks = [c[1] for (st_, _), c in grows.items() if st_ == s]
    k8["all_correct"].append(sum(k == 8 for k in ks))
    k8["all_wrong"].append(sum(k == 0 for k in ks))
a = ax[0]
a.bar(S, mixed, color="#1f77b4", label="mixed groups (nonzero advantage)")
a.bar(S, k8["all_correct"], bottom=mixed, color="#9ecae1", label="all-correct groups (advantage 0)")
a.bar(S, k8["all_wrong"], bottom=[m + c for m, c in zip(mixed, k8["all_correct"])], color="#dddddd",
      label="all-wrong groups (advantage 0)")
a.set_title(f"GRPO: the 128 groups of each update (mixed: mean {sum(mixed)/32:.1f} per update)", fontsize=11.5)
a.set_xlabel("update step")
a.set_ylabel("groups (8 rollouts each)")
a.legend(fontsize=9, loc="upper left")
pu = DD["DAPO"]["per_update"]
mx = [pu[str(s)].get("mixed", 0) for s in S]
ac = [pu[str(s)].get("all_correct", 0) for s in S]
aw = [pu[str(s)].get("all_wrong", 0) for s in S]
kept = [128] * 32
surplus = [max(0, m - 128) for m in mx]
a = ax[1]
a.bar(S, kept, color="#ff7f0e", label="kept mixed groups -> update (128)")
a.bar(S, surplus, bottom=kept, color="#fdd0a2", label="mixed groups generated but not used")
a.bar(S, ac, bottom=[k + u for k, u in zip(kept, surplus)], color="#9ecae1", label="dropped: all-correct")
a.bar(S, aw, bottom=[k + u + c for k, u, c in zip(kept, surplus, ac)], color="#dddddd", label="dropped: all-wrong")
b = a.twinx()
b.plot(S, [pu[str(s)]["calls"] for s in S], color="black", marker="o", ms=3, lw=1.2, label="generation rounds")
b.set_ylabel("generation rounds (x 32 groups) per update")
b.set_ylim(0, 12)
a.set_title(f"DAPO: generated per update (mean {sum(mx)/32:.0f} mixed, {sum(ac)/32:.0f} all-correct, "
            f"{sum(aw)/32:.0f} all-wrong)", fontsize=11.5)
a.set_xlabel("update step")
a.legend(handles=a.get_legend_handles_labels()[0] + b.get_lines(), fontsize=9, loc="upper left")
for x in ax:
    x.grid(alpha=.2, axis="y")
fig.suptitle("Groups entering each update: GRPO updates on ~72 mixed + ~56 zero-advantage groups; DAPO on 128 mixed "
             "groups after dropping ~110 all-same groups\n" + SUB, fontsize=12)
save(fig, "a1_update_composition")

# a2: difficulty of updated groups
fig, ax = plt.subplots(1, 2, figsize=(16, 6), layout="constrained")
runs = ["GRPO", "DAPO", "DAPO-SB", "A+B", "GRPO+A"]
w = 0.16
for j, m in enumerate(runs):
    h = T[m]["updated_k_hist"]
    tot = sum(h[1:8])
    ax[0].bar([k + (j - 2) * w for k in range(1, 8)], h[1:8], w, color=COL[m], label=f"{m} ({tot} mixed groups)")
    ax[1].bar([k + (j - 2) * w for k in range(1, 8)], [x / tot for x in h[1:8]], w, color=COL[m], label=m)
ax[0].set_title("Mixed groups in all 32 updates, by #correct of 8 (counts)", fontsize=11.5)
ax[1].set_title("Same, as share of each run's mixed groups", fontsize=11.5)
for x in ax:
    x.set_xlabel("#correct rollouts in the group (of 8)")
    x.set_xticks(range(1, 8))
    x.grid(alpha=.2, axis="y")
ax[0].set_ylabel("groups")
ax[1].set_ylabel("share")
ax[0].legend(fontsize=9)
fig.suptitle("DAPO / DAPO-SB update on ~1.77x as many mixed groups as GRPO; the difficulty shape of mixed groups is "
             "similar across runs\n" + SUB, fontsize=12)
save(fig, "a2_updated_group_difficulty")

# a3: positive / negative advantage mass per step
fig, ax = plt.subplots(figsize=(13, 6.5), layout="constrained")
for m in ["GSPO", "GiGPO", "DTPO", "G1", "A+B+C", "GRPO", "A+B", "GRPO+A", "DAPO", "DAPO-SB"]:
    ys = [PS[m][str(s)]["pos_neg"] for s in S]
    em = m in ("GRPO", "DAPO", "DAPO-SB", "A+B", "GRPO+A")
    ax.plot(S, ys, color=COL[m], lw=2.6 if m == "DAPO-SB" else 2.0 if em else 1.1, alpha=1 if em else .4,
            label=f"{m} (all steps {T[m]['pos_over_neg_mass_token_weighted']:.2f})")
ax.axhline(1, color="black", lw=.8, ls="--")
ax.set_xlabel("update step")
ax.set_ylabel("positive / negative advantage mass (token-weighted)")
ax.set_title("Advantage balance in the update batch, recomputed from rollout ledgers with each method's formula",
             fontsize=11.5)
ax.grid(alpha=.2)
ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.01, .5), frameon=False)
fig.suptitle("Positive / negative mass: DAPO-SB 2.23; all group-mean methods 0.91-0.97 (wrong trajectories are longer)"
             "\n" + SUB + " | GiGPO: episode term only; Htau: not recomputable", fontsize=12)
save(fig, "a3_advantage_balance")

# a4: DAPO repeated questions
fig, ax = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
st_ = ["all_wrong", "mixed", "all_correct"]
tr = DD["DAPO"]["first_vs_second_generation_status"]
M = [[tr.get(f"{a}->{b}", 0) for b in st_] for a in st_]
im = ax[0].imshow(M, cmap="Oranges")
for i in range(3):
    for j in range(3):
        ax[0].text(j, i, M[i][j], ha="center", va="center", fontsize=12)
ax[0].set_xticks(range(3))
ax[0].set_xticklabels([f"2nd: {s}" for s in st_])
ax[0].set_yticks(range(3))
ax[0].set_yticklabels([f"1st: {s}" for s in st_])
ax[0].set_title("DAPO: status of a question at its 1st vs 2nd generation (3,776 questions)", fontsize=11.5)
for j, m in enumerate(["DAPO", "DAPO-SB"]):
    hh = DD[m]["updated_twice_k_change_hist"]
    xs = sorted(int(k) for k in hh)
    ax[1].bar([x + (j - .5) * .4 for x in xs], [hh[str(x)] for x in xs], .4, color=COL[m],
              label=f"{m}: {DD[m]['questions_updated_twice']} questions, mean k {DD[m]['updated_twice_k_first_mean']:.2f}"
                    f" -> {DD[m]['updated_twice_k_second_mean']:.2f}")
ax[1].set_xlabel("change in #correct of 8 between the two updates of the same question")
ax[1].set_ylabel("questions")
ax[1].set_title(f"Questions updated twice (mean gap {DD['DAPO']['updated_twice_step_gap_mean']} updates)",
                fontsize=11.5)
ax[1].legend(fontsize=9.5)
ax[1].grid(alpha=.2, axis="y")
fig.suptitle("DAPO refills from a second pass over the pool: 154 all-correct and 181 all-wrong questions became mixed;"
             " 385 mixed became all-correct\n" + SUB, fontsize=12)
save(fig, "a4_dapo_second_pass")

# a5: E1 deltas with CI
fig, ax = plt.subplots(1, 2, figsize=(16, 6.5), layout="constrained")
for a, key, ref in ((ax[0], "delta_vs_GRPO_pts_ci95", "GRPO"), (ax[1], "delta_vs_DAPO_pts_ci95", "DAPO")):
    items = sorted(((m, v) for m, v in E[key].items() if m != "A-SFT"), key=lambda kv: kv[1][0])
    for i, (m, (o, lo, hi)) in enumerate(items):
        a.plot([lo, hi], [i, i], color=COL[m], lw=2.5)
        a.scatter([o], [i], color=COL[m], s=60, zorder=3)
        a.text(hi + .15, i, f"{o:+.2f} [{lo:+.2f}, {hi:+.2f}]", va="center", fontsize=9.5)
    a.axvline(0, color="black", lw=.8)
    a.set_yticks(range(len(items)))
    a.set_yticklabels([m for m, _ in items], fontsize=10.5)
    a.set_xlabel(f"E1 accuracy minus {ref} (points); 95% CI, question-level bootstrap (552 questions, 2,000 draws)")
    a.set_title(f"vs {ref}", fontsize=12)
    a.grid(alpha=.2, axis="x")
    a.set_xlim(-7, 7.5)
fig.suptitle("Only DAPO's gain over GRPO has a 95% CI above 0; vs DAPO, Htau / DTPO / A+B / GRPO+A CIs include 0 and "
             "the rest are below\n" + SUB, fontsize=12)
save(fig, "a5_e1_delta_ci")

# a6: E1 by A-SFT difficulty bucket
fig, ax = plt.subplots(figsize=(13, 6.5), layout="constrained")
nb = E["asft_bucket_n_questions"]
for m in ["A-SFT", "GSPO", "GiGPO", "G1", "A+B+C", "Htau", "DTPO", "GRPO+A", "A+B", "GRPO", "DAPO-SB", "DAPO"]:
    ys = [E["acc_by_asft_bucket"][m][str(k)] for k in range(5)]
    em = m in ("GRPO", "DAPO", "DAPO-SB", "A+B", "GRPO+A")
    ax.plot(range(5), ys, marker="o", color=COL[m], lw=2.4 if em else 1.1, alpha=1 if em or m == "A-SFT" else .45,
            label=m)
ax.set_xticks(range(5))
ax.set_xticklabels([f"A-SFT {k}/4 correct\n({nb[str(k)]} questions)" for k in range(5)])
ax.set_ylabel("E1 accuracy within bucket")
ax.set_title("E1 accuracy by the A-SFT starting point's per-question accuracy (4 samples)", fontsize=11.5)
ax.grid(alpha=.2)
ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.01, .5), frameon=False)
g = E["net_correct_vs_GRPO_by_bucket"]["DAPO"]
fig.suptitle(f"DAPO's net gain over GRPO by bucket (0..4): {', '.join(f'{v:+d}' for v in g.values())} rollouts\n" + SUB,
             fontsize=12)
save(fig, "a6_e1_by_asft_bucket")
print(sorted(p.name for p in OUT.glob("*.png")))
