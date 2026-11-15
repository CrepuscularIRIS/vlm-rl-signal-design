"""Mentor report figures (11-14): training curves of the two standard-GRPO arms and the equal-budget trade-off.
Inputs (data/): <arm>.training_metrics.csv (Codex per-step table), <arm>.kl_raw.txt (entropy_loss / ppo_kl lines grepped
from train.log, one pair per step), all8/benchmark_scores.csv. Writes fig_training.{png,pdf}, fig_tradeoff.{png,pdf},
training_long.csv, tradeoff.csv next to this script.
usage: python plot_report.py"""
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
D = HERE / "data"
ARMS = {"q4b_fv2prim_grpo8_pv2": ("SFT -> GRPO", "tab:blue"), "q4b_base_native_grpo8_pv2": ("Base -> GRPO", "tab:orange")}


def arm_rows(arm):
    rows = list(csv.DictReader(open(D / f"{arm}.training_metrics.csv")))
    kl = [float(m) for m in re.findall(r"ppo_kl: ([-0-9.e]+)", (D / f"{arm}.kl_raw.txt").read_text())]
    assert len(kl) == len(rows), (arm, len(kl), len(rows))
    out = []
    for r, k in zip(rows, kl):
        n, g = int(r["n"]), int(r["groups"])
        out.append({"arm": arm, "step": int(r["step"]), "acc": float(r["reward"]), "entropy": float(r["entropy_loss"]),
                    "ppo_kl": k, "crops": float(r["mean_crops"]), "cap5": int(r["cap5"]) / n,
                    "gen_per_traj": int(r["gen_tokens"]) / n, "zero_var": (int(r["all_correct"]) + int(r["all_wrong"])) / g,
                    "step_s": float(r["total_s"])})
    return out


def training_fig(rows):
    panels = [("acc", "Rollout accuracy (training reward, 0/1)", "accuracy"),
              ("crops", "Mean executed crops / trajectory (cap 5)", "crops"),
              ("cap5", "Share of trajectories using all 5 crops", "share"),
              ("gen_per_traj", "Generated tokens / trajectory", "tokens"),
              ("entropy", "Logged actor entropy (entropy_loss)", "entropy"),
              ("ppo_kl", "Policy diff: logged ppo_kl (log often rounds to 1e-3)", "ppo_kl"),
              ("zero_var", "All-correct + all-wrong groups (zero advantage)", "share of groups"),
              ("step_s", "Wall time per step (4 GPUs)", "seconds")]
    fig, axes = plt.subplots(2, 4, figsize=(19, 8), layout="constrained", dpi=180)
    for ax, (key, title, yl) in zip(axes.flat, panels):
        for arm, (name, col) in ARMS.items():
            R = [r for r in rows if r["arm"] == arm]
            ax.plot([r["step"] for r in R], [r[key] for r in R], color=col, lw=2 if arm.startswith("q4b_fv2") else 1.6, label=name)
            ax.annotate(f"{R[-1][key]:.3g}", (R[-1]["step"], R[-1][key]), color=col, fontsize=8, xytext=(3, 0), textcoords="offset points")
        ax.set_title(title, fontsize=9); ax.set_xlabel("step"); ax.set_ylabel(yl); ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=9)
    fig.suptitle("Standard GRPO from SFT vs from Base: SFT arm crops 2.20 -> 3.78 per trajectory; Base arm stays ~3.4-3.9\n"
                 "Same 13k pool (one pass, 51 steps), n=8, clip .2/.3, T 1.0, outcome-only 0/1 reward, single seed; protocols differ "
                 "(SFT = caption v5.1 [0,1]; Base = native zoom tool [0,1000]); Base arm skips zero-advantage groups", fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"fig_training.{ext}")


def tradeoff_fig():
    rows = [r for r in csv.DictReader(open(D / "all8/benchmark_scores.csv")) if r["subset"] in ("all9", "core7")]
    out = []
    for r in rows:
        ends = eval(r["ends"])  # noqa: S307  file written by our own report script
        out.append({"subset": r["subset"], "cell": r["cell"], "acc": float(r["strict_pct"]), "crops": float(r["mean_crops"]),
                    "gen_k": int(r["gen_tokens"]) / 1000, "truncated": ends.get("truncated", 0), "n": int(r["n"])})
    col = {"Base": "tab:gray", "SFT": "tab:green", "BaseGRPO": "tab:orange", "SFTGRPO": "tab:blue"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained", dpi=180)
    for ax, (xk, xl) in zip(axes[:2], (("crops", "mean crops / trajectory"), ("gen_k", "generated tokens, thousands (192 trajectories)"))):
        for r in out:
            if r["subset"] != "all9":
                continue
            m, b = r["cell"].rsplit("-", 1)[0].split("-")[0], r["cell"][-3:].lstrip("-")
            ax.scatter(r[xk], r["acc"], s=70, color=col[m], marker="o" if b == "B5" else "s", edgecolor="k", lw=.5,
                       label=f"{m} {b}")
            ax.annotate(f"{m} {b}", (r[xk], r["acc"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
        ax.set_xlabel(xl); ax.set_ylabel("strict accuracy, 9 benchmarks (%)"); ax.grid(alpha=.2)
    axes[0].set_title("Accuracy vs crops (circle = budget 5, square = budget 11)", fontsize=9)
    axes[1].set_title("Accuracy vs generated tokens", fontsize=9)
    ax = axes[2]
    cells = [r for r in out if r["subset"] == "all9"]
    ax.bar(range(len(cells)), [r["truncated"] for r in cells], color=[col[r["cell"].split("-")[0]] for r in cells])
    ax.set_xticks(range(len(cells)), [r["cell"].replace("-Default", "").replace("-Cap", "") for r in cells], rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("trajectories hitting the 8,192-token limit (of 192)"); ax.set_title("Truncated generations", fontsize=9); ax.grid(alpha=.2, axis="y")
    fig.suptitle("Equal crop budget: both GRPO arms reach 55.7-56.8%; SFT->GRPO pays in crops (5.45 at B11), Base->GRPO pays in tokens (18-22 truncated)\n"
                 "192 items per cell (MME-Lite 16x4 + 128 others x1), T 0.7, strict scoring; small n (16 items per non-MME benchmark)", fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"fig_tradeoff.{ext}")
    return out


def write(rows, name):
    with open(HERE / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    tr = arm_rows("q4b_fv2prim_grpo8_pv2") + arm_rows("q4b_base_native_grpo8_pv2")
    write(tr, "training_long.csv"); training_fig(tr)
    write(tradeoff_fig(), "tradeoff.csv")
    print("ok")
