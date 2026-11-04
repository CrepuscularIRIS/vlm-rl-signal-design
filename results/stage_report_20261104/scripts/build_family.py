"""Assemble per-update-step training metrics for every same-protocol RL run into family_long.csv (run, step, metric,
value, source). Metrics: acc (rollout task accuracy), crops, gen_tok (per trajectory), reward (logged mean),
entropy, ppo_kl, grad_norm, adv_nz (share of response tokens with nonzero advantage). Missing = not logged.
python build_family.py
"""
import csv
import os
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

H = Path(__file__).parent
RUNS = H.parents[1]
REPO = Path(os.environ.get("DESIGN_REPO", "."))  # public copy: set DESIGN_REPO
sys.path.insert(0, str(RUNS.parent / "src"))
from train_curves import parse_log  # noqa: E402

rows = []


def add(run, step, metric, value, source):
    if value is not None:
        rows.append({"run": run, "step": int(step), "metric": metric, "value": float(value), "source": source})


# 1) frozen algorithm curves (GRPO, DAPO, GSPO, GiGPO, RLOO control, RLOO decision)
NAME = {"grpo": "GRPO", "dapo": "DAPO", "gspo": "GSPO", "gigpo": "GiGPO", "rloo": "RLOO", "decision": "RLOO-decision"}
src = "vlm-rl-signal-design/data/algorithm_curves/curves.csv"
for r in csv.DictReader(open(REPO / "data/algorithm_curves/curves.csv")):
    run = NAME[r["method"]]
    for m, k in (("acc", "generated_accuracy"), ("crops", "crops"), ("gen_tok", "generated_tokens"),
                 ("adv_nz", "adv_nonzero_token_share"), ("entropy", "entropy")):
        add(run, r["step"], m, r[k] if r[k] != "" else None, src)

# 2) logged actor metrics for the k8 runs
k8 = json.load(open(RUNS / "k8_20261028/curves_k8.json"))
for key, run in (("k8_grpo", "GRPO"), ("k8_dapo", "DAPO"), ("k8_gspo", "GSPO"), ("k8_gigpo", "GiGPO")):
    for s, m in k8[key].items():
        if int(s) == 0:
            continue
        add(run, s, "reward", m.get("critic/rewards/mean"), "k8_20261028/curves_k8.json")
        add(run, s, "ppo_kl", m.get("actor/ppo_kl"), "k8_20261028/curves_k8.json")
        add(run, s, "grad_norm", m.get("actor/grad_norm"), "k8_20261028/curves_k8.json")


def ledger_metrics(run, path, src, cont_ok=True):
    R = [json.loads(l) for l in open(path)]
    for s in range(1, len(R) // 1024 + 1):
        B = [r for r in R[(s - 1) * 1024:s * 1024] if cont_ok or not r.get("cont")]
        add(run, s, "acc", st.mean(r["correct"] for r in B), src)
        add(run, s, "crops", st.mean(r["crops"] for r in B), src)
        add(run, s, "gen_tok", st.mean(r["gen_tokens"] for r in B), src)


def log_metrics(run, path, src, entropy=True):
    for s, m in parse_log(str(path)).items():
        if int(s) == 0:
            continue
        add(run, s, "reward", m.get("critic/rewards/mean"), src)
        if entropy:
            add(run, s, "entropy", m.get("actor/entropy_loss"), src)  # = mean token entropy (EasyR1 name)
        add(run, s, "ppo_kl", m.get("actor/ppo_kl"), src)
        add(run, s, "grad_norm", m.get("actor/grad_norm"), src)


def adv_nz(run, path, src):
    for i, l in enumerate(open(path), 1):
        add(run, i, "adv_nz", json.loads(l).get("adv_nonzero_token_share"), src)


# 3) G1 (R-read), A+B, A+B+C: logs + ledgers + update ledgers
G1 = RUNS / "rread_adm_20261103/data/curves"
for s, m in json.load(open(G1 / "g1_curves.json"))["g1_rread"].items():
    if int(s) == 0:
        continue
    for met, k in (("reward", "critic/rewards/mean"), ("entropy", "actor/entropy_loss"), ("ppo_kl", "actor/ppo_kl"),
                   ("grad_norm", "actor/grad_norm")):
        add("G1 R-read", s, met, m.get(k), "rread_adm_20261103/data/curves/g1_curves.json")
ledger_metrics("G1 R-read", G1 / "g1_rollouts.jsonl", "g1_rollouts.jsonl")
adv_nz("G1 R-read", H / "d2_g1_rread_updates.jsonl", "d2_g1_rread/updates.jsonl")
SG = RUNS / "sg_20261104/data"
for tag, run in (("ab", "A+B"), ("abc", "A+B+C")):
    for s, m in json.load(open(SG / "sg_curves.json"))[f"sg_{tag}"].items():
        if int(s) == 0:
            continue
        for met, k in (("reward", "critic/rewards/mean"), ("entropy", "actor/entropy_loss"),
                       ("ppo_kl", "actor/ppo_kl"), ("grad_norm", "actor/grad_norm")):
            add(run, s, met, m.get(k), f"sg_{tag}.log")
    ledger_metrics(run, SG / f"sg_{tag}_rollouts.jsonl", f"sg_{tag}/rollouts.jsonl", cont_ok=False)
    adv_nz(run, SG / f"sg_{tag}_updates.jsonl", f"sg_{tag}/updates.jsonl")

# 4) Htau (DAPO + tau-gated coverage, filtered): map each rollout call to the update that kept its groups
log_metrics("Htau", H / "d2_ht5.log", "d2_ht5.log")
adv_nz("Htau", H / "d2_ht5_updates.jsonl", "d2_ht5/updates.jsonl")
upd = [set(json.loads(l)["qids"]) for l in open(H / "d2_ht5_updates.jsonl")]
calls = defaultdict(list)
for l in open(H / "d2_ht5_rollouts.jsonl"):
    r = json.loads(l)
    calls[r["step"]].append(r)
u, per_update = 0, defaultdict(list)
for c in sorted(calls):
    q = {r["qid"] for r in calls[c]}
    while u < len(upd) - 1 and not (q & upd[u]) and any(q & x for x in upd[u + 1:]):
        u += 1
    per_update[u + 1].extend(calls[c])
    if q & upd[u] and len(upd[u] - set().union(*[{r["qid"] for r in per_update[u + 1]}])) == 0:
        u += 1  # this update's kept groups are all generated
for s, B in sorted(per_update.items()):
    if s <= 32:
        add("Htau", s, "acc", st.mean(r["correct"] for r in B), "d2_ht5/rollouts.jsonl (all generated, incl. filtered)")
        add("Htau", s, "crops", st.mean(r["crops"] for r in B), "d2_ht5/rollouts.jsonl")
        add("Htau", s, "gen_tok", st.mean(r["gen_tokens"] for r in B), "d2_ht5/rollouts.jsonl")

# 5) DTPO: ledger + update ledger + rank-0 grad norm; reward from diagnostics; entropy / KL not logged
ledger_metrics("DTPO", H / "dtpo_f1102a2_rollouts.jsonl", "dtpo_f1102a2/rollouts.jsonl")
adv_nz("DTPO", H / "dtpo_f1102a2_updates.jsonl", "dtpo_f1102a2/updates.jsonl")
for r in csv.DictReader(open(REPO / "results/dtpo_e1_20261103/diagnostics.csv")):
    add("DTPO", r["step"], "reward", r["reward"], "results/dtpo_e1_20261103/diagnostics.csv")
g = [json.loads(l)["grad_norm"] for l in open(H / "dtpo_f1102a2_updates_rank0.jsonl")]
k = len(g) // 32
for s in range(1, 33):
    add("DTPO", s, "grad_norm", st.mean(g[(s - 1) * k:s * k]), "dtpo_f1102a2/updates_rank0.jsonl (mean of updates)")

# 6) SAPO: train accuracy points + rank-0 grad norm (1 rollout / question; no ledger, entropy, KL)
for r in csv.DictReader(open(REPO / "results/dtpo_e1_20261103/training_points.csv")):
    if r["method"] == "SAPO" and r["series"] == "train":
        add("SAPO", r["step"], "acc", r["accuracy"], "results/dtpo_e1_20261103/training_points.csv")
g = [json.loads(l)["grad_norm"] for l in open(H / "sapo_f1101a1_updates_rank0.jsonl")]
k = len(g) // 32
for s in range(1, 33):
    add("SAPO", s, "grad_norm", st.mean(g[(s - 1) * k:s * k]), "sapo_f1101a1/updates_rank0.jsonl (mean of updates)")

with open(H / "family_long.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["run", "step", "metric", "value", "source"])
    w.writeheader()
    w.writerows(rows)
cov = defaultdict(set)
for r in rows:
    cov[(r["run"], r["metric"])].add(r["step"])
runs = sorted({r["run"] for r in rows})
mets = ["acc", "reward", "entropy", "ppo_kl", "grad_norm", "crops", "gen_tok", "adv_nz"]
print("run".ljust(14), " ".join(m.rjust(9) for m in mets))
for run in runs:
    print(run.ljust(14), " ".join(str(len(cov.get((run, m), ()))).rjust(9) for m in mets))
