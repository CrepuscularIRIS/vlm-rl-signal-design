"""Cross-method signal audit (2026-11-05): what each method feeds into the update, how its advantage is balanced,
what DAPO's dynamic sampling dropped / refilled, and where E1 gains sit by difficulty. Facts only.

Training side (rollout ledgers, one row per trajectory; updates.jsonl = qids kept per update for filtered runs):
  Y = correct if terminal == "answer" else 0 (V*Y).  Advantage recomputed per method at trajectory level:
  z-score runs: A = (r - mean_g r) / (std_g r + 1e-6), std ddof=1 (EasyR1 torch.std); r per method below.
  A+B / A+B+C: task-scaled advantage (sg audit formula).  DAPO-SB: per turn (Y - b_k)/(sigma_Y + 1e-6), lambda = 1.
  GiGPO: episode part only (step-anchor part needs anchors that are not in the ledger).  Htau: Phi_tau not logged.
  Token weight = generated tokens of the trajectory (DAPO-SB: gen_tokens / turns per turn).
E1 side: 552 questions x 4 samples; paired deltas with question-level bootstrap (2,000 resamples, seed 0).
python audit_signal.py -> audit_signal.json"""
import json
import random
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

H = Path(__file__).parent
R = H.parent
EPS = 1e-6
TRAIN = {  # name: (rollouts, updates or None, reward kind, filter key or None)
    "GRPO": (R / "k8_20261028/k8_grpo/rollouts.jsonl", None, "vy", None),
    "DAPO": (R / "k8_20261028/k8_dapo/rollouts.jsonl", R / "k8_20261028/k8_dapo/updates.jsonl", "vy", "overall"),
    "GSPO": (R / "k8_20261028/k8_gspo/rollouts.jsonl", None, "vy", None),
    "GiGPO": (R / "k8_20261028/k8_gigpo/rollouts.jsonl", None, "vy", None),
    "DTPO": (R / "sg_20261104/family/dtpo_f1102a2_rollouts.jsonl", None, "vy", None),
    "G1": (R / "rread_adm_20261103/data/curves/g1_rollouts.jsonl", None, "g1", None),
    "A+B": (R / "sg_20261104/data/sg_ab_rollouts.jsonl", None, "ts", None),
    "A+B+C": (R / "sg_20261104/data/sg_abc_rollouts.jsonl", None, "ts", None),
    "GRPO+A": (R / "sg_20261104/data/sg_a_rollouts.jsonl", None, "sga", None),
    "Htau": (R / "sg_20261104/family/d2_ht5_rollouts.jsonl", R / "sg_20261104/family/d2_ht5_updates.jsonl", None,
             "accuracy"),
    "DAPO-SB": (R / "sb_20261104/data/rollouts.jsonl", R / "sb_20261104/data/updates.jsonl", "sb", "overall"),
}


def y_of(r):
    return float(r["correct"] >= 1) if r["terminal"] == "answer" else 0.0


def reward(r, kind):
    y = y_of(r)
    v = 1.0 if r["terminal"] == "answer" else 0.0
    if kind == "g1":
        return v * (y + 0.25 * r.get("m", 0) * (r.get("phi") or 0.0))
    if kind == "sga":
        return v * (y + 0.1 * r.get("m", 0) * (r.get("s") or 0.0))
    return y


def zscore(xs):
    m = sum(xs) / len(xs)
    sd = st.stdev(xs)
    return [(x - m) / (sd + EPS) for x in xs]


def ts_adv(g):
    y = [y_of(r) for r in g]
    s = [(0.1 * r.get("m", 0) * (r.get("s") or 0.0)) if r["terminal"] == "answer" else 0.0 for r in g]
    n = len(g)
    my, ms = sum(y) / n, sum(s) / n
    sig = max(st.stdev(y), 1 / n ** 0.5)
    return [((yi - my) + (si - ms)) / sig for yi, si in zip(y, s)]


def load_groups(path):
    calls = defaultdict(list)
    for l in open(path):
        r = json.loads(l)
        calls[r["step"]].append(r)
    out = {}
    for c, rows in calls.items():
        byu = defaultdict(list)
        for r in rows:
            byu[r["uid"]].append(r)
        out[c] = [g for g in byu.values() if len(g) == 8]
    return out


def map_calls(calls, upd):
    """reward call -> update index (filtered runs); same rule as stage_report_20261105/build_family.map_calls."""
    u, res, done = 0, {}, defaultdict(set)
    for c in sorted(calls):
        q = {g[0]["qid"] for g in calls[c]}
        while u < len(upd) - 1 and not (q & upd[u]) and any(q & x for x in upd[u + 1:]):
            u += 1
        res[c] = u
        done[u] |= q
        if q & upd[u] and not (upd[u] - done[u]):
            u += 1
    return res


def k_of(g):
    return int(sum(y_of(r) for r in g))


def status(k):
    return "all_wrong" if k == 0 else "all_correct" if k == 8 else "mixed"


train = {}
dapo_detail = {}
per_step = defaultdict(dict)
for name, (rp, up, kind, fkey) in TRAIN.items():
    calls = load_groups(rp)
    gen = [g for c in calls for g in calls[c]]
    comp = Counter(status(k_of(g)) for g in gen)
    if up:
        upd = [set(json.loads(l)["qids"]) for l in open(up)]
        cu = map_calls(calls, upd)
        updated, seen = [], set()
        for c in sorted(calls):
            s = cu[c]
            for g in calls[c]:
                q = g[0]["qid"]
                ks = k_of(g)
                if q in upd[s] and 0 < ks < 8 and (s, q) not in seen:
                    seen.add((s, q))
                    updated.append((s + 1, g))
    else:
        updated = [(c, g) for c in sorted(calls) for g in calls[c]]
    rec = {"generated_groups": len(gen), "generated_composition": dict(comp), "updated_groups": len(updated),
           "updates": len({s for s, _ in updated})}
    rec["updated_k_hist"] = [sum(1 for _, g in updated if k_of(g) == k) for k in range(9)]
    if kind is None:
        rec["advantage"] = "not recomputable (process score not in ledger)"
        train[name] = rec
        continue
    # advantage per trajectory (or per turn for SB) on updated groups
    pos = neg = pos_wrong = neg_correct = tok_all = tok_nz = 0.0
    cell = defaultdict(list)  # (correct?, zero-crop?) -> mean A over the trajectory
    step_pos, step_neg, step_tok, step_nz = (defaultdict(float) for _ in range(4))
    for s, g in updated:
        if kind == "sb":
            ys = [y_of(r) for r in g]
            sd = st.stdev(ys)
            rows = []
            for r, yv in zip(g, ys):
                n = len(r["b"])
                rows.append((r, yv, [((yv - bk) / (sd + EPS), r["gen_tokens"] / n) for bk in r["b"]]))
        else:
            adv = ts_adv(g) if kind == "ts" else zscore([reward(r, kind) for r in g])
            rows = [(r, y_of(r), [(a, r["gen_tokens"])]) for r, a in zip(g, adv)]
        for r, yv, parts in rows:
            for a, w in parts:
                tok_all += w
                step_tok[s] += w
                if abs(a) > 1e-9:
                    tok_nz += w
                    step_nz[s] += w
                if a > 0:
                    pos += a * w
                    step_pos[s] += a * w
                    pos_wrong += a * w if yv == 0 else 0
                elif a < 0:
                    neg += -a * w
                    step_neg[s] += -a * w
                    neg_correct += -a * w if yv == 1 else 0
            cell[(int(yv), r["crops"] == 0)].append(sum(a for a, _ in parts) / len(parts))
    rec.update({"nonzero_adv_token_share": round(tok_nz / tok_all, 4),
                "pos_over_neg_mass_token_weighted": round(pos / neg, 3),
                "share_of_pos_mass_on_wrong": round(pos_wrong / pos, 4),
                "share_of_neg_mass_on_correct": round(neg_correct / neg, 4),
                "mean_A": {f"{'correct' if c else 'wrong'}_{'zero_crop' if z else 'cropped'}":
                           [len(v), round(st.mean(v), 3)] for (c, z), v in sorted(cell.items())}})
    for s in sorted(step_tok):
        per_step[name][s] = {"pos_neg": round(step_pos[s] / step_neg[s], 3) if step_neg[s] else None,
                             "nz_share": round(step_nz[s] / step_tok[s], 4)}
    if name == "GiGPO":
        rec["note"] = "episode-level advantage only; the step-anchor term is not recomputed"
    train[name] = rec
    # updated-group difficulty per step
    ks = defaultdict(list)
    for s, g in updated:
        ks[s].append(k_of(g))
    for s, v in ks.items():
        per_step[name].setdefault(s, {})
        per_step[name][s].update({"mean_k_updated": round(st.mean(v), 3),
                                  "share_mixed": round(sum(0 < x < 8 for x in v) / len(v), 4),
                                  "share_k_le2": round(sum(0 < x <= 2 for x in v) / len(v), 4),
                                  "share_k_ge6": round(sum(6 <= x < 8 for x in v) / len(v), 4)})
    if name in ("DAPO", "DAPO-SB"):
        upd = [set(json.loads(l)["qids"]) for l in open(TRAIN[name][1])]
        cu = map_calls(calls, upd)
        by_u = defaultdict(lambda: Counter())
        for c in calls:
            for g in calls[c]:
                by_u[cu[c] + 1][status(k_of(g))] += 1
            by_u[cu[c] + 1]["calls"] += 1
        firsts, trans, upk = {}, Counter(), defaultdict(list)
        for c in sorted(calls):
            for g in calls[c]:
                q = g[0]["qid"]
                if q in firsts:
                    trans[(firsts[q], status(k_of(g)))] += 1
                else:
                    firsts[q] = status(k_of(g))
        for s, g in updated:
            upk[g[0]["qid"]].append((s, k_of(g)))
        twice = [v for v in upk.values() if len(v) >= 2]
        src_pool = Counter(q.split("_")[0] for q in firsts)
        src_upd = Counter(g[0]["qid"].split("_")[0] for _, g in updated)
        dapo_detail[name] = {
            "per_update": {s: dict(v) for s, v in sorted(by_u.items())},
            "first_vs_second_generation_status": {f"{a}->{b}": n for (a, b), n in sorted(trans.items())},
            "questions_updated_twice": len(twice),
            "updated_twice_k_first_mean": round(st.mean(v[0][1] for v in twice), 3),
            "updated_twice_k_second_mean": round(st.mean(v[1][1] for v in twice), 3),
            "updated_twice_k_change_hist": dict(sorted(Counter(v[1][1] - v[0][1] for v in twice).items())),
            "updated_twice_step_gap_mean": round(st.mean(v[1][0] - v[0][0] for v in twice), 2),
            "source_share_pool_vs_updated": {k: [round(src_pool[k] / sum(src_pool.values()), 3),
                                                 round(src_upd[k] / sum(src_upd.values()), 3)] for k in src_pool}}

# ---------------- E1 ----------------
F = R / "baselines_v52_20261024/full"
E1 = {"A-SFT": F / "traj_B3_sft.jsonl", "GRPO": F / "traj_k8_grpo.jsonl", "DAPO": F / "traj_k8_dapo.jsonl",
      "GSPO": F / "traj_k8_gspo.jsonl", "GiGPO": F / "traj_k8_gigpo.jsonl", "Htau": F / "traj_ht532.jsonl",
      "DTPO": R / "dtpo_vlm_20261101/review_final/traj_dtpo32.jsonl",
      "G1": R / "rread_adm_20261103/data/e1_g1_rread/traj_g1_rread.jsonl",
      "A+B": R / "sg_20261104/data/traj_sg_ab.jsonl", "A+B+C": R / "sg_20261104/data/traj_sg_abc.jsonl",
      "GRPO+A": R / "sg_20261104/data/traj_sg_a.jsonl", "DAPO-SB": R / "sb_20261104/data/traj_sb_dapo_sb.jsonl"}
Q = {}
for name, p in E1.items():
    d = defaultdict(int)
    n = Counter()
    for r in map(json.loads, open(p)):
        d[r["uid"]] += int(r["ok"])
        n[r["uid"]] += 1
    assert len(d) == 552 and set(n.values()) == {4}, (name, len(d))
    Q[name] = d
qs = sorted(Q["GRPO"])
rng = random.Random(0)
boots = [[rng.choice(qs) for _ in qs] for _ in range(2000)]


def delta_ci(a, b):
    obs = sum(Q[a][q] - Q[b][q] for q in qs) / (4 * len(qs))
    ds = sorted(sum(Q[a][q] - Q[b][q] for q in bs) / (4 * len(qs)) for bs in boots)
    return [round(obs * 100, 2), round(ds[49] * 100, 2), round(ds[1949] * 100, 2)]


e1 = {"acc": {m: round(sum(Q[m].values()) / 2208, 4) for m in Q},
      "delta_vs_GRPO_pts_ci95": {m: delta_ci(m, "GRPO") for m in Q if m != "GRPO"},
      "delta_vs_DAPO_pts_ci95": {m: delta_ci(m, "DAPO") for m in Q if m != "DAPO"}}
bucket = {k: [q for q in qs if Q["A-SFT"][q] == k] for k in range(5)}
e1["asft_bucket_n_questions"] = {k: len(v) for k, v in bucket.items()}
e1["acc_by_asft_bucket"] = {m: {k: round(sum(Q[m][q] for q in v) / (4 * len(v)), 3) for k, v in bucket.items()}
                            for m in Q}
e1["net_correct_vs_GRPO_by_bucket"] = {m: {k: sum(Q[m][q] - Q["GRPO"][q] for q in v) for k, v in bucket.items()}
                                       for m in Q if m != "GRPO"}
json.dump({"train": train, "per_step": per_step, "dapo_detail": dapo_detail, "e1": e1},
          open(H / "audit_signal.json", "w"), indent=1)
for m, v in train.items():
    print(m, {k: v[k] for k in v if k not in ("mean_A", "per_update")})
    if "mean_A" in v:
        print("   mean_A", v["mean_A"])
for m, v in dapo_detail.items():
    print(m, {k: v[k] for k in v if k != "per_update"})
print(json.dumps(e1, indent=0)[:3000])
