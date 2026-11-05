"""D0 within-group contrast (2026-11-05). Question: inside groups that carry gradient (mixed groups), where does the
difference between correct and wrong trajectories sit -- evidence acquisition, after evidence, or unrelated to evidence?
Facts only; CPU; reads existing ledgers + rl4k L0/L1/L2 + E1 trajectories.

Evidence state per trajectory (ledger fields from reward_sg.process_score, scale thresholds UNCALIBRATED):
  EV   = phi == 1   : every box of some L2 evidence_set reached usable scale in some presented view (original incl.)
  EV0  = phi0 == 1  : the original view alone already gives that (question-level, policy-independent)
  GEO  = every box of the first evidence_set is >=90% covered by one executed crop (geometry only, no scale)
Eligible questions (m == 1): answerable, gt agree, has L1 + complete evidence_set (reward_sg index).
Pairs: every (correct i, wrong j) pair inside a mixed group (8 rollouts, 1..7 correct).
python d0_within_group.py -> d0.json (+ prints)"""
import ast
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

H = Path(__file__).parent
R = H.parent
ANN = R.parent / "notes/rl_pool_annotation_20261101/out/final/rl4k"
sys.path.insert(0, str(R.parent / "notes/rereview_20261104"))


def jl(p):
    return [json.loads(l) for l in open(p)]


L0 = {r["uid"]: r for r in jl(ANN / "L0/batch.jsonl")}
L1 = {r["uid"]: r for r in jl(ANN / "L1/batch.jsonl")}
L2 = {r["uid"]: r for r in jl(ANN / "L2/batch.jsonl")}


def y_of(r):
    return int(r["terminal"] == "answer" and r["correct"] >= 1)


def geo_full(r):
    l1 = L1.get(r["qid"])
    if not l1 or not r.get("wh"):
        return None
    W, Hh = r["wh"]
    boxes = {b["id"]: b["bbox"] for b in l1["boxes"]}
    sets = (L2.get(r["qid"]) or {}).get("evidence_sets") or [list(boxes)]
    ids = [i for i in sets[0] if i in boxes]
    if not ids:
        return None
    crops = r.get("boxes") or []

    def cov(c, b):
        x1, y1, x2, y2 = c
        bx1, by1, bx2, by2 = b[0] * W, b[1] * Hh, b[2] * W, b[3] * Hh
        iw, ih = max(0, min(x2, bx2) - max(x1, bx1)), max(0, min(y2, by2) - max(y1, by1))
        a = (bx2 - bx1) * (by2 - by1)
        return iw * ih / a if a > 0 else 0
    return int(all(any(cov(c, boxes[i]) >= 0.9 for c in crops) for i in ids))


def groups_by_update(rollouts, updates=None):
    """-> {update_step: [group rows]} ; filtered runs map reward calls to updates (same rule as audit_signal)."""
    calls = defaultdict(lambda: defaultdict(list))
    for r in rollouts:
        calls[r["step"]][r["uid"]].append(r)
    if updates is None:
        return {c: [g for g in calls[c].values() if len(g) == 8] for c in calls}
    upd = [set(json.loads(l)["qids"]) for l in open(updates)]
    u, res, done = 0, defaultdict(list), defaultdict(set)
    for c in sorted(calls):
        q = {g[0]["qid"] for g in calls[c].values()}
        while u < len(upd) - 1 and not (q & upd[u]) and any(q & x for x in upd[u + 1:]):
            u += 1
        res[u + 1] += [g for g in calls[c].values() if len(g) == 8]
        done[u] |= q
        if q & upd[u] and not (upd[u] - done[u]):
            u += 1
    return res


def auc_within(groups, f):
    """pair-weighted within-group AUC of score f vs Y over mixed groups (ties = .5); None scores skipped."""
    num = den = 0.0
    for g in groups:
        c = [f(r) for r in g if y_of(r)]
        w = [f(r) for r in g if not y_of(r)]
        for a in c:
            for b in w:
                if a is None or b is None:
                    continue
                num += 1.0 if a > b else 0.5 if a == b else 0.0
                den += 1
    return round(num / den, 4) if den else None


def analyse(name, groups_by_step, steps, has_b):
    G = [g for s in steps for g in groups_by_step.get(s, [])]
    out = {"groups": len(G)}
    k = [sum(map(y_of, g)) for g in G]
    out["composition"] = {"all_wrong": sum(x == 0 for x in k), "mixed": sum(0 < x < 8 for x in k),
                          "all_correct": sum(x == 8 for x in k)}
    mixed = [g for g, x in zip(G, k) if 0 < x < 8]
    # L0 make-up of mixed groups (what the gradient is built from)
    lab = Counter()
    for g in mixed:
        l0 = L0.get(g[0]["qid"], {})
        if l0.get("answerability") != "answerable":
            lab["unanswerable"] += 1
        elif l0.get("gt_check") != "agree":
            lab["gt_" + str(l0.get("gt_check"))] += 1
        elif g[0].get("m") == 1:
            lab["eligible_m1"] += 1
        else:
            lab["answerable_agree_but_m0"] += 1
    out["mixed_group_L0"] = dict(lab)
    elig = [g for g in mixed if g[0].get("m") == 1]
    out["mixed_eligible"] = len(elig)
    # pair decomposition on EV
    cell = Counter()
    cell0 = Counter()
    by_tag = defaultdict(Counter)
    by_src = defaultdict(Counter)
    geo = Counter()
    for g in elig:
        ev0 = int(g[0].get("phi0", 0) >= 0.999)
        tag = ((L0.get(g[0]["qid"]) or {}).get("task_tags") or ["none"])[0]
        src = g[0]["qid"].split("_")[0]
        C = [r for r in g if y_of(r)]
        W = [r for r in g if not y_of(r)]
        for a in C:
            for b in W:
                ea, eb = int(a.get("phi", 0) >= 0.999), int(b.get("phi", 0) >= 0.999)
                key = ("orig_sufficient" if ev0 else
                       {(1, 0): "acq_diff", (1, 1): "post_evidence", (0, 0): "no_evidence_both",
                        (0, 1): "reverse"}[(ea, eb)])
                cell[key] += 1
                by_tag[tag][key] += 1
                by_src[src][key] += 1
                if not ev0:
                    cell0[(ea, eb)] += 1
                ga, gb = geo_full(a), geo_full(b)
                if ga is not None and gb is not None:
                    geo[{(1, 0): "acq_diff", (1, 1): "post_evidence", (0, 0): "no_evidence_both",
                         (0, 1): "reverse"}[(ga, gb)]] += 1
    tot = sum(cell.values())
    out["pairs"] = tot
    out["pair_cells"] = {k2: [v, round(v / tot, 4)] for k2, v in sorted(cell.items())} if tot else {}
    gt = sum(geo.values())
    out["pair_cells_geometry_only"] = {k2: [v, round(v / gt, 4)] for k2, v in sorted(geo.items())} if gt else {}
    out["pair_cells_by_source"] = {s: {k2: round(v / sum(c.values()), 3) for k2, v in c.items()} | {"n": sum(c.values())}
                                   for s, c in by_src.items()}
    out["pair_cells_by_first_tag"] = {t: {k2: round(v / sum(c.values()), 3) for k2, v in c.items()} | {"n": sum(c.values())}
                                      for t, c in sorted(by_tag.items(), key=lambda kv: -sum(kv[1].values()))[:8]}
    # within-group AUC of candidate signals
    sig = {"phi": lambda r: r.get("phi"), "gate": lambda r: r.get("gate"), "crops": lambda r: r["crops"],
           "gen_tokens": lambda r: r["gen_tokens"], "geo_full": geo_full}
    if has_b:
        sig |= {"b_last": lambda r: r["b"][-1], "b_max": lambda r: max(r["b"])}
    out["within_group_auc_mixed_eligible"] = {n: auc_within(elig, f) for n, f in sig.items()}
    out["within_group_auc_mixed_all"] = {n: auc_within(mixed, f) for n, f in sig.items() if n not in ("phi", "gate", "geo_full")}
    # k = 1 groups: is the lone correct one evidence-backed?
    lone = [r for g in elig if sum(map(y_of, g)) == 1 for r in g if y_of(r)]
    many = [r for g in elig if sum(map(y_of, g)) >= 4 for r in g if y_of(r)]
    out["k1_lone_correct"] = {"n": len(lone), "EV_rate": round(st.mean(r.get("phi", 0) >= .999 for r in lone), 3) if lone else None,
                              "zero_crop_rate": round(st.mean(r["crops"] == 0 for r in lone), 3) if lone else None}
    out["k_ge4_correct"] = {"n": len(many), "EV_rate": round(st.mean(r.get("phi", 0) >= .999 for r in many), 3) if many else None,
                            "zero_crop_rate": round(st.mean(r["crops"] == 0 for r in many), 3) if many else None}
    # wrong despite evidence: per-trajectory rates among eligible mixed groups (not orig-sufficient)
    nev0 = [r for g in elig if g[0].get("phi0", 0) < .999 for r in g]
    out["P(correct|EV)"] = round(st.mean(y_of(r) for r in nev0 if r.get("phi", 0) >= .999), 3) if nev0 else None
    out["P(correct|not EV)"] = round(st.mean(y_of(r) for r in nev0 if r.get("phi", 0) < .999), 3) if nev0 else None
    # all-wrong eligible groups: did anyone reach evidence?
    aw = [g for g, x in zip(G, k) if x == 0 and g[0].get("m") == 1]
    out["all_wrong_eligible"] = {"n": len(aw),
                                 "share_with_some_EV": round(st.mean(any(r.get("phi", 0) >= .999 for r in g) for g in aw), 3) if aw else None,
                                 "share_orig_sufficient": round(st.mean(g[0].get("phi0", 0) >= .999 for g in aw), 3) if aw else None}
    if has_b and aw:
        out["all_wrong_eligible"]["share_with_some_b_last_ge_.5"] = round(st.mean(any(r["b"][-1] >= .5 for r in g) for g in aw), 3)
    if has_b:
        # divergence turn: within-group mean b_k(correct) - b_k(wrong), b padded with its last value
        dk = defaultdict(list)
        for g in mixed:
            for kk in range(4):
                c = [r["b"][min(kk, len(r["b"]) - 1)] for r in g if y_of(r)]
                w = [r["b"][min(kk, len(r["b"]) - 1)] for r in g if not y_of(r)]
                dk[kk].append(st.mean(c) - st.mean(w))
        out["divergence_b_k_correct_minus_wrong"] = {kk: round(st.mean(v), 4) for kk, v in dk.items()}
        # CPC admission numbers (turn level, weight 1 per turn)
        xs, ys, a0, a1 = [], [], [], []
        for g in mixed:
            yb = st.mean(map(y_of, g))
            sd = st.stdev([y_of(r) for r in g])
            for r in g:
                b0 = r["b"][0]
                for bk in r["b"]:
                    xs.append(bk - b0)
                    ys.append(y_of(r) - yb)
                    a0.append((y_of(r) - yb) / sd)
                    a1.append(((y_of(r) - yb) - (bk - b0)) / sd)
        mx, my = st.mean(xs), st.mean(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)
        var = sum((x - mx) ** 2 for x in xs) / len(xs)
        lam = cov / var if var else None
        a_star = [a - lam * x / s if False else None for a, x, s in []]  # placeholder kept simple below
        v0 = st.pvariance(a0)
        out["cpc"] = {"lambda_star": round(lam, 4) if lam is not None else None,
                      "adv_var_reduction_lambda1": round(1 - st.pvariance(a1) / v0, 4)}
    return out


RES = {}
sb = jl(R / "sb_20261104/data/rollouts.jsonl")
gsb = groups_by_update(sb, R / "sb_20261104/data/updates.jsonl")
ga = groups_by_update(jl(R / "sg_20261104/data/sg_a_rollouts.jsonl"))
RES["DAPO-SB_start(upd1-2,A-SFT policy)"] = analyse("sb", gsb, [1, 2], True)
RES["DAPO-SB_late(upd25-32)"] = analyse("sb", gsb, range(25, 33), True)
RES["DAPO-SB_all"] = analyse("sb", gsb, range(1, 33), True)
RES["GRPO+A_start(step1-2,A-SFT policy)"] = analyse("ga", ga, [1, 2], False)
RES["GRPO+A_late(step25-32)"] = analyse("ga", ga, range(25, 33), False)

# ---------------- E1 V* (official GT boxes), 4 samples per question as the group ----------------
from vstar_recheck import cover, find_item  # noqa: E402
F = R / "baselines_v52_20261024/full"
E1 = {"A-SFT": F / "traj_B3_sft.jsonl", "GRPO": F / "traj_k8_grpo.jsonl", "DAPO": F / "traj_k8_dapo.jsonl",
      "DTPO": R / "dtpo_vlm_20261101/review_final/traj_dtpo32.jsonl",
      "A+B": R / "sg_20261104/data/traj_sg_ab.jsonl", "GRPO+A": R / "sg_20261104/data/traj_sg_a.jsonl",
      "DAPO-SB": R / "sb_20261104/data/traj_sb_dapo_sb.jsonl"}
vres = {}
for m, p in E1.items():
    byq = defaultdict(list)
    for r in jl(p):
        if r["source"] != "vstar":
            continue
        it = find_item(r["uid"])
        b = r["boxes_abs"]
        crops = (ast.literal_eval(b) if isinstance(b, str) else (b or []))[1:]
        gts = [[x, y, x + w, y + h] for x, y, w, h in it["bbox"]]
        sc = []
        for gbox in gts:
            cs = [c for c in crops if cover(c, gbox) >= .9]
            if not cs:
                sc = None
                break
            c = min(cs, key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))
            sc.append(max(gbox[2] - gbox[0], gbox[3] - gbox[1]) / max(c[2] - c[0], c[3] - c[1]))
        byq[r["uid"]].append({"y": int(r["ok"]), "cov": int(sc is not None), "scale": min(sc) if sc else 0.0,
                              "crops": len(crops)})
    cell = Counter()
    for q, g in byq.items():
        C = [t for t in g if t["y"]]
        W = [t for t in g if not t["y"]]
        for a in C:
            for b in W:
                cell[{(1, 0): "acq_diff", (1, 1): "post_evidence", (0, 0): "no_evidence_both", (0, 1): "reverse"}[(a["cov"], b["cov"])]] += 1
    tot = sum(cell.values())
    both = []
    for q, g in byq.items():
        C = [t for t in g if t["y"] and t["cov"]]
        W = [t for t in g if not t["y"] and t["cov"]]
        for a in C:
            for b in W:
                both.append(int(a["scale"] > b["scale"]))
    allt = [t for g in byq.values() for t in g]
    vres[m] = {"questions": len(byq), "mixed_questions": sum(0 < sum(t["y"] for t in g) < 4 for g in byq.values()),
               "pairs": tot, "pair_cells": {k: [v, round(v / tot, 3)] for k, v in sorted(cell.items())} if tot else {},
               "post_evidence_pairs_correct_has_larger_scale": round(st.mean(both), 3) if both else None,
               "P(correct|covered)": round(st.mean(t["y"] for t in allt if t["cov"]), 3),
               "P(correct|not covered)": round(st.mean(t["y"] for t in allt if not t["cov"]), 3),
               "covered_rate": round(st.mean(t["cov"] for t in allt), 3)}
RES["E1_Vstar"] = vres
json.dump(RES, open(H / "d0.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(RES, indent=1, ensure_ascii=False))
