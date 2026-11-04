"""Behavior audit: GRPO / DAPO / A+B / A+B+C (offline, existing logs + E1 trajectories). Writes audit.json and prints.
Sections: (1) training budget & group signal, (2) A+B credit in no-task-signal groups, (3) E1 crop behavior,
(4) V* evidence path with GT boxes, (5) paired gains/losses by source, (6) case lists.
python audit_behavior.py
"""
import ast
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
RUNS = HERE.parents[1]
sys.path.insert(0, str(RUNS.parent / "notes" / "rereview_20261104"))
from vstar_recheck import cover, find_item  # noqa: E402

OUT = {}
TRAIN = {"GRPO": RUNS / "k8_20261028/k8_grpo/rollouts.jsonl", "DAPO": RUNS / "k8_20261028/k8_dapo/rollouts.jsonl",
         "A+B": RUNS / "sg_20261104/data/sg_ab_rollouts.jsonl", "A+B+C": RUNS / "sg_20261104/data/sg_abc_rollouts.jsonl"}
UPD = {"DAPO": RUNS / "k8_20261028/k8_dapo/updates.jsonl"}
E1 = {"GRPO": RUNS / "baselines_v52_20261024/full/traj_k8_grpo.jsonl",
      "DAPO": RUNS / "baselines_v52_20261024/full/traj_k8_dapo.jsonl",
      "A+B": RUNS / "sg_20261104/data/traj_sg_ab.jsonl", "A+B+C": RUNS / "sg_20261104/data/traj_sg_abc.jsonl"}
VAL = {"GRPO": {0: .533, 25: .576, 32: .602}, "DAPO": {0: .533, 25: .595, 32: .637}}
for t in ("ab", "abc"):
    cur = json.load(open(RUNS / "sg_20261104/data/sg_curves.json"))[f"sg_{t}"]
    VAL["A+B" if t == "ab" else "A+B+C"] = {int(s): m["val/accuracy_reward"] for s, m in cur.items()
                                            if "val/accuracy_reward" in m}


# ---------- (1) training budget & group composition ----------
def groups_of(path):
    by = defaultdict(list)
    order = []
    for l in open(path):
        r = json.loads(l)
        k = r["uid"]
        if k not in by:
            order.append(k)
        by[k].append(r)
    return [by[k] for k in order]


budget = {}
for c, p in TRAIN.items():
    G = groups_of(p)
    rows = [r for g in G for r in g]
    comp = Counter()
    for g in G:
        if g[0].get("cont"):
            comp["continuation"] += 1
            continue
        y = sum(r["correct"] for r in g)
        comp["all_correct" if y == len(g) else "all_wrong" if y == 0 else "mixed"] += 1
    qexp = Counter(g[0]["qid"] for g in G if not g[0].get("cont"))
    cum_tok, cum_val = 0, {}
    per_call = defaultdict(int)
    for r in rows:
        per_call[r["step"]] += r["proc_tokens"]
    budget[c] = {"generated_groups": len(G), "rollouts": len(rows), "group_composition": dict(comp),
                 "unique_qids_generated": len(qexp), "qids_generated_twice_or_more": sum(v >= 2 for v in qexp.values()),
                 "processed_tokens": sum(r["proc_tokens"] for r in rows),
                 "generated_tokens": sum(r["gen_tokens"] for r in rows)}
for c, p in UPD.items():
    U = [json.loads(l) for l in open(p)]
    uq = Counter(q for u in U for q in u["qids"])
    budget[c]["update_groups"] = sum(len(u["qids"]) for u in U)
    budget[c]["unique_qids_updated"] = len(uq)
    budget[c]["qids_updated_twice_or_more"] = sum(v >= 2 for v in uq.values())
for c in budget:
    budget[c]["val_dev422"] = VAL.get(c)
OUT["budget"] = budget

# DAPO cost-aligned point: cumulative processed tokens when DAPO reached GRPO's total
grpo_tok = budget["GRPO"]["processed_tokens"]
cum, step_at = 0, None
dapo_rows = [json.loads(l) for l in open(TRAIN["DAPO"])]
calls = defaultdict(int)
for r in dapo_rows:
    calls[r["step"]] += r["proc_tokens"]
upd_calls = [len(json.loads(l)["qids"]) for l in open(UPD["DAPO"])]
OUT["dapo_tokens_ratio_vs_grpo"] = round(budget["DAPO"]["processed_tokens"] / grpo_tok, 3)

# ---------- (2) A+B credit: task-scaled advantage recomputed from the ledger ----------
def ts_adv(g, lam=0.1):
    y = [r["correct"] if r["terminal"] == "answer" else 0.0 for r in g]
    s = [(lam * r.get("m", 0) * r.get("s", 0.0)) if r["terminal"] == "answer" else 0.0 for r in g]
    n = len(g)
    my, ms = sum(y) / n, sum(s) / n
    sd = (sum((v - my) ** 2 for v in y) / (n - 1)) ** 0.5
    sig = max(sd, 1 / n ** 0.5)
    return [((yi - my) + (si - ms)) / sig for yi, si in zip(y, s)], y, s


credit = {}
for c in ("A+B", "A+B+C"):
    G = [g for g in groups_of(TRAIN[c]) if not g[0].get("cont")]
    cnt = Counter()
    pos_wrong = []
    for g in G:
        A, y, s = ts_adv(g)
        ny = sum(y)
        kind = "all_wrong" if ny == 0 else "all_correct" if ny == len(g) else "mixed"
        sv = max(s) - min(s) > 1e-9
        cnt[f"{kind}{'_Svar' if sv else ''}"] += 1
        for r, a, yy in zip(g, A, y):
            if yy == 0 and a > 1e-9:
                pos_wrong.append({"qid": r["qid"], "step": r["step"], "adv": round(a, 3), "s": r.get("s"),
                                  "crops": r["crops"], "group": kind})
            if kind == "mixed" and ((yy == 1 and a <= 0) or (yy == 0 and a >= 0)):
                cnt["mixed_sign_flip"] += 1
    wrong_rows = sum(1 for g in G for r in g if not (r["terminal"] == "answer" and r["correct"]))
    credit[c] = {"groups": dict(cnt), "wrong_rollouts_with_positive_adv": len(pos_wrong),
                 "wrong_rollouts_total": wrong_rows,
                 "pos_wrong_mean_adv": round(st.mean(p["adv"] for p in pos_wrong), 3) if pos_wrong else 0,
                 "pos_wrong_mean_crops": round(st.mean(p["crops"] for p in pos_wrong), 2) if pos_wrong else 0,
                 "pos_wrong_by_group": dict(Counter(p["group"] for p in pos_wrong))}
OUT["ab_credit"] = credit


# ---------- (3) E1 crop behavior ----------
def boxes(r):
    b = r["boxes_abs"]
    b = ast.literal_eval(b) if isinstance(b, str) else (b or [])
    return [tuple(map(float, x)) for x in b[1:]]


def turns(r):
    t = r["turns"]
    return ast.literal_eval(t) if isinstance(t, str) else (t or [])


def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


T = {c: {(r["uid"], r["sample_idx"]): r for r in map(json.loads, open(p))} for c, p in E1.items()}
keys = sorted(set.intersection(*(set(t) for t in T.values())))
beh = {}
for c, t in T.items():
    R = [t[k] for k in keys]
    nc = Counter(min(r["n_crops"] or 0, 5) for r in R)
    area_first, area_later, nested, revisit, bigcrop, words, revise, hold, abstain = [], [], 0, 0, 0, Counter(), 0, 0, 0
    total_crops = 0
    acc_by_nc = defaultdict(list)
    for r in R:
        W, H = r["image_wh"]
        bs = boxes(r)
        total_crops += len(bs)
        acc_by_nc[min(len(bs), 5)].append(r["ok"])
        for j, b in enumerate(bs):
            a = (b[2] - b[0]) * (b[3] - b[1]) / (W * H)
            (area_first if j == 0 else area_later).append(a)
            bigcrop += a >= 0.6
            revisit += any(iou(b, e) >= 0.5 for e in bs[:j])
        for tt in turns(r):
            p = tt.get("parsed") or {}
            if p.get("behavior"):
                words[p["behavior"]] += 1
            if p.get("kind") == "grounding" and str(p.get("source", "")).startswith("observation"):
                nested += 1
            txt = tt.get("text") or ""
            revise += "REVISE" in txt
            hold += "HOLD" in txt
        abstain += (r.get("pred") or "") == "UNCLEAR" or "UNCLEAR" in (r.get("final_text") or "")[-60:]
    beh[c] = {"acc": round(st.mean(r["ok"] for r in R), 4),
              "crops_dist(0..5+)": {k: nc[k] for k in range(6)},
              "used_all_8": sum((r["n_crops"] or 0) >= 8 for r in R),
              "first_crop_area_median": round(st.median(area_first), 3) if area_first else None,
              "later_crop_area_median": round(st.median(area_later), 3) if area_later else None,
              "share_crops_from_observation(nested)": round(nested / max(1, total_crops), 3),
              "share_crops_revisit_iou50": round(revisit / max(1, total_crops), 3),
              "share_crops_area_ge_0.6": round(bigcrop / max(1, total_crops), 3),
              "words": dict(words.most_common()), "turns_with_REVISE": revise, "turns_with_HOLD": hold,
              "abstain_UNCLEAR": abstain, "no_pred": sum(r.get("pred") is None for r in R),
              "acc_by_crops(0..5+)": {k: [round(st.mean(v), 3), len(v)] for k, v in sorted(acc_by_nc.items())}}
OUT["e1_behavior"] = beh


# ---------- (4) V* evidence path ----------
def vstar_path(r):
    it = find_item(r["uid"])
    gts = [[x, y, x + w, y + h] for x, y, w, h in it["bbox"]]
    bs = boxes(r)
    first, scale = None, None
    for j, b in enumerate(bs):
        if all(cover(b, g) >= 0.9 for g in gts):
            first = j
            scale = min(max(g[2] - g[0], g[3] - g[1]) / max(b[2] - b[0], b[3] - b[1]) for g in gts)
            break
    cat = "not_covered" if first is None else ("covered_small" if scale < .05 else "covered_ok")
    best = None
    for b in bs:
        if all(cover(b, g) >= 0.9 for g in gts):
            sc = min(max(g[2] - g[0], g[3] - g[1]) / max(b[2] - b[0], b[3] - b[1]) for g in gts)
            best = sc if best is None else max(best, sc)
    bcat = "not_covered" if best is None else ("covered_small" if best < .05 else "covered_ok")
    return {"first_cover_idx": first, "after_first_cover": None if first is None else len(bs) - first - 1,
            "cat_best": bcat, "n": len(bs)}


vs = {}
paths = {}
for c, t in T.items():
    P = {k: vstar_path(t[k]) for k in keys if t[k]["source"] == "vstar"}
    paths[c] = P
    cells = Counter((P[k]["cat_best"], t[k]["ok"]) for k in P)
    after = [P[k]["after_first_cover"] for k in P if P[k]["after_first_cover"] is not None]
    vs[c] = {"cells(best covering crop, ok)": {f"{a}|ok={o}": v for (a, o), v in sorted(cells.items())},
             "acc_by_cat": {a: round(st.mean(t[k]["ok"] for k in P if P[k]["cat_best"] == a), 3)
                            for a in ("not_covered", "covered_small", "covered_ok") if any(P[k]["cat_best"] == a for k in P)},
             "crops_after_first_cover_mean": round(st.mean(after), 2) if after else None,
             "acc_if_more_crops_after_cover": round(st.mean(t[k]["ok"] for k in P if (P[k]["after_first_cover"] or 0) >= 1 and P[k]["first_cover_idx"] is not None), 3),
             "acc_if_stop_at_cover": round(st.mean(t[k]["ok"] for k in P if P[k]["after_first_cover"] == 0), 3),
             "n_stop_at_cover": sum(1 for k in P if P[k]["after_first_cover"] == 0),
             "n_more_after_cover": sum(1 for k in P if (P[k]["after_first_cover"] or 0) >= 1)}
for a, b in (("GRPO", "A+B"), ("DAPO", "A+B"), ("A+B", "A+B+C")):
    tr = Counter()
    for k in paths[a]:
        tr[(paths[a][k]["cat_best"] + ("+" if T[a][k]["ok"] else "-"),
            paths[b][k]["cat_best"] + ("+" if T[b][k]["ok"] else "-"))] += 1
    vs[f"transition {a}->{b}"] = {f"{x} -> {y}": v for (x, y), v in sorted(tr.items(), key=lambda z: -z[1]) if x != y}
OUT["vstar"] = vs


# ---------- (5) paired by source ----------
def paired(a, b):
    d = defaultdict(lambda: [0, 0])
    for k in keys:
        x, y = T[a][k]["ok"], T[b][k]["ok"]
        if x != y:
            d[T[a][k]["source"]][0 if x else 1] += 1
    return {s: {"only_" + a: v[0], "only_" + b: v[1], "net_" + b: v[1] - v[0]} for s, v in sorted(d.items())}


OUT["paired"] = {f"{a} vs {b}": paired(a, b) for a, b in
                 (("GRPO", "A+B"), ("DAPO", "A+B"), ("DAPO", "A+B+C"), ("GRPO", "DAPO"), ("A+B", "A+B+C"))}


# ---------- (6) case lists (question level, 4 samples) ----------
def qscore(c):
    s = defaultdict(int)
    for k in keys:
        s[k[0]] += T[c][k]["ok"]
    return s


Q = {c: qscore(c) for c in T}
src = {k[0]: T["GRPO"][k]["source"] for k in keys}
crops_q = {c: defaultdict(list) for c in T}
for c in T:
    for k in keys:
        crops_q[c][k[0]].append(T[c][k]["n_crops"] or 0)
cases = {}
for name, a, b, s in (("wb_AB_gain_vs_GRPO", "GRPO", "A+B", "worldbench"), ("mme_AB_loss_vs_DAPO", "DAPO", "A+B", "mme_realworld"),
                      ("vstar_AB_loss_vs_DAPO", "DAPO", "A+B", "vstar")):
    rows = [(Q[b][u] - Q[a][u], u) for u in Q[a] if src[u] == s]
    rows.sort(reverse=name.startswith("wb"))
    cases[name] = [{"uid": u, **{c: Q[c][u] for c in T},
                    **{f"crops_{c}": round(st.mean(crops_q[c][u]), 2) for c in ("DAPO", "GRPO", "A+B")}}
                   for d, u in rows[:12]]
OUT["cases"] = cases
json.dump(OUT, open(HERE / "audit.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(OUT, indent=1, ensure_ascii=False))
