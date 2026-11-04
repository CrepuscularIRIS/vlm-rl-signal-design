"""Behavior checklist measurement (user list A1-A4, B1-B4, C1-C3), definitions frozen before looking at DAPO data.
usage: python measure_behaviors.py <out.json> <traj.jsonl> [<traj.jsonl> ...]
Per behavior (+ agreement of compliance with correctness; all-wrong question groups whose samples disagree on compliance = signal Y cannot give): applicable questions / trajectories, compliance rate, share of questions whose 8 samples disagree on
compliance (within-group variance), and same-question accuracy gap acc(compliant) - acc(not) averaged over those
questions (difficulty controlled by comparing samples of the same question).

Definitions (final turn = text after the last </grounding>; views = original image + executed crops;
"legible" = covers the box >= 0.9 at usable input scale, reward_sg a_b >= 0.5):
A1 cited views (final-turn [original_image]/[observation_k]; if none cited, the last view) make every box of some L2
   evidence_set legible.                                        applicable: sg m=1
A2 answers with 0 crops.                                        applicable: sg m=1, target_too_small=false, key evidence
                                                                legible in the original view
A3 row header, column header and value box of the first table binding are all legible in some presented view.
                                                                applicable: L2 table binding with header roles
A4 final turn does not state a negative-box value while omitting every target value.
                                                                applicable: L2 values on negative boxes (n*)
B1 evfact E > 0 (some table/entity fact stated with correct binding and legible support).   applicable: evfact index
B2 final turn states the annotated count total.                 applicable: L2 count binding
B3 final turn has the canonical direction word(s) and none of the opposite ones.   applicable: relation_canonical != NA
B4 trajectory uses REVISE (usage only).                          applicable: all
C1 when the final turn states exactly one option's text (non-"not feature"), the chosen letter is that option.
                                                                applicable: trajectories with exactly one such option
C2 final turn cites a view and states a value (number or quoted string).   applicable: all
C3 chooses the "image does not feature" option when it is the gt.   applicable: gt option is "does not feature"
C3b does not choose "does not feature" when the gt is another option.  applicable: questions offering that option
"""
import ast
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

OBS = Path(__import__("os").environ.get("OBS_DIR", "."))  # public copy
sys.path.insert(0, str(OBS / "src/evfact"))
sys.path.insert(0, str(OBS / "src/grpo_sg"))
sys.path.insert(0, str(OBS / "src/rread"))
import evfact as EF  # noqa: E402
sys.path.insert(0, str(OBS / "src/bind"))
from bind_scorer import strip_meta  # noqa: E402  (Design_PH §1: protocol ids are not readings)
import reward_sg as RS  # noqa: E402
from rread_scorer import match_num, match_txt, norm_num  # noqa: E402

A = OBS / "notes/rl_pool_annotation_20261101/out/final/rl4k"
L1 = {r["uid"]: r for r in map(json.loads, filter(str.strip, open(A / "L1/batch.jsonl")))}
L2 = {r["uid"]: r for r in map(json.loads, filter(str.strip, open(A / "L2/batch.jsonl")))}
SG = json.load(open(OBS / "src/grpo_sg/sg_index.json"))
RS._INDEX = SG
EFI = json.load(open(OBS / "src/evfact/evfact_index.json"))
REL = {r["uid"]: r for r in map(json.loads, open(OBS / "notes/rereview_20261104/relation_canonical/relation_canonical.jsonl"))
       if r["dir_canonical"] != "NA"}
TR = pd.read_parquet(Path(__file__).parent / "rl4k_train.parquet")
Q = {r["uid"]: str(r["env_head"]).split("\nOriginal image")[0] for _, r in TR.iterrows()}
NOTFEAT = re.compile(r"(does not|doesn'?t) feature", re.I)
DIRW = {"L": r"\bleft\b", "R": r"\bright\b", "U": r"\b(above|top|upper|over)\b", "D": r"\b(below|bottom|lower|under|beneath)\b"}
OPP = {"L": "R", "R": "L", "U": "D", "D": "U"}


def options(uid):
    return {m.group(1): m.group(2).strip() for m in re.finditer(r"\(([A-J])\)\s*([^\n]*)", Q.get(uid, ""))}


def turns(r):
    t = r["turns"]
    return ast.literal_eval(t) if isinstance(t, str) else (t or [])


def full(r):
    return "".join(x.get("text") or "" for x in turns(r)) or (r.get("final_text") or "")


def crops(r):
    b = r["boxes_abs"]
    b = ast.literal_eval(b) if isinstance(b, str) else (b or [])
    return [tuple(map(float, x)) for x in b[1:]]


def legible(bbox_rel, view, scale, W, H, kind="text", lines=1):
    b = (bbox_rel[0] * W, bbox_rel[1] * H, bbox_rel[2] * W, bbox_rel[3] * H)
    d = (b[3] - b[1]) / max(1, lines) if kind == "text" else min(b[2] - b[0], b[3] - b[1])
    return RS._cover(b, view) >= RS.COVER and RS._h(kind, d * scale) >= 0.5


def node_legible_in(node, views, W, H):
    if node["kind"] == "context":
        return any(RS._cover((node["bbox"][0] * W, node["bbox"][1] * H, node["bbox"][2] * W, node["bbox"][3] * H), v) >= RS.COVER
                   for v, _ in views)
    return any(legible(node["bbox"], v, s, W, H, node["kind"], node.get("lines", 1)) for v, s in views)


def a1(r, ft):
    e = SG.get(r["uid"])
    if not e or not e.get("m"):
        return None
    W, H = r["image_wh"]
    views = RS._views(W, H, crops(r))
    ft = EF.commit_text(full(r))  # view ids are needed here: use the unstripped final turn
    cited = {0 if a == "original_image" else int(k) for a, k in re.findall(r"\[(original_image|observation_(\d+))\]", ft) if a}
    cited = {int(k) for k in re.findall(r"\[observation_(\d+)\]", ft)} | ({0} if "[original_image]" in ft else set())
    cited = {c for c in cited if c < len(views)} or {len(views) - 1}
    vs = [views[c] for c in sorted(cited)]
    return any(all(node_legible_in(e["nodes"][b], vs, W, H) for b in S) for S in e["sets"])


def a2(r, ft):
    e = SG.get(r["uid"])
    if not e or not e.get("m") or L1.get(r["uid"], {}).get("target_too_small"):
        return None
    W, H = r["image_wh"]
    v0 = RS._views(W, H, [])
    keys = [n for n in e["nodes"].values() if n.get("key")]
    if not keys or not any(node_legible_in(n, v0, W, H) for n in keys):
        return None
    return len(crops(r)) == 0


def a3(r, ft):
    l2, l1 = L2.get(r["uid"]), L1.get(r["uid"])
    if not l2 or not l1:
        return None
    boxes = {b["id"]: b for b in l1["boxes"] or []}
    for b in l2.get("bindings") or []:
        if b["type"] != "table":
            continue
        sup = [s for s in b.get("support", []) if s in boxes]
        roles = {boxes[s]["role"] for s in sup}
        if not ({"row_header", "col_header"} <= roles):
            continue
        W, H = r["image_wh"]
        views = RS._views(W, H, crops(r))
        return all(any(legible(boxes[s]["bbox"], v, sc, W, H) for v, sc in views) for s in sup)
    return None


def _says(text, v):
    v = str(v or "").strip()
    if len(v) < 1 or len(v) > 40:
        return False
    return match_num(text, v) if norm_num(v) is not None else (len(v) >= 3 and match_txt(text, v))


def a4(r, ft):
    l2 = L2.get(r["uid"])
    if not l2:
        return None
    vals = l2.get("values") or {}
    vt = {k: (v.get("text") if isinstance(v, dict) else v) for k, v in vals.items()}
    negs = [v for k, v in vt.items() if k.startswith("n") and v]
    tgt = [b.get("value") for b in l2.get("bindings") or [] if isinstance(b.get("value"), str)]
    negs = [n for n in negs if all(str(n).strip() != str(t).strip() for t in tgt)]
    if not negs or not tgt:
        return None
    return not (any(_says(ft, n) for n in negs) and not any(_says(ft, t) for t in tgt))


def b1(r, ft):
    e = EFI.get(r["uid"])
    if not e:
        return None
    return EF.score(e, tuple(r["image_wh"]), crops(r), full(r))[0] > 0


def b2(r, ft):
    l2 = L2.get(r["uid"])
    c = [b for b in (l2 or {}).get("bindings") or [] if b["type"] == "count" and isinstance(b.get("total"), (int, float))]
    if not c:
        return None
    return match_num(ft, str(int(c[0]["total"]))) if int(c[0]["total"]) >= 10 else bool(
        re.search(r"(?<![\d.])" + str(int(c[0]["total"])) + r"(?![\d.])", ft))


def b3(r, ft):
    rel = REL.get(r["uid"])
    if not rel:
        return None
    t = ft.lower()
    return all(re.search(DIRW[d], t) for d in rel["dir_canonical"]) and not any(re.search(DIRW[OPP[d]], t) for d in rel["dir_canonical"])


def b4(r, ft):
    return "REVISE" in full(r)


def c1(r, ft):
    ops = {k: v for k, v in options(r["uid"]).items() if v and not NOTFEAT.search(v) and len(v) >= 2}
    hit = [k for k, v in ops.items() if _says(ft.split("<answer>")[0], v)]
    if len(hit) != 1:
        return None
    return r.get("pred") == hit[0]


def c2(r, ft):
    raw = EF.commit_text(full(r))
    return bool(re.search(r"\[(original_image|observation_\d+)\]", raw)) and bool(re.search(r"\d|\"[^\"]+\"|“[^”]+”", ft))


def c3(r, ft):
    ops = options(r["uid"])
    nf = [k for k, v in ops.items() if NOTFEAT.search(v)]
    if not nf or r["gt"] not in nf:
        return None
    return r.get("pred") == r["gt"]


def c3b(r, ft):
    ops = options(r["uid"])
    nf = [k for k, v in ops.items() if NOTFEAT.search(v)]
    if not nf or r["gt"] in nf:
        return None
    return r.get("pred") not in nf


BEH = [("A1 cited views make an evidence set legible", a1), ("A2 0 crops when >=1 key node legible in original", a2),
       ("A3 first table binding boxes legible in any view", a3), ("A4 not (distractor value stated & no target value)", a4),
       ("B1 evfact v0 E>0 (table+entity)", b1), ("B2 count total stated", b2),
       ("B3 direction word consistent with boxes", b3), ("B4 uses REVISE (usage)", b4),
       ("C1 chosen letter = option whose text is stated", c1), ("C2 final turn cites view + states value", c2),
       ("C3 says 'not featured' when target absent", c3), ("C3b no false 'not featured'", c3b)]


def main(out, paths):
    rows = [json.loads(l) for p in paths for l in open(p)]
    res = {}
    q_allwrong = defaultdict(lambda: True)
    for r in rows:
        q_allwrong[r["uid"]] &= not r["ok"]
    for name, fn in BEH:
        per_q = defaultdict(list)
        for r in rows:
            ft = strip_meta(EF.commit_text(full(r)))
            v = fn(r, ft)
            if v is not None:
                per_q[r["uid"]].append((bool(v), r["ok"]))
        t = [x for g in per_q.values() for x in g]
        if not t:
            res[name] = {"applicable_q": 0}
            continue
        var = [g for g in per_q.values() if 0 < sum(c for c, _ in g) < len(g)]
        gaps = [st.mean(o for c, o in g if c) - st.mean(o for c, o in g if not c) for g in var]
        aw = [g for u, g in per_q.items() if q_allwrong[u]]  # all 8 samples of the question wrong
        aw_var = [g for g in aw if 0 < sum(c for c, _ in g) < len(g)]
        res[name] = {"applicable_q": len(per_q), "applicable_traj": len(t),
                     "compliance": round(st.mean(c for c, _ in t), 3),
                     "q_with_variance": len(var), "q_var_share": round(len(var) / len(per_q), 3),
                     "same_q_acc_gap": round(st.mean(gaps), 3) if gaps else None,
                     "acc_compliant": round(st.mean(o for c, o in t if c), 3) if any(c for c, _ in t) else None,
                     "acc_not": round(st.mean(o for c, o in t if not c), 3) if any(not c for c, _ in t) else None,
                     "agree_with_ok": round(st.mean(c == bool(o) for c, o in t), 3),
                     "allwrong_q": len(aw), "allwrong_q_with_variance": len(aw_var)}
    json.dump(res, open(out, "w"), indent=1)
    w = max(len(n) for n, _ in BEH)
    print(f"{'behavior':{w}}  q    traj   comply  q_var  gap(same q)  acc|c  acc|not  agree_ok  allwrong(var/n)")
    for n, _ in BEH:
        x = res[n]
        if not x.get("applicable_q"):
            print(f"{n:{w}}  0")
            continue
        print(f"{n:{w}} {x['applicable_q']:4d} {x['applicable_traj']:6d}  {x['compliance']:.3f}  {x['q_var_share']:.3f}"
              f"  {x['same_q_acc_gap'] if x['same_q_acc_gap'] is not None else '-':>10}  {x['acc_compliant']}  {x['acc_not']}"
              f"  {x['agree_with_ok']}  {x['allwrong_q_with_variance']}/{x['allwrong_q']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
