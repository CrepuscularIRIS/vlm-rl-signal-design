"""Phi_bind v2 (Design_PH §3): final adopted (row, full column path, value) table bindings with visual support.

v2 changes after the 60-case manual audit (audit60.txt): number-to-label assignment (each number goes to its nearest
row label and nearest column label; a number counts for the slot only if both are the target), the LAST such number in
the LAST binding clause is the adopted value; whitespace-insensitive labels, unicode minus/dash normalisation,
compound row labels split on ' / ' and ' - ', column labels inside a row-label span ignored, leading-dot decimals.

Slot u_k = 1 iff, in the FINAL answer turn with protocol metadata stripped (Step N | WORD, view ids, action tags):
  the last clause that mentions the row and the column path adopts the annotated value, and
  the value box, its row header and column header box(es) are each legible (cover >= 0.9, usable input scale) in some
  presented view (original image included).
Column path mention: some level whose label is unambiguous on its own, otherwise every level.
Phi_bind = mean_k u_k over slots fixed before training (numeric table bindings; entity slots off).
Known misses (kept conservative): row/column named only in an earlier sentence; paraphrased labels ("t=0").
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rread"))
sys.path.insert(0, str(HERE.parent / "grpo_sg"))
from rread_scorer import norm_num  # noqa: E402
import reward_sg as RS  # noqa: E402

SUPPORT_A = 0.5
META = [r"Step\s*\d+\s*\|\s*\w+", r"\[(?:original_image|observation_\d+)\]", r"<answer>.*?</answer>",
        r"<grounding>.*?</grounding>", r"</?think>", r"observation_\d+", r"original_image"]
NUM = re.compile(r"\(?-?\$?(?:\d(?:[\d,]*\d)?(?:\.\d+)?|\.\d+)%?\)?")
DASH = str.maketrans({"−": "-", "–": "-", "—": "-", "‒": "-", " ": " "})


def norm_text(t):
    return t.translate(DASH)


def strip_meta(text):
    for p in META:
        text = re.sub(p, " ", text, flags=re.S)
    return norm_text(text)


def commit_text(full):
    i = full.rfind("</grounding>")
    return full[i + len("</grounding>"):] if i >= 0 else full


def clauses(text):
    out, s = [], 0
    for m in re.finditer(r"[.。！？!?](?=\s|$)|[;；]|\n", text):
        out.append(text[s:m.end()])
        s = m.end()
    out.append(text[s:])
    return [c for c in out if c.strip()]


def _label(s):
    s = norm_text(str(s or "")).split("\n")[0]
    s = re.sub(r"\s*\(.*?\)\s*", " ", s).strip()
    return s if len(s) >= 2 else None


def _alts(label, seps=(r"\s*/\s*",)):
    parts = [label]
    for sep in seps:
        parts = [q.strip() for p in parts for q in re.split(sep, p)]
    out = [p for p in parts if len(p) >= 2]
    return sorted(set(out + ([label] if label not in out else [])), key=len, reverse=True)


def _spans(clause, label):
    pat = r"\s*".join(re.escape(tok) for tok in label.lower().split())
    return [(m.start(), m.end()) for m in re.finditer(r"(?<![A-Za-z0-9])" + pat + r"(?![A-Za-z0-9])", clause.lower())]


# ---------------- index ----------------
def build_index(l1_path, l2_path, sg_index_path):
    L1 = {r["uid"]: r for r in map(json.loads, filter(str.strip, open(l1_path)))}
    L2 = {r["uid"]: r for r in map(json.loads, filter(str.strip, open(l2_path)))}
    sg = json.load(open(sg_index_path))
    out = {}
    row_seps = (r"\s*/\s*", r"\s+-\s+")
    for qid, r in L2.items():
        if not sg.get(qid, {}).get("m") or qid not in L1:
            continue
        boxes = {b["id"]: b for b in L1[qid]["boxes"] or []}
        vt = {k: (v.get("text") if isinstance(v, dict) else v) for k, v in (r.get("values") or {}).items()}
        col_lbl = {k: _label(vt.get(k)) for k, b in boxes.items() if b["role"] == "col_header"}
        row_lbl = {k: _label(vt.get(k)) for k, b in boxes.items() if b["role"] == "row_header"}
        slots = []
        for b in r.get("bindings") or []:
            v = b.get("value")
            if b["type"] != "table" or not isinstance(v, str) or norm_num(norm_text(v)) is None:
                continue
            sup = [s for s in b.get("support", []) if s in boxes]
            roles = {s: boxes[s]["role"] for s in sup}
            vbox = next((s for s in sup if str(vt.get(s) or "").strip() == v.strip()), None) or \
                next((s for s in sup if roles[s] in ("cell", "answer_text", "value_label")), None)
            hdr = [s for s in sup if roles[s] in ("row_header", "col_header")]
            row = _label(b.get("row_key"))
            path = [p for p in (_label(x) for x in (b.get("col_path") or [])) if p]
            if not (vbox and row and path and any(roles[s] == "col_header" for s in hdr)):
                continue
            path_alts = [_alts(p) for p in path]
            mine_c = {a.lower() for alts in path_alts for a in alts}
            others_c = sorted({a for k, l in col_lbl.items() if l and k not in sup for a in _alts(l)
                               if a.lower() not in mine_c})
            row_alts = _alts(row, row_seps)
            mine_r = {a.lower() for a in row_alts}
            others_r = sorted({a for k, l in row_lbl.items() if l and k not in sup for a in _alts(l, row_seps)
                               if a.lower() not in mine_r})
            alone = [i for i, alts in enumerate(path_alts) if not any(a.lower() in {o.lower() for o in others_c} for a in alts)]
            slots.append({"value": norm_text(v), "row": row_alts, "path": path_alts, "alone": alone,
                          "other_rows": others_r, "other_cols": others_c,
                          "boxes": [boxes[vbox]["bbox"]] + [boxes[s]["bbox"] for s in hdr]})
        if slots:
            out[qid] = {"slots": slots}
    return out


# ---------------- scoring ----------------
def _occ(clause, labels, exclude=()):
    out = []
    for lab in labels:
        for a, b in _spans(clause, lab):
            if not any(x <= a < y or x < b <= y for x, y in exclude):
                out.append((a, b, lab))
    return out


def _binding(slot, clause):
    """(binds?, adopted value token or None)."""
    row_t = _occ(clause, slot["row"])
    if not row_t:
        return False, None
    row_spans = [(a, b) for a, b, _ in row_t] + [(a, b) for a, b, _ in _occ(clause, slot["other_rows"])]
    col_hits = {i: _occ(clause, alts, row_spans) for i, alts in enumerate(slot["path"])}
    if not (any(col_hits[i] for i in slot["alone"]) or all(col_hits.values())):
        return False, None
    row_o = _occ(clause, slot["other_rows"])
    col_t = [h for hs in col_hits.values() for h in hs]
    col_o = _occ(clause, slot["other_cols"], row_spans)
    masked = clause
    for a, b, _ in row_t + row_o + col_t + col_o:
        masked = masked[:a] + " " * (b - a) + masked[b:]
    cands = []
    for m in NUM.finditer(masked):
        tok = m.group()
        if norm_num(tok.strip("()").replace("$", "")) is None and norm_num(tok.replace("$", "")) is None:
            continue
        p = m.start()

        def nearest(ts, os_):
            best = min(((abs(a - p), True) for a, _, _ in ts), default=(1e9, True))
            other = min(((abs(a - p), False) for a, _, _ in os_), default=(1e9, False))
            return best[0] <= other[0] and best[0] < 1e9

        if nearest(row_t, row_o) and nearest(col_t, col_o):
            cands.append(tok)
    return True, (cands[-1] if cands else None)


def _same(tok, v):
    a = norm_num(tok.replace("$", ""))
    b = norm_num(v.replace("$", ""))
    if a is None or b is None:
        return False
    return abs(abs(a) - abs(b)) < 1e-9


def stated(slot, text):
    final = None
    for c in clauses(text):
        ok, tok = _binding(slot, c)
        if ok:
            final = tok
    return final is not None and _same(final, slot["value"])


def supported(slot, wh, executed_boxes):
    W, H = wh
    views = RS._views(W, H, executed_boxes)
    for b in slot["boxes"]:
        px = (b[0] * W, b[1] * H, b[2] * W, b[3] * H)
        if not any(RS._cover(px, v) >= RS.COVER and RS._h("text", (px[3] - px[1]) * s) >= SUPPORT_A for v, s in views):
            return False
    return True


def score(entry, wh, executed_boxes, full_response):
    text = strip_meta(commit_text(full_response))
    us = [1.0 if stated(s, text) and supported(s, wh, executed_boxes) else 0.0 for s in entry["slots"]]
    return (sum(us) / len(us) if us else 0.0), us
