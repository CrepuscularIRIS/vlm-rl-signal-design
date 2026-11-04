"""Verifiable evidence-fact scorer (DAPO auxiliary line, Design_New round 3 §4). Draft v0 — not frozen.

Slot u_k = 1  iff  in the FINAL answer turn (text after the last </grounding>):
  table : some clause states the L2 value, the nearest column label to that value is the target column, and the
          target row label is in the clause with no other row label closer to the value;
  entity: some clause states the (short-text) L2 value of the candidate box;
  and (both) some presented view (original image included) covers the value box >= 0.9 at usable input scale
  (reward_sg.view_scale / _h, a_b >= 0.5).
E = mean_k u_k over a fixed slot list K_q (compiled before training). Each slot counts once. Enumerations and negations
of the value do not count (rread_scorer guards). Returns the char span of the supporting clause for token credit.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rread"))
sys.path.insert(0, str(HERE.parent / "grpo_sg"))
from rread_scorer import match_num, match_txt, norm_num, _negated  # noqa: E402
import reward_sg as RS  # noqa: E402

SUPPORT_A = 0.5
TEXT_MAX = 40


# ---------------- index ----------------
def _clean_label(s):
    s = str(s or "").split("\n")[0]
    s = re.sub(r"\s*\(.*?\)\s*", " ", s).strip()
    return s if len(s) >= 2 else None


def _kind(v):
    return "number" if norm_num(str(v)) is not None else "text"


def build_index(l1_path, l2_path, sg_index_path):
    L1 = {r["uid"]: r for r in map(json.loads, filter(str.strip, open(l1_path)))}
    L2 = {r["uid"]: r for r in map(json.loads, filter(str.strip, open(l2_path)))}
    sg = json.load(open(sg_index_path))
    out = {}
    for qid, r in L2.items():
        if not sg.get(qid, {}).get("m") or qid not in L1:
            continue
        boxes = {b["id"]: b for b in L1[qid]["boxes"] or []}
        vals = r.get("values") or {}
        vtext = {k: (v.get("text") if isinstance(v, dict) else v) for k, v in vals.items()}
        row_hdrs = {k for k, b in boxes.items() if b["role"] == "row_header"}
        col_hdrs = {k for k, b in boxes.items() if b["role"] == "col_header"}
        slots = []
        for b in r.get("bindings") or []:
            v = b.get("value")
            if not isinstance(v, str) or not (1 <= len(v) <= TEXT_MAX) or re.search(r"[一-鿿]{6,}", v):
                continue
            sup = [s for s in b.get("support", []) if s in boxes]
            if b["type"] == "table":
                vbox = next((s for s in sup if str(vtext.get(s) or "").strip() == v.strip()), None) or \
                       next((s for s in sup if boxes[s]["role"] in ("cell", "answer_text", "value_label")), None)
                row = _clean_label(b.get("row_key"))
                cols = [c for c in (_clean_label(x) for x in (b.get("col_path") or [])) if c]
                if not (vbox and row and cols):
                    continue
                other_rows = sorted({l for k in row_hdrs - set(sup) if (l := _clean_label(vtext.get(k))) and l.lower() != row.lower()})
                other_cols = sorted({l for k in col_hdrs - set(sup) if (l := _clean_label(vtext.get(k)))
                                     and l.lower() not in {c.lower() for c in cols}})
                slots.append({"type": "table", "value": v, "kind": _kind(v), "box": boxes[vbox]["bbox"],
                              "row": row, "cols": cols, "other_rows": other_rows, "other_cols": other_cols})
            elif b["type"] == "entity":
                cand = b.get("candidate")
                if cand not in boxes:
                    continue
                slots.append({"type": "entity", "value": v, "kind": _kind(v), "box": boxes[cand]["bbox"]})
        if slots:
            out[qid] = {"slots": slots}
    return out


# ---------------- scoring ----------------
def commit_text(full):
    i = full.rfind("</grounding>")
    return full[i + len("</grounding>"):] if i >= 0 else full


def clauses(text):
    """(start, end) spans split on sentence ends / semicolons / newlines; decimals and thousands are not split."""
    spans, s = [], 0
    for m in re.finditer(r"(?<!\d)[.;。！？!?](?!\d)|\n", text):
        spans.append((s, m.end()))
        s = m.end()
    spans.append((s, len(text)))
    return [(a, b) for a, b in spans if text[a:b].strip()]


def _value_positions(clause, slot):
    if slot["kind"] == "number":
        out = []
        nt = norm_num(slot["value"])
        for m in re.finditer(r"-?[\d][\d,.]*\d|\d", clause):
            if norm_num(m.group()) == nt and not _negated(clause, m.start()) and match_num(clause[max(0, m.start() - 12):m.end() + 3], slot["value"]):
                out.append(m.start())
        return out
    return [m.start() for m in re.finditer(re.escape(slot["value"].lower()), clause.lower())
            if match_txt(clause[max(0, m.start() - 1):m.end() + 1], slot["value"]) and not _negated(clause, m.start())]


def _label_pos(clause, label):
    return [m.start() for m in re.finditer(r"(?<![A-Za-z0-9])" + re.escape(label.lower()) + r"(?![A-Za-z0-9])", clause.lower())]


def _nearest(pos, clause, labels):
    best = None
    for lab in labels:
        for p in _label_pos(clause, lab):
            d = abs(p - pos)
            if best is None or d < best[0]:
                best = (d, lab)
    return best


def stated(slot, clause):
    for pos in _value_positions(clause, slot):
        if slot["type"] == "entity":
            return True
        col = _nearest(pos, clause, slot["cols"] + slot["other_cols"])
        row = _nearest(pos, clause, [slot["row"]] + slot["other_rows"])
        if col and col[1] in slot["cols"] and row and row[1] == slot["row"]:
            return True
    return False


def supported(slot, wh, executed_boxes):
    W, H = wh
    b = slot["box"]
    bpx = (b[0] * W, b[1] * H, b[2] * W, b[3] * H)
    detail = bpx[3] - bpx[1] if slot["type"] == "table" or slot["kind"] == "text" else min(bpx[2] - bpx[0], bpx[3] - bpx[1])
    kind = "text"
    for box, s in RS._views(W, H, executed_boxes):
        if RS._cover(bpx, box) >= RS.COVER and RS._h(kind, detail * s) >= SUPPORT_A:
            return True
    return False


def score(entry, wh, executed_boxes, full_response):
    """-> (E, u_list, spans): spans[k] = (start, end) of the supporting clause within the FULL response, or None."""
    text = commit_text(full_response)
    off = len(full_response) - len(text)
    cl = clauses(text)
    us, spans = [], []
    for slot in entry["slots"]:
        hit = next(((a, b) for a, b in cl if stated(slot, text[a:b])), None)
        ok = hit is not None and supported(slot, wh, executed_boxes)
        us.append(1.0 if ok else 0.0)
        spans.append((hit[0] + off, hit[1] + off) if ok else None)
    return (sum(us) / len(us) if us else 0.0), us, spans
