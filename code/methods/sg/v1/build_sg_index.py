"""Build sg_index.json for reward_sg from rl4k L1/L2 (final) + the D2 eligibility index.

m = 1 iff: D2 reason == "enabled" (answerable, no GT disagreement, confidence ok, primary role present),
L1 unit not in {absent, region_global}, L1 confidence != low, and >= 1 evidence set survives
(every member box present in L1 and not marked readable=false in L2).
usage: python build_sg_index.py <L1.jsonl> <L2.jsonl> <d2_index.json> <out.json>
"""
import json
import sys
from collections import Counter

TEXT = {"cell", "answer_text", "row_header", "col_header", "axis_tick", "panel_title", "value_label", "legend",
        "footnote", "step"}
CONTEXT = {"membership_context", "main_figure"}
KEY = {"cell", "answer_text", "value_label", "instance", "target", "data_mark", "subject"}


def kind(role):
    return "text" if role in TEXT else ("context" if role in CONTEXT else "object")


def build(l1_path, l2_path, d2_path):
    L1 = {r["uid"]: r for r in map(json.loads, filter(str.strip, open(l1_path)))}
    L2 = {r["uid"]: r for r in map(json.loads, filter(str.strip, open(l2_path)))}
    d2 = json.load(open(d2_path))
    out, why = {}, Counter()
    for qid, rec in d2.items():
        l1, l2 = L1.get(qid), L2.get(qid) or {}
        if rec.get("reason") != "enabled":
            why[rec.get("reason")] += 1
            out[qid] = {"m": 0, "why": rec.get("reason")}
            continue
        if l1 is None or not l1.get("boxes"):
            why["no_l1_boxes"] += 1
            out[qid] = {"m": 0, "why": "no_l1_boxes"}
            continue
        if l1.get("unit") in ("absent", "region_global") or l1.get("annotation_confidence") == "low":
            r = "unit_" + l1["unit"] if l1.get("unit") in ("absent", "region_global") else "l1_low_conf"
            why[r] += 1
            out[qid] = {"m": 0, "why": r}
            continue
        values = l2.get("values") or {}
        nodes = {}
        for b in l1["boxes"]:
            v = values.get(b["id"]) or {}
            text = v.get("text") if isinstance(v, dict) else None
            nodes[b["id"]] = {"bbox": b["bbox"], "kind": kind(b["role"]), "role": b["role"],
                              "lines": max(1, str(text).count("\n") + 1) if text else 1,
                              "key": b["role"] in KEY,
                              "unreadable": isinstance(v, dict) and v.get("readable") is False}
        raw_sets = l2.get("evidence_sets") or [[b["id"] for b in l1["boxes"]]]
        sets = [S for S in raw_sets if S and all(i in nodes and not nodes[i]["unreadable"] for i in S)]
        if not sets:
            why["no_valid_set"] += 1
            out[qid] = {"m": 0, "why": "no_valid_set"}
            continue
        used = {i for S in sets for i in S}
        nodes = {i: n for i, n in nodes.items() if i in used}
        if not any(n["key"] for n in nodes.values()):  # no answer-bearing role: every set member can open the gate
            for n in nodes.values():
                n["key"] = True
        for n in nodes.values():
            n.pop("unreadable")
        why["enabled"] += 1
        out[qid] = {"m": 1, "sets": sets, "nodes": nodes}
    return out, why


if __name__ == "__main__":
    idx, why = build(*sys.argv[1:4])
    json.dump(idx, open(sys.argv[4], "w"), ensure_ascii=False)
    print(len(idx), dict(why))
    kinds = Counter(n["kind"] for r in idx.values() if r.get("m") for n in r["nodes"].values())
    print("node kinds", dict(kinds), "sets/q", Counter(len(r["sets"]) for r in idx.values() if r.get("m")))
