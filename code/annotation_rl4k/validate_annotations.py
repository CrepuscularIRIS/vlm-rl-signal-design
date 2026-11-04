"""Validate tiered evidence annotations (ANNOTATION_SPEC v2 §6) and render box overlays.

usage:
  python3 validate_annotations.py MANIFEST.jsonl OUT_DIR            # OUT_DIR holds L0/ L1/ L2/ *.jsonl
  python3 validate_annotations.py --render MANIFEST.jsonl OUT_DIR IMAGE_ROOT OVERLAY_DIR [uid ...]
  python3 validate_annotations.py --selftest
"""
import json
import re
import sys
from pathlib import Path

UNITS = {"single_box", "multi_set", "keyed_set", "pair_relation", "multi_compare", "ordered_chain",
         "region_global", "absent", "unanswerable"}
REQUIRED_TIER = {"unanswerable": 0, "absent": 1, "region_global": 1, "single_box": 2, "multi_set": 2,
                 "keyed_set": 2, "pair_relation": 2, "multi_compare": 2, "ordered_chain": 2}
ANSWERABILITY = {"answerable", "target_absent", "visually_insufficient", "question_unanswerable"}
REASONS = {"needs_temporal", "needs_external_knowledge", "evidence_not_found", "ambiguous_question",
           "low_resolution"}
GT = {"agree", "disagree", "cannot_tell"}
BINDING_TYPES = {"table", "chart", "entity", "relation", "count", "compare", "chain"}
MULTI = {"multi_set", "keyed_set", "pair_relation", "multi_compare", "ordered_chain"}


def bad_box(b) -> bool:
    return not (isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b)
                and 0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1)


def area(b) -> float:
    return (b[2] - b[0]) * (b[3] - b[1])


def iou(a, b) -> float:
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / (area(a) + area(b) - i) if i else 0.0


def check_l0(r: dict) -> list[str]:
    e = []
    u, a = r.get("unit"), r.get("answerability")
    if u not in UNITS:
        return [f"L0_bad_unit:{u}"]
    if a not in ANSWERABILITY:
        e.append(f"L0_bad_answerability:{a}")
    elif (u == "absent") != (a == "target_absent") or (u == "unanswerable") != (a in {"visually_insufficient", "question_unanswerable"}):
        e.append("L0_unit_answerability_mismatch")
    if u == "unanswerable" and r.get("unanswerable_reason") not in REASONS:
        e.append("L0_unanswerable_reason")
    if r.get("required_tier") != REQUIRED_TIER[u]:
        e.append(f"L0_required_tier:{r.get('required_tier')}!={REQUIRED_TIER[u]}")
    if r.get("gt_check") not in GT:
        e.append("L0_bad_gt_check")
    elif r["gt_check"] != "agree" and not r.get("gt_note"):
        e.append("L0_gt_note_missing")
    return e


def all_boxes(l1: dict) -> list[dict]:
    out = list(l1.get("boxes") or [])
    for k in ("negatives", "checked_candidates", "key_parts"):
        out += list(l1.get(k) or [])
    return out


def check_l1(r: dict, unit: str) -> tuple[list[str], list[str]]:
    e, w = [], []
    boxes = r.get("boxes") or []
    objs = all_boxes(r)
    for x in objs:
        if bad_box(x.get("bbox")):
            e.append(f"L1_bad_bbox:{x.get('id')}")
    for b in r.get("search_scope") or []:
        if bad_box(b):
            e.append("L1_bad_search_scope")
    if e:
        return e, w
    ids = [x.get("id") for x in objs]
    if None in ids or len(ids) != len(set(ids)):
        e.append("L1_box_ids")
    if unit != "region_global" and any(area(x["bbox"]) > 0.9 for x in boxes) and not r.get("note"):
        e.append("L1_mega_box_without_note")
    roles = [x.get("role") for x in boxes]
    need = {
        "single_box": len(boxes) >= 1,
        "multi_set": len(boxes) >= 1,
        "keyed_set": len(boxes) >= 2 and len(set(roles)) >= 2,
        "pair_relation": "subject" in roles and "reference" in roles,
        "multi_compare": roles.count("candidate") >= 2,
        "ordered_chain": len(boxes) >= 2 or len(r.get("points") or []) >= 2,
        "region_global": roles.count("main_figure") == 1,
        "absent": not boxes and bool(r.get("search_scope")),
        "unanswerable": True,
    }
    if not need[unit]:
        e.append(f"L1_unit_fields:{unit}")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if iou(boxes[i]["bbox"], boxes[j]["bbox"]) > 0.9:
                w.append(f"L1_duplicate_boxes:{boxes[i]['id']},{boxes[j]['id']}")
    return e, w


def check_l2(r: dict, unit: str, l1: dict, gt_option: str) -> tuple[list[str], list[str]]:
    e, w = [], []
    known = {x["id"]: x for x in all_boxes(l1)}
    refs = list((r.get("values") or {}).keys())
    sets = r.get("evidence_sets") or []
    if not sets:
        e.append("L2_no_evidence_sets")
    for s in sets:
        refs += list(s)
    binds = r.get("bindings") or []
    if unit in MULTI and not binds:
        e.append("L2_no_bindings")
    for b in binds:
        t = b.get("type")
        if t not in BINDING_TYPES:
            e.append(f"L2_bad_binding_type:{t}")
            continue
        refs += list(b.get("support") or [])
        if t == "table" and not (b.get("row_key") and b.get("col_path") and b.get("value")):
            e.append("L2_table_binding_incomplete")
        if t == "count":
            ids = b.get("instance_ids") or []
            refs += ids
            if b.get("total") is not None and str(b["total"]).strip() != str(len(ids)):
                w.append(f"L2_count_total:{b['total']}vs{len(ids)}")
            if re.fullmatch(r"\s*\d+\s*\.?\s*", gt_option or "") and str(b.get("total")).strip() != gt_option.strip(" ."):
                w.append(f"L2_count_vs_gt:{b.get('total')}vs{gt_option.strip()}")
        if t == "relation":
            refs += [b.get("subject"), b.get("reference")]
            if b.get("computable_from_boxes") and b.get("subject") in known and b.get("reference") in known:
                s, q = known[b["subject"]]["bbox"], known[b["reference"]]["bbox"]
                dx, dy = (s[0] + s[2] - q[0] - q[2]) / 2, (s[1] + s[3] - q[1] - q[3]) / 2
                rel = str(b.get("relation"))
                for word, ok in (("left", dx < 0), ("right", dx > 0), ("above", dy < 0), ("below", dy > 0)):
                    if word in rel and not ok:
                        w.append(f"L2_relation_geometry:{word}")
        if t == "entity":
            refs.append(b.get("candidate"))
        if t in ("compare", "chain"):
            refs += list(b.get("candidates") or b.get("steps") or [])
    missing = sorted({x for x in refs if x not in known})
    if missing:
        e.append(f"L2_unknown_ids:{missing}")
    negs = {x["id"] for x in l1.get("negatives") or []}
    covered = {d.get("negative") for d in r.get("distinguishing_condition") or []}
    if negs - covered:
        w.append(f"L2_negative_without_condition:{sorted(negs - covered)}")
    return e, w


def load_dir(d: Path) -> tuple[dict, list[str]]:
    recs, dup = {}, []
    for f in sorted(d.glob("*.jsonl")) if d.exists() else []:
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                if r.get("uid") in recs:
                    dup.append(r.get("uid"))
                recs[r.get("uid")] = r
    return recs, dup


def validate(manifest: str, out_dir: str, quiet: bool = False) -> dict:
    items = {r["uid"]: r for r in map(json.loads, open(manifest)) if r}
    L = [load_dir(Path(out_dir) / f"L{k}") for k in range(3)]
    stats = {"items": len(items), "L0": len(L[0][0]), "L1": len(L[1][0]), "L2": len(L[2][0]),
             "complete": 0, "with_errors": 0}
    for k in range(3):
        for uid in L[k][1]:
            print(json.dumps({"uid": uid, "errors": [f"L{k}_duplicate_uid"]})) if not quiet else None
    for uid in sorted(set().union(*(set(x[0]) for x in L))):
        e, w = [], []
        if uid not in items:
            e.append("uid_not_in_manifest")
        l0, l1, l2 = (L[k][0].get(uid) for k in range(3))
        if l0 is None:
            e.append("L0_missing")
        else:
            e += check_l0(l0)
        unit = (l0 or {}).get("unit")
        if unit in UNITS and not e:
            need = REQUIRED_TIER[unit]
            done = max([k for k, x in enumerate((l0, l1, l2)) if x is not None])
            if l1 is not None:
                a, b = check_l1(l1, unit)
                e += a
                w += b
            if l2 is not None:
                if l1 is None:
                    e.append("L2_without_L1")
                else:
                    a, b = check_l2(l2, unit, l1, items.get(uid, {}).get("gt_option", ""))
                    e += a
                    w += b
            complete = done >= need and not e
            stats["complete"] += complete
        stats["with_errors"] += bool(e)
        if (e or w) and not quiet:
            print(json.dumps({"uid": uid, "errors": e, "warnings": w}, ensure_ascii=False))
    if not quiet:
        print("# " + json.dumps(stats))
    return stats


def render(manifest: str, out_dir: str, root: str, overlay: str, uids: list[str]) -> None:
    from PIL import Image, ImageDraw
    items = {r["uid"]: r for r in map(json.loads, open(manifest))}
    l1s, _ = load_dir(Path(out_dir) / "L1")
    Path(overlay).mkdir(parents=True, exist_ok=True)
    colors = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#9a6324"]
    for uid in uids or list(l1s):
        r, it = l1s[uid], items[uid]
        im = Image.open(Path(root) / it["image_rel"]).convert("RGB")
        W, H = im.size
        d = ImageDraw.Draw(im)
        lw = max(2, round(max(W, H) / 400))
        layers = [(x, colors[i % 8]) for i, x in enumerate(r.get("boxes") or [])]
        layers += [(x, "#888888") for x in (r.get("negatives") or []) + (r.get("checked_candidates") or [])]
        layers += [(x, "#00aaaa") for x in r.get("key_parts") or []]
        for x, c in layers:
            b = x["bbox"]
            d.rectangle([b[0] * W, b[1] * H, b[2] * W, b[3] * H], outline=c, width=lw)
            d.text((b[0] * W + lw, b[1] * H + lw), f"{x.get('id')}:{x.get('role', 'neg')}"[:30], fill=c)
        for b in r.get("search_scope") or []:
            d.rectangle([b[0] * W, b[1] * H, b[2] * W, b[3] * H], outline="#000000", width=lw)
        im.thumbnail((1600, 1600))
        im.save(Path(overlay) / f"{uid}.jpg", quality=88)


def selftest() -> None:
    import tempfile
    t = Path(tempfile.mkdtemp())
    (t / "m.jsonl").write_text(json.dumps({"uid": "u1", "gt_option": "2"}) + "\n" + json.dumps({"uid": "u2", "gt_option": "x"}) + "\n")
    for k in range(3):
        (t / "o" / f"L{k}").mkdir(parents=True)
    l0 = lambda uid, unit, tier: {"uid": uid, "unit": unit, "answerability": "answerable", "required_tier": tier, "gt_check": "agree"}
    (t / "o/L0/a.jsonl").write_text(json.dumps(l0("u1", "multi_set", 2)) + "\n" + json.dumps(l0("u2", "keyed_set", 2)) + "\n")
    inst = [{"id": f"b{i}", "role": "instance", "bbox": [0.1 * i, 0.1, 0.1 * i + 0.05, 0.2]} for i in (1, 2, 3)]
    tbl = [{"id": "b1", "role": "cell", "bbox": [0.5, 0.5, 0.6, 0.55]}, {"id": "b2", "role": "row_header", "bbox": [0.0, 0.5, 0.2, 0.55]}]
    (t / "o/L1/a.jsonl").write_text(json.dumps({"uid": "u1", "boxes": inst}) + "\n" + json.dumps({"uid": "u2", "boxes": tbl,
                                    "negatives": [{"id": "n1", "bbox": [0.62, 0.5, 0.7, 0.55]}]}) + "\n")
    (t / "o/L2/a.jsonl").write_text(
        json.dumps({"uid": "u1", "evidence_sets": [["b1", "b2", "b3"]], "bindings": [{"type": "count", "instance_ids": ["b1", "b2", "b3"], "total": 3, "support": []}]}) + "\n"
        + json.dumps({"uid": "u2", "evidence_sets": [["b1", "b9"]], "bindings": [{"type": "table", "row_key": "r", "col_path": [], "value": "1", "support": ["b1"]}]}) + "\n")
    s = validate(str(t / "m.jsonl"), str(t / "o"), quiet=True)
    assert s == {"items": 2, "L0": 2, "L1": 2, "L2": 2, "complete": 1, "with_errors": 1}, s
    l1 = {"boxes": inst}
    assert check_l2({"evidence_sets": [["b1"]], "bindings": [{"type": "count", "instance_ids": ["b1", "b2", "b3"], "total": 3}]}, "multi_set", l1, "2")[1] == ["L2_count_vs_gt:3vs2"]
    rel = {"boxes": [{"id": "s", "role": "subject", "bbox": [0.6, 0, 0.7, 0.1]}, {"id": "r", "role": "reference", "bbox": [0.1, 0, 0.2, 0.1]}]}
    assert check_l2({"evidence_sets": [["s", "r"]], "bindings": [{"type": "relation", "subject": "s", "reference": "r", "relation": "left_of", "frame": "image", "computable_from_boxes": True}]}, "pair_relation", rel, "")[1] == ["L2_relation_geometry:left"]
    assert check_l0({"unit": "absent", "answerability": "answerable", "required_tier": 1, "gt_check": "agree"}) == ["L0_unit_answerability_mismatch"]
    assert check_l1({"boxes": [{"id": "a", "role": "target", "bbox": [0, 0, 1, 0.95]}]}, "single_box")[0] == ["L1_mega_box_without_note"]
    print("selftest ok")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--selftest":
        selftest()
    elif a and a[0] == "--render":
        render(a[1], a[2], a[3], a[4], a[5:])
    else:
        st = validate(a[0], a[1])
        sys.exit(1 if st["with_errors"] else 0)
