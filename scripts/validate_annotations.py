"""Validate RL-pool evidence annotations (ANNOTATION_SPEC v1 §6) and optionally render box overlays.

usage:
  python3 validate_annotations.py MANIFEST.jsonl ANNOT.jsonl [ANNOT2.jsonl ...]        # report errors / warnings
  python3 validate_annotations.py --render MANIFEST.jsonl ANNOT.jsonl IMAGE_ROOT OUT_DIR  # draw boxes for spot checks
  python3 validate_annotations.py --selftest
"""
import json
import re
import sys
from pathlib import Path

UNITS = {"single_box", "multi_set", "keyed_set", "pair_relation", "multi_compare", "ordered_chain",
         "region_global", "absent", "unanswerable"}
REASONS = {"needs_temporal", "needs_external_knowledge", "evidence_not_found", "ambiguous_question"}
GT = {"agree", "disagree", "cannot_tell"}


def bad_box(b) -> bool:
    return not (isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b)
                and 0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1)


def area(b) -> float:
    return (b[2] - b[0]) * (b[3] - b[1])


def iou(a, b) -> float:
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / (area(a) + area(b) - i) if i else 0.0


def check(rec: dict, item: dict | None) -> tuple[list[str], list[str]]:
    e, w = [], []
    if item is None:
        return ["uid_not_in_manifest"], w
    u = rec.get("unit")
    if u not in UNITS:
        return [f"bad_unit:{u}"], w
    boxes = rec.get("boxes") or []
    all_b = [x.get("bbox") for x in boxes]
    for k in ("negatives", "checked_candidates", "key_parts"):
        all_b += [x.get("bbox") for x in rec.get(k) or []]
    all_b += list(rec.get("search_scope") or [])
    for b in all_b:
        if bad_box(b):
            e.append(f"bad_bbox:{b}")
    if e:
        return e, w
    for x in boxes:
        if u != "region_global" and area(x["bbox"]) > 0.9 and not rec.get("note"):
            e.append("mega_box_without_note")
    roles = [x.get("role") for x in boxes]
    need = {
        "single_box": len(boxes) == 1,
        "multi_set": len(boxes) >= 1,
        "keyed_set": len(boxes) >= 2 and len(set(roles)) >= 2,
        "pair_relation": sorted(roles) == ["reference", "subject"] and bool(rec.get("relation")),
        "multi_compare": len(boxes) >= 2,
        "ordered_chain": len(boxes) >= 2 or len(rec.get("points") or []) >= 2,
        "region_global": roles.count("main_figure") == 1,
        "absent": not boxes and bool(rec.get("search_scope")),
        "unanswerable": not boxes and rec.get("unanswerable_reason") in REASONS,
    }
    if not need[u]:
        e.append(f"unit_fields:{u}")
    if rec.get("gt_check") not in GT:
        e.append("bad_gt_check")
    elif rec["gt_check"] != "agree" and not rec.get("gt_note"):
        e.append("gt_note_missing")
    # consistency warnings
    if u == "multi_set" and re.fullmatch(r"\s*\d+\s*\.?\s*", item.get("gt_option", "")):
        if len(boxes) != int(item["gt_option"].strip(" .")):
            w.append(f"count_mismatch:{len(boxes)}vs{item['gt_option'].strip()}")
    if u == "pair_relation" and not e:
        s = next(x["bbox"] for x in boxes if x["role"] == "subject")
        r = next(x["bbox"] for x in boxes if x["role"] == "reference")
        dx, dy = (s[0] + s[2] - r[0] - r[2]) / 2, (s[1] + s[3] - r[1] - r[3]) / 2
        rel = rec["relation"]
        for word, ok in (("left_of", dx < 0), ("right_of", dx > 0), ("above", dy < 0), ("below", dy > 0)):
            if word in rel and not ok:
                w.append(f"relation_geometry:{word}")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if iou(boxes[i]["bbox"], boxes[j]["bbox"]) > 0.9:
                w.append(f"duplicate_boxes:{i},{j}")
    return e, w


def load(p):
    return [json.loads(x) for x in open(p) if x.strip()]


def validate(manifest: str, files: list[str]) -> int:
    items = {r["uid"]: r for r in load(manifest)}
    seen, n_err = set(), 0
    for f in files:
        for rec in load(f):
            uid = rec.get("uid")
            e, w = check(rec, items.get(uid))
            if uid in seen:
                e.append("duplicate_uid")
            seen.add(uid)
            n_err += bool(e)
            if e or w:
                print(json.dumps({"uid": uid, "file": f, "errors": e, "warnings": w}, ensure_ascii=False))
    print(f"# records={len(seen)} with_errors={n_err} remaining={len(set(items) - seen)}")
    return 1 if n_err else 0


def render(manifest: str, annot: str, root: str, out: str) -> None:
    from PIL import Image, ImageDraw
    items = {r["uid"]: r for r in load(manifest)}
    Path(out).mkdir(parents=True, exist_ok=True)
    colors = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#9a6324"]
    for rec in load(annot):
        it = items[rec["uid"]]
        im = Image.open(Path(root) / it["image_rel"]).convert("RGB")
        W, H = im.size
        d = ImageDraw.Draw(im)
        lw = max(2, round(max(W, H) / 400))
        layers = [(x, colors[i % 8]) for i, x in enumerate(rec.get("boxes") or [])]
        layers += [(x, "#888888") for x in (rec.get("negatives") or []) + (rec.get("checked_candidates") or [])]
        for x, c in layers:
            b = x["bbox"]
            d.rectangle([b[0] * W, b[1] * H, b[2] * W, b[3] * H], outline=c, width=lw)
            d.text((b[0] * W + lw, b[1] * H + lw), f"{x.get('role', 'neg')}:{x.get('label', '')}"[:40], fill=c)
        for b in rec.get("search_scope") or []:
            d.rectangle([b[0] * W, b[1] * H, b[2] * W, b[3] * H], outline="#000000", width=lw)
        im.thumbnail((1600, 1600))
        im.save(Path(out) / f"{rec['uid']}.jpg", quality=88)


def selftest() -> None:
    item = {"uid": "u", "gt_option": "3"}
    ok = {"uid": "u", "unit": "multi_set", "gt_check": "agree",
          "boxes": [{"role": "instance", "bbox": [0.1 * i, 0.1, 0.1 * i + 0.05, 0.2]} for i in range(3)]}
    assert check(ok, item) == ([], [])
    assert check(dict(ok, boxes=ok["boxes"][:2]), item)[1] == ["count_mismatch:2vs3"]
    assert check(dict(ok, unit="single_box"), item)[0] == ["unit_fields:single_box"]
    assert check(dict(ok, boxes=[{"role": "x", "bbox": [0.5, 0.1, 0.4, 0.2]}]), item)[0][0].startswith("bad_bbox")
    rel = {"uid": "u", "unit": "pair_relation", "gt_check": "agree", "relation": "subject left_of reference",
           "boxes": [{"role": "subject", "bbox": [0.6, 0, 0.7, 0.1]}, {"role": "reference", "bbox": [0.1, 0, 0.2, 0.1]}]}
    assert check(rel, item)[1] == ["relation_geometry:left_of"]
    ab = {"uid": "u", "unit": "absent", "gt_check": "disagree", "gt_note": "", "boxes": [], "search_scope": [[0, 0, 1, 1]]}
    assert check(ab, item)[0] == ["gt_note_missing"]
    big = {"uid": "u", "unit": "single_box", "gt_check": "agree", "boxes": [{"role": "target", "bbox": [0, 0, 1, 0.95]}]}
    assert check(big, item)[0] == ["mega_box_without_note"]
    print("selftest ok")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--selftest":
        selftest()
    elif a and a[0] == "--render":
        render(*a[1:5])
    else:
        sys.exit(validate(a[0], a[1:]))
