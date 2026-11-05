"""Clue-layer (LC) annotation for items without v2 evidence boxes, per CLUE_ANNOTATION_AND_CONTROLLED_RUNS §2–§3.

modes:
  vision MANIFEST IMAGE_DIR OUT [--workers 24]   two passes (draft -> overlay+crops), classes needing images
  text   MANIFEST FINAL_V2 OUT                   reading_answer for gt_check=disagree items from v2 L2 + gt_note
  validate MANIFEST OUT                           check required fields per class
"""
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import annotate_api as A  # noqa: E402
import validate_annotations as V  # noqa: E402

SPEC = (Path(__file__).resolve().parent / "ANNOTATION_SPEC.md").read_text()
CLASSES = {"needs_temporal", "ambiguous", "external", "not_found", "low_res", "insufficient", "disagree"}

SYSTEM = SPEC + """

---
# Clue-layer session (API mode, v2c)
You annotate ONE item per conversation; the images you receive are all you get. Coordinates: normalized
[x1,y1,x2,y2] in [0,1] relative to the ORIGINAL image, 3 decimals. Return ONE JSON object, no prose.
This item's final answer cannot (or need not) be verified from the image. Do NOT force an answer-evidence frame.
Record the three testbed layers: what is VISIBLE (clue boxes), what it SUPPORTS (one short inference), what
remains UNRESOLVED. Boxes are audit references, never prescribed actions. Never edit gt.
"""

REQ = {
    "needs_temporal": "clues >=1 (signs, lane lines, light states, vehicle poses, SLOW markings...); unresolved non-empty (what a single frame cannot give: actual speed/steering/future motion).",
    "ambiguous": "interpretations >=2, each {label, reading, answer_letter}; clues optional (the competing regions).",
    "external": "clues = what the image actually shows (may be empty); unresolved = the outside conditions the answer needs.",
    "not_found": "search_scope boxes (regions that would contain the answer evidence if it existed) + checked_candidates [{id,label,bbox,reason}] (things that looked relevant but are not).",
    "low_res": "clues = parts that ARE clearly readable; also box the target region with readable=false in its label.",
    "insufficient": "same as low_res.",
}
TASK = """TASK (pass {p}): uid {uid}, class "{cls}". Original W={W} H={H}.
Question (complete):
{question}
gt letter: {gt} | gt option: {gt_opt}
{extra}
Return {{"clues":[{{"id":"b1","role":"clue","label":"...","bbox":[...],"supports":"..."}}],
"context":"...","inference":"...","unresolved":["..."],
"interpretations":[{{"label":"...","reading":"...","answer_letter":"?"}}] or null,
"search_scope":[[x1,y1,x2,y2]] or null,
"checked_candidates":[{{"id":"n1","label":"...","bbox":[...],"reason":"..."}}] or null,
"annotation_confidence":"high|medium|low","note":""}}
Requirements for class "{cls}": {req}"""


def class_of(l0):
    m = {"needs_temporal": "needs_temporal", "ambiguous_question": "ambiguous",
         "needs_external_knowledge": "external", "evidence_not_found": "not_found",
         "low_resolution": "low_res", "visually_insufficient": "insufficient"}
    return m.get(l0.get("unanswerable_reason") or l0.get("answerability"))


def bad(b):
    return V.bad_box(b)


def check(rec):
    e = []
    if rec.get("class") not in CLASSES:
        return ["bad_class"]
    for c in rec.get("clues") or []:
        if bad(c.get("bbox")):
            e.append(f"bad_clue_bbox:{c.get('id')}")
    for c in rec.get("checked_candidates") or []:
        if bad(c.get("bbox")):
            e.append("bad_candidate_bbox")
    for b in rec.get("search_scope") or []:
        if bad(b):
            e.append("bad_search_scope")
    cls = rec["class"]
    if cls == "needs_temporal" and (not rec.get("clues") or not rec.get("unresolved")):
        e.append("needs_temporal_fields")
    if cls == "ambiguous" and len(rec.get("interpretations") or []) < 2:
        e.append("ambiguous_needs_2_interpretations")
    if cls == "not_found" and not rec.get("search_scope"):
        e.append("not_found_needs_search_scope")
    if (cls in ("low_res", "insufficient")) and not rec.get("clues"):
        e.append("needs_readable_clues")
    if e:
        return e
    ids = [c.get("id") for c in (rec.get("clues") or []) + (rec.get("checked_candidates") or [])]
    if None in ids or len(ids) != len(set(ids)):
        e.append("dup_ids")
    return e


def vision_item(item, l0, image_dir, out, model):
    uid, log, t0 = item["uid"], [], __import__("time").time()
    im = Image.open(Path(image_dir) / item["image_rel"]).convert("RGB")
    ov = im.copy()
    ov.thumbnail((A.OVERVIEW, A.OVERVIEW))
    cls = class_of(l0)
    extra = (f"v2 L0 note: {l0.get('note','')} | gt_note: {l0.get('gt_note','')}" if l0.get("note") or l0.get("gt_note") else "")
    txt = TASK.format(p=1, uid=uid, cls=cls, W=item["W"], H=item["H"], question=item["question"],
                      gt=item["gt"], gt_opt=item["gt_option"], req=REQ[cls], extra=extra)
    parts = [{"type": "text", "text": txt}, A.img_part(ov)]
    for k, (box, t) in enumerate(A.tiles(im), 2):
        parts += [{"type": "text", "text": f"Tile {k}, normalized region {list(box)}:"}, A.img_part(t)]
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": parts}]
    res = A.call(model, msgs, log, "pass1")
    l1 = {"boxes": [dict(id=c["id"], role="clue", label=c.get("label"), bbox=c["bbox"])
                    for c in res.get("clues") or [] if not bad(c.get("bbox"))],
          "negatives": [dict(id=c["id"], label=c.get("label"), bbox=c["bbox"])
                        for c in res.get("checked_candidates") or [] if not bad(c.get("bbox"))]}
    txt2 = TASK.format(p=2, uid=uid, cls=cls, W=item["W"], H=item["H"], question=item["question"],
                       gt=item["gt"], gt_opt=item["gt_option"], req=REQ[cls], extra=extra) + \
        "\n\nBelow: overlay of your draft clue boxes on the overview, then full-resolution crops. Correct any box that truncates its clue; add missed clues; then return the SAME JSON schema complete."
    parts2 = [{"type": "text", "text": txt2}, {"type": "text", "text": "Overlay:"}, A.img_part(A.overlay(ov, l1))]
    for bid, r, c in A.crops(im, l1):
        parts2 += [{"type": "text", "text": f"Crop {bid}, region {r}:"}, A.img_part(c)]
    res = A.call(model, msgs + [{"role": "assistant", "content": json.dumps(res, ensure_ascii=False)},
                                {"role": "user", "content": parts2}], log, "pass2")
    rec = dict(res, uid=uid, spec_version="v2c", tier="C")
    rec["class"] = cls
    for k in range(2):
        err = check(rec)
        if not err:
            break
        res = A.call(model, msgs + [{"role": "assistant", "content": json.dumps(res, ensure_ascii=False)},
                                    {"role": "user", "content": [{"type": "text", "text":
                                     "Validator errors; return corrected full JSON: " + json.dumps(err)}]}],
                     log, f"repair{k+1}")
        rec = dict(res, uid=uid, spec_version="v2c", tier="C")
        rec["class"] = cls
    err = check(rec)
    sec = round(__import__("time").time() - t0, 1)
    tok = sum((x.get("usage") or {}).get("total_tokens", 0) for x in log)
    (Path(out) / "raw" / f"{uid}.json").write_text(json.dumps({"uid": uid, "sec": sec, "tokens": tok, "errors": err, "calls": log}, ensure_ascii=False))
    with A.LOCK:
        with open(Path(out) / "LC" / "batch.jsonl", "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return uid, sec, tok, err


def text_reading(item, l2, gt_note, out, model):
    uid, log = item["uid"], []
    body = {"uid": uid, "question": item["question"], "gt": item["gt"], "gt_option": item["gt_option"],
            "v2_values": (l2 or {}).get("values"), "v2_bindings": (l2 or {}).get("bindings"),
            "v2_summary": (l2 or {}).get("evidence_summary"), "gt_note": gt_note}
    txt = ("The annotation read values from the image that disagree with the supplied gt. Based ONLY on the read "
           "values/summary/gt_note, which option letter do the readings support? Return "
           '{"reading_answer":"A-J","confidence":"high|medium|low","why":"one sentence"}\n\n' + json.dumps(body, ensure_ascii=False))
    res = A.call(model, [{"role": "user", "content": txt}], log, "reading")
    rec = {"uid": uid, "spec_version": "v2c", "tier": "C", "class": "disagree",
           "clues": None, "context": None, "inference": None, "unresolved": None,
           "interpretations": None, "search_scope": None, "checked_candidates": None,
           "reading_answer": res.get("reading_answer"), "annotation_confidence": res.get("confidence"),
           "note": res.get("why", "")}
    err = [] if rec["reading_answer"] in list("ABCDEFGHIJ") else ["bad_reading_answer"]
    with A.LOCK:
        with open(Path(out) / "LC" / "batch.jsonl", "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return uid, err


def load_v0(v2dir):
    l0 = {json.loads(x)["uid"]: json.loads(x) for x in open(Path(v2dir) / "L0/batch.jsonl")}
    p2 = Path(v2dir) / "L2/batch.jsonl"
    l2 = {json.loads(x)["uid"]: json.loads(x) for x in open(p2)} if p2.exists() else {}
    return l0, l2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("manifest")
    ap.add_argument("dir2_or_imagedir")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--model", default="gpt-6-astra")
    ap.add_argument("--v2dir", default="$OBS/notes/rl_pool_annotation_20261101/out/final/rl4k")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "LC").mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(exist_ok=True)
    items = {json.loads(x)["uid"]: json.loads(x) for x in open(a.manifest)}
    if a.mode == "vision":
        l0, _ = load_v0(a.v2dir)
        done = set()
        p = out / "LC/batch.jsonl"
        if p.exists():
            done = {json.loads(x)["uid"] for x in open(p)}
        todo = [(it, l0[u]) for u, it in items.items() if u not in done and class_of(l0.get(u, {})) in CLASSES - {"disagree"}]
        with open(out / "run.log", "a") as lg:
            with ThreadPoolExecutor(a.workers) as ex:
                futs = {ex.submit(vision_item, it, l, a.dir2_or_imagedir, out, a.model): it["uid"] for it, l in todo}
                for n, f in enumerate(as_completed(futs), 1):
                    try:
                        uid, sec, tok, err = f.result()
                        lg.write(json.dumps({"uid": uid, "sec": sec, "tokens": tok, "errors": err}) + "\n")
                    except Exception as e:
                        lg.write(json.dumps({"uid": futs[f], "fatal": f"{type(e).__name__}: {e}"[:200]}) + "\n")
                    lg.flush()
                    if n % 25 == 0:
                        print(n, "/", len(todo), flush=True)
    elif a.mode == "text":
        l0, l2 = load_v0(a.dir2_or_imagedir)
        done = set()
        p = out / "LC/batch.jsonl"
        if p.exists():
            done = {json.loads(x)["uid"] for x in open(p) if '"class": "disagree"' in x}
        todo = [(it, l2.get(u), l0[u].get("gt_note", "")) for u, it in items.items()
                if u not in done and l0.get(u, {}).get("gt_check") == "disagree"]
        for it, l2r, note in todo:
            try:
                uid, err = text_reading(it, l2r, note, out, a.model)
                print(uid, err, flush=True)
            except Exception as e:
                print("FATAL", it["uid"], type(e).__name__, str(e)[:120], flush=True)
    elif a.mode == "validate":
        man = items
        n = err_n = 0
        for x in open(out / "LC/batch.jsonl"):
            rec = json.loads(x)
            if rec.get("class") == "disagree":
                err = [] if rec.get("reading_answer") in list("ABCDEFGHIJ") + ["none"] else ["bad_reading_answer"]
            else:
                err = check(rec)
            n += 1
            err_n += bool(err)
            if err:
                print(json.dumps({"uid": rec["uid"], "errors": err}, ensure_ascii=False))
        print(f"# LC records={n} with_errors={err_n}")
    else:
        raise SystemExit("mode?")


if __name__ == "__main__":
    main()
