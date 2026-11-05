"""Build one review sheet per item (overview with labelled boxes + full-resolution crops) and a text packet,
so a reviewer agent needs one image read and one text read per item.

usage: python -I make_review_sheets.py MANIFEST IMAGE_DIR ANNOT_DIR UIDS_FILE SHEET_DIR PACKET_JSONL
PACKET_JSONL lines: {uid, sheet, image, W, H, question, gt, gt_option, L0, L1, L2}
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import annotate_api as A  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def sheet(im: Image.Image, l1: dict | None) -> Image.Image:
    ov = im.copy()
    ov.thumbnail((1400, 1400))
    left = A.overlay(ov, l1) if l1 else ov
    cs = A.crops(im, l1, limit=12) if l1 else []
    cw = 420
    tiles = []
    for bid, r, c in cs:
        c = c.copy()
        c.thumbnail((cw, cw))
        t = Image.new("RGB", (cw, c.height + 22), "white")
        t.paste(c, (0, 22))
        role = next((x.get("role", "neg") for k in ("boxes", "negatives", "checked_candidates", "key_parts")
                     for x in (l1.get(k) or []) if x.get("id") == bid), "")
        ImageDraw.Draw(t).text((4, 4), f"{bid} {role} region={r}"[:60], fill="black")
        tiles.append(t)
    cols = 2 if len(tiles) > 3 else 1
    colh = [0] * cols
    pos = []
    for t in tiles:
        k = colh.index(min(colh))
        pos.append((k, colh[k]))
        colh[k] += t.height + 6
    H = max(left.height, max(colh) if colh else 0)
    out = Image.new("RGB", (left.width + 8 + cols * (cw + 6), H), "#dddddd")
    out.paste(left, (0, 0))
    for (k, y), t in zip(pos, tiles):
        out.paste(t, (left.width + 8 + k * (cw + 6), y))
    return out


def main(manifest, image_dir, annot, uids_file, sheet_dir, packet):
    items = {json.loads(x)["uid"]: json.loads(x) for x in open(manifest)}
    ann = [{json.loads(x)["uid"]: json.loads(x) for x in open(Path(annot) / f"L{k}/batch.jsonl")} for k in range(3)]
    Path(sheet_dir).mkdir(parents=True, exist_ok=True)
    with open(packet, "w") as f:
        for uid in Path(uids_file).read_text().split():
            it = items[uid]
            l0, l1, l2 = (ann[k].get(uid) for k in range(3))
            im = Image.open(Path(image_dir) / it["image_rel"]).convert("RGB")
            p = Path(sheet_dir) / f"{uid}.jpg"
            sheet(im, l1).save(p, quality=88)
            rec = {"uid": uid, "sheet": str(p), "image": str(Path(image_dir) / it["image_rel"]),
                   "W": it["W"], "H": it["H"], "question": it["question"], "gt": it["gt"],
                   "gt_option": it["gt_option"], "L0": l0, "L1": l1, "L2": l2}
            (Path(sheet_dir) / f"{uid}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:7])
