# Render V* wrong-but-hit crops (smallest covering crop) with GT box outlined, for manual inspection.
# usage: python vstar_hit_render.py <traj.jsonl> <out_dir> [n] [seed]
import ast
import os
import json
import random
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from vstar_recheck import cover, find_item  # noqa: E402

TRAJ, OUT = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 8
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 7
ROOT = os.environ.get("VSTAR_ROOT", "obs/benchmarks/vstar/")

cands = []
for l in open(TRAJ):
    r = json.loads(l)
    if not r["uid"].startswith("vstar") or r["ok"]:
        continue
    it = find_item(r["uid"])
    b = r["boxes_abs"]
    crops = (ast.literal_eval(b) if isinstance(b, str) else (b or []))[1:]
    gts = [[x, y, x + w, y + h] for x, y, w, h in it["bbox"]]
    best = []
    for g in gts:
        cs = [c for c in crops if cover(c, g) >= 0.9]
        if not cs:
            best = None
            break
        best.append(min(cs, key=lambda c: (c[2] - c[0]) * (c[3] - c[1])))
    if best:
        cands.append((r, it, gts, best))

random.Random(SEED).shuffle(cands)
meta = []
for k, (r, it, gts, best) in enumerate(cands[:N]):
    im = Image.open(ROOT + it["image_path"]).convert("RGB")
    c = best[0]
    crop = im.crop(tuple(int(v) for v in c))
    d = ImageDraw.Draw(crop)
    for g in gts:
        d.rectangle([g[0] - c[0], g[1] - c[1], g[2] - c[0], g[3] - c[1]], outline=(255, 0, 0), width=2)
    name = f"{OUT}/vstar_hit_{k:02d}_{r['uid']}_s{r['sample_idx']}.png"
    crop.save(name)
    gw, gh = gts[0][2] - gts[0][0], gts[0][3] - gts[0][1]
    meta.append({"k": k, "uid": r["uid"], "s": r["sample_idx"], "item": it["item_id"], "q": it["question"].split("\nAnswer")[0],
                 "gt": r["gt"], "pred": r["pred"], "crop": c, "crop_wh": [c[2] - c[0], c[3] - c[1]], "gt_box_wh": [gw, gh],
                 "commit": (r.get("final_text") or "")[-350:]})
json.dump(meta, open(f"{OUT}/vstar_hit_meta.json", "w"), ensure_ascii=False, indent=1)
print(f"{len(cands)} wrong-hit candidates; rendered {len(meta)}")
for m in meta:
    print(m["k"], m["uid"], m["item"], "gt", m["gt"], "pred", m["pred"], "crop_wh", m["crop_wh"], "gtbox_wh", m["gt_box_wh"])
    print("   Q:", m["q"].replace("\n", " | "))
    print("   C:", m["commit"].replace("\n", " ")[-260:])
