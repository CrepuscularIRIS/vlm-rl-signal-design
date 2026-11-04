# Re-check of V* "seen-but-wrong" decomposition: how many "hits" come from near-full-image crops?
# usage: python vstar_recheck.py <traj.jsonl> [...]
import ast
import os
import json
import sys

HERE = __file__.rsplit("/", 1)[0]
VSTAR = os.environ.get("VSTAR_SUBSET", "obs/benchmarks/vstar/subset.jsonl")
uid_q = json.load(open(f"{HERE}/vstar_uid_q.json"))
items = [json.loads(l) for l in open(VSTAR)]
by_head = {it["question"][:60]: it for it in items}


def find_item(uid):
    q = uid_q[uid]
    q = q.split("Question:", 1)[-1].strip() if "Question:" in q else q
    for head, it in by_head.items():
        if q.startswith(head[:60]) or head[:60] in uid_q[uid]:
            return it
    return None


def cover(crop, box):
    x1, y1, x2, y2 = crop
    bx1, by1, bx2, by2 = box
    iw = max(0, min(x2, bx2) - max(x1, bx1))
    ih = max(0, min(y2, by2) - max(y1, by1))
    a = (bx2 - bx1) * (by2 - by1)
    return iw * ih / a if a > 0 else 0


def run(path):
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r["uid"].startswith("vstar")]
    stats = {"n": 0, "unmatched": 0, "wrong": 0, "wrong_hit": 0, "wrong_hit_small": 0, "wrong_hit_big": 0,
             "right": 0, "right_hit": 0}
    for r in rows:
        it = find_item(r["uid"])
        if it is None:
            stats["unmatched"] += 1
            continue
        stats["n"] += 1
        W, H = r["image_wh"]
        b = r["boxes_abs"]
        crops = (ast.literal_eval(b) if isinstance(b, str) else (b or []))[1:]
        gts = [[x, y, x + w, y + h] for x, y, w, h in it["bbox"]]
        # hit = every GT box covered >=90% by some single crop; record smallest such crop area ratio
        ratios = []
        hit = True
        for g in gts:
            cs = [c for c in crops if cover(c, g) >= 0.9]
            if not cs:
                hit = False
                break
            ratios.append(min((c[2] - c[0]) * (c[3] - c[1]) / (W * H) for c in cs))
        key = "right" if r["ok"] else "wrong"
        stats[key] += 1
        if hit:
            stats[key + "_hit"] += 1
            if key == "wrong":
                stats["wrong_hit_big" if max(ratios) >= 0.25 else "wrong_hit_small"] += 1
    s = stats
    print(path.rsplit("/", 1)[-1], s,
          f"| wrong-hit {s['wrong_hit']}/{s['wrong']}={s['wrong_hit']/max(1,s['wrong']):.1%}"
          f" | of which via crop>=25% image area: {s['wrong_hit_big']}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        run(p)
