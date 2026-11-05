"""Arm A+B+C training pool: the k8 sampler order (torch.randperm(4096, seed 42) == RandomSampler order, verified
32/32 step sets against v52mc_k8_grpo rollouts) with 384 evidence-continuation rows substituted, run shuffle=false.

Continuation rows (Design_New §4C): the same question with injected GT evidence view(s) built from L1 boxes of its
first L2 evidence set: one union view if every node reaches a_b >= U_GATE in it, else one view per node (<= 3).
Questions are taken from the END of the order (so every question still appears exactly once).
Schedule: 32 rows/step for steps 1-8, 16 rows/step for steps 9-16, 0 after (aux share decays to zero).
usage: python build_cont_pool.py <train.parquet> <out_dir>
"""
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import reward_sg as R  # noqa: E402

SCHEDULE = [32] * 8 + [16] * 8 + [0] * 16
STEP, CHUNK = 128, 32
MARGIN = 0.10
MIN_CROP_PX = 56


def node_px(n, W, H):
    b = n["bbox"]
    return (b[0] * W, b[1] * H, b[2] * W, b[3] * H)


def a_score(n, view, W, H):
    b = node_px(n, W, H)
    s = R.view_scale(view[2] - view[0], view[3] - view[1])
    if R._cover(b, view) < R.COVER:
        return 0.0
    detail = (b[3] - b[1]) / max(1, n.get("lines", 1)) if n["kind"] == "text" else min(b[2] - b[0], b[3] - b[1])
    return R._h(n["kind"], detail * s)


def pad(box, W, H):
    w, h = box[2] - box[0], box[3] - box[1]
    m = max(8.0, MARGIN * max(w, h))
    x0, y0, x1, y1 = max(0.0, box[0] - m), max(0.0, box[1] - m), min(W, box[2] + m), min(H, box[3] + m)
    if x1 - x0 < MIN_CROP_PX:  # mirror caption_env.expand_min_crop so the scale we check is the one executed
        c = (x0 + x1) / 2
        x0, x1 = max(0.0, c - MIN_CROP_PX / 2), min(W, c + MIN_CROP_PX / 2)
    if y1 - y0 < MIN_CROP_PX:
        c = (y0 + y1) / 2
        y0, y1 = max(0.0, c - MIN_CROP_PX / 2), min(H, c + MIN_CROP_PX / 2)
    return (x0, y0, x1, y1)


def gt_views(rec, W, H):
    nodes = [rec["nodes"][i] for i in rec["sets"][0]]
    px = [node_px(n, W, H) for n in nodes]
    union = pad((min(b[0] for b in px), min(b[1] for b in px), max(b[2] for b in px), max(b[3] for b in px)), W, H)
    if all(a_score(n, union, W, H) >= R.U_GATE for n in nodes):
        return [union]
    views = [pad(b, W, H) for n, b in zip(nodes, px)]
    if len(views) > 3 or not all(a_score(n, v, W, H) >= R.U_GATE for n, v in zip(nodes, views)):
        return None
    return views


def main(train_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(train_path)
    perm = torch.randperm(len(df), generator=torch.Generator().manual_seed(42)).tolist()
    order = df.iloc[perm].reset_index(drop=True)
    need = sum(SCHEDULE)
    idx = R.index()
    picked, why = [], {"m0": 0, "no_view": 0}
    for pos in range(len(order) - 1, -1, -1):
        if len(picked) == need:
            break
        row = order.iloc[pos]
        rec = idx.get(row["uid"], {})
        if not rec.get("m"):
            why["m0"] += 1
            continue
        W, H = Image.open(row["images"][0]).size
        views = gt_views(rec, W, H)
        if views is None:
            why["no_view"] += 1
            continue
        picked.append((pos, [[v[0] / W, v[1] / H, v[2] / W, v[3] / H] for v in views]))
    assert len(picked) == need, (len(picked), need)
    drop = {p for p, _ in picked}
    originals = [order.iloc[p] for p in range(len(order)) if p not in drop]
    conts = []
    for p, boxes in picked:
        r = order.iloc[p].copy()
        r["cont_boxes"] = boxes
        conts.append(r)
    rows, oi, ci = [], 0, 0
    for k in SCHEDULE:                       # spread k continuation rows evenly over the 4 chunks of the step
        per_chunk = k // (STEP // CHUNK)
        for _ in range(STEP // CHUNK):
            for j in range(CHUNK):
                if j < per_chunk:
                    rows.append(conts[ci]); ci += 1
                else:
                    rows.append(originals[oi]); oi += 1
    assert oi == len(originals) and ci == len(conts) and len(rows) == len(df)
    res = pd.DataFrame(rows).reset_index(drop=True)
    res["cont_boxes"] = [r.get("cont_boxes") if isinstance(r.get("cont_boxes"), list) else None for r in rows]
    assert res["uid"].nunique() == len(df)
    res.to_parquet(out / "train.parquet")
    nviews = [len(b) for _, b in picked]
    manifest = {"rows": len(res), "cont_rows": len(conts), "schedule": SCHEDULE, "skipped_from_tail": why,
                "views_per_cont": {str(v): nviews.count(v) for v in sorted(set(nviews))},
                "order": "torch.randperm(n, Generator().manual_seed(42)) == k8 RandomSampler order",
                "cont_uids": [order.iloc[p]["uid"] for p, _ in picked]}
    (out / "CONT_MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print({k: v for k, v in manifest.items() if k != "cont_uids"})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
