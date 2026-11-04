"""Manual-audit dump for Phi_bind v1: 15 per cell (phi>0 / phi=0) x (correct / wrong), random.Random(7).
Per slot: stated? supported? and the final-turn evidence text (metadata stripped).
usage: python audit_sample.py <bind_index.json> <bind_scores.jsonl> <out.jsonl> <traj.jsonl> [...]"""
import ast
import json
import random
import sys

import bind_scorer as B

idx = json.load(open(sys.argv[1]))
S = [json.loads(l) for l in open(sys.argv[2])]
T = {(r["uid"], r["sample_idx"]): r for p in sys.argv[4:] for r in map(json.loads, open(p))}
rng = random.Random(7)
pick = []
for pos in (True, False):
    for ok in (1, 0):
        cell = [s for s in S if (s["phi"] > 0) == pos and s["ok"] == ok]
        pick += [(f"phi{'>0' if pos else '=0'}_{'right' if ok else 'wrong'}", s) for s in rng.sample(cell, min(15, len(cell)))]
with open(sys.argv[3], "w") as f:
    for i, (cell, s) in enumerate(pick):
        r = T[(s["uid"], s["s"])]
        t = r["turns"]
        t = ast.literal_eval(t) if isinstance(t, str) else t
        full = "".join(x.get("text") or "" for x in t)
        text = B.strip_meta(B.commit_text(full))
        b = r["boxes_abs"]
        b = ast.literal_eval(b) if isinstance(b, str) else b
        crops = [tuple(map(float, x)) for x in b[1:]]
        slots = [{"row": sl["row"], "path": sl["path"], "value": sl["value"],
                  "stated": B.stated(sl, text), "supported": B.supported(sl, tuple(r["image_wh"]), crops)}
                 for sl in idx[s["uid"]]["slots"]]
        rec = {"i": i, "cell": cell, "uid": s["uid"], "s": s["s"], "gt": r["gt"], "pred": r["pred"], "phi": s["phi"],
               "n_crops": len(crops), "slots": slots, "text": " ".join(text.split())[:600]}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{i:02d}] {cell} {s['uid']} gt={r['gt']} pred={r['pred']} phi={s['phi']:.2f} crops={len(crops)}")
        for sl in slots:
            print(f"     slot row='{sl['row'][:40]}' col={sl['path']} val={sl['value']} stated={sl['stated']} supported={sl['supported']}")
        print(f"     TEXT: {rec['text'][:420]}")
