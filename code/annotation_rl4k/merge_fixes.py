"""Overlay review fixes on API annotations, validate, optionally write the merged final set.

usage: python -I merge_fixes.py MANIFEST ANNOT FIX [FIX2 ...] [--write FINAL_DIR]
Later FIX dirs override earlier ones; within a dir the last record per (uid, tier) wins. Prints errors for every
reviewed uid (and with --write, for all merged uids) plus a summary line starting with '#'.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import annotate_api as A  # noqa: E402
import validate_annotations as V  # noqa: E402


def load(d: Path, k: int) -> dict:
    p = d / f"L{k}" / "batch.jsonl"
    out = {}
    if p.exists():
        for x in open(p):
            if x.strip():
                r = json.loads(x)
                out[r["uid"]] = r
    return out


def main(argv):
    write = None
    if "--write" in argv:
        i = argv.index("--write")
        write, argv = Path(argv[i + 1]), argv[:i] + argv[i + 2:]
    manifest, annot, fixes = argv[0], Path(argv[1]), [Path(x) for x in argv[2:]]
    items = {json.loads(x)["uid"]: json.loads(x) for x in open(manifest)}
    merged = [load(annot, k) for k in range(3)]
    reviewed, verdicts = set(), {}
    for fx in fixes:
        for k in range(3):
            merged[k].update(load(fx, k))
        if (fx / "review.jsonl").exists():
            for x in open(fx / "review.jsonl"):
                if x.strip():
                    r = json.loads(x)
                    reviewed.add(r["uid"])
                    verdicts[r["uid"]] = r.get("verdict")
    uids = sorted(merged[0]) if write else sorted(reviewed)
    n_err = 0
    final = [[], [], []]
    for uid in uids:
        l0 = merged[0].get(uid)
        if l0 is None or uid not in items:
            print(json.dumps({"uid": uid, "errors": ["missing_L0_or_not_in_manifest"]}))
            n_err += 1
            continue
        need = V.REQUIRED_TIER.get(l0.get("unit"), 0)
        recs = [l0] + [merged[k].get(uid) if k <= need else None for k in (1, 2)]
        if recs[1] is not None:
            recs[1] = dict(recs[1], unit=l0.get("unit"))
        err = A.errors_of(items[uid], *recs)
        if err:
            n_err += 1
            print(json.dumps({"uid": uid, "errors": err}, ensure_ascii=False))
        for k, r in enumerate(recs):
            if r is not None:
                final[k].append(r)
    if write:
        for k in range(3):
            (write / f"L{k}").mkdir(parents=True, exist_ok=True)
            with open(write / f"L{k}" / "batch.jsonl", "w") as f:
                for r in final[k]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    vc = {}
    for v in verdicts.values():
        vc[v] = vc.get(v, 0) + 1
    print("# " + json.dumps({"checked": len(uids), "with_errors": n_err, "reviewed": len(reviewed), "verdicts": vc}))


if __name__ == "__main__":
    main(sys.argv[1:])
