"""Cut newly finished API-annotated uids into 100-item review batches and build their sheets.

usage: python -I prep_review.py ANNOT_DIR MANIFEST POOL_TAG [--final]   (--final also emits the last partial batch)
Writes review_batches/<POOL_TAG>_rvNNN.txt and sheets to /tmp/annot_review_sheets/<POOL_TAG>_rvNNN/. Prints new batch ids.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
annot, manifest, tag = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
final = "--final" in sys.argv
RB = HERE / "review_batches"
RB.mkdir(exist_ok=True)
done = [json.loads(x)["uid"] for x in open(annot / "run.log") if '"uid"' in x and '"fatal"' not in x]
assigned = set()
for f in RB.glob(f"{tag}_rv*.txt"):
    assigned |= set(f.read_text().split())
todo = [u for u in dict.fromkeys(done) if u not in assigned]
k = len(list(RB.glob(f"{tag}_rv*.txt")))
new = []
while len(todo) >= 100 or (final and todo):
    k += 1
    bid = f"{tag}_rv{k:03d}"
    (RB / f"{bid}.txt").write_text("\n".join(todo[:100]) + "\n")
    todo = todo[100:]
    subprocess.run([sys.executable, "-I", str(HERE / "make_review_sheets.py"), manifest,
                    "/tmp/annot_imgs_20261101/rl4k_orig", str(annot), str(RB / f"{bid}.txt"),
                    f"/tmp/annot_review_sheets/{bid}", f"/tmp/annot_review_sheets/{bid}.packet.jsonl"], check=True)
    new.append(bid)
print(" ".join(new))
