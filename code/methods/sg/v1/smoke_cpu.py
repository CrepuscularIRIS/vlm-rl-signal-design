"""CPU smoke for grpo_sg (run on the server with the easyr1 env, from grpo_sg/).
1) continuation prefix: encode_prefix twice -> identical ids/grid; _fresh_env copy keeps injected views;
   actor re-render (render_views of env.boxes[1:]) gives the same grids; injected view PNGs saved for inspection.
2) task-scaled advantage: y/s decode and group normalisation on a synthetic batch.
usage: python smoke_cpu.py <model_dir> <sgc_train.parquet> <png_out_dir>
"""
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CAPTION_LAZY_MM", "1")
from transformers import AutoProcessor  # noqa: E402

import caption_env as CENV  # noqa: E402
import easyr1_caption as E  # noqa: E402
from caption_contracts import grid_list  # noqa: E402

model, pq, outd = sys.argv[1:4]
os.makedirs(outd, exist_ok=True)
proc = AutoProcessor.from_pretrained(model, trust_remote_code=True)
df = pd.read_parquet(pq)
cont = df[df["cont_boxes"].notna()].head(3)
MIN_PX, MAX_PX = 1024, 589824
for _, r in cont.iterrows():
    boxes = E._cont_boxes(r["cont_boxes"])
    env1 = CENV.CropEnv(r["images"][0])
    ids1, imgs1, g1 = E.encode_prefix(proc, env1, r["env_head"], r["env_tail"], MIN_PX, MAX_PX, boxes)
    env2 = CENV.CropEnv(r["images"][0])
    ids2, _, g2 = E.encode_prefix(proc, env2, r["env_head"], r["env_tail"], MIN_PX, MAX_PX, boxes)
    assert ids1 == ids2 and g1 == g2
    fresh = E._fresh_env(env1)
    assert fresh.crops == len(boxes) and len(fresh.images) == len(boxes) + 1 and fresh.boxes == env1.boxes
    rendered = CENV.render_views(r["images"][0], env1.boxes[1:], MIN_PX, MAX_PX)
    feats = proc.image_processor(images=rendered, return_tensors="pt")
    assert grid_list(feats["image_grid_thw"]) == g1, (grid_list(feats["image_grid_thw"]), g1)
    for k, im in enumerate(imgs1[1:], 1):
        im.save(f"{outd}/{r['uid']}_view{k}.png")
    print("ok", r["uid"], "prefix_tokens", len(ids1), "views", len(boxes), "grid", g1)
    print("   Q:", r["env_head"][:240].replace("\n", " | "))

E.install_task_scaled_advantage()
import verl.trainer.core_algos as CA  # noqa: E402
lam = 0.1
# group a: mixed (y = 1,0,...), group b: all wrong with different S, group c: all correct, identical S
y = [1, 0, 0, 0, 0, 0, 0, 0] + [0] * 8 + [1] * 8
s = [0.5, 0.9, 0, 0, 0, 0, 0, 0] + [0, 0.2, 0.4, 0.6, 0.8, 1.0, 0, 0] + [0.3] * 8
R = torch.tensor([yi + lam * si for yi, si in zip(y, s)])
tlr = torch.zeros(24, 5)
tlr[:, -1] = R
mask = torch.ones(24, 5)
index = ["a"] * 8 + ["b"] * 8 + ["c"] * 8
adv, _ = CA.ADV_ESTIMATOR_MAP["grpo"](token_level_rewards=tlr, response_mask=mask, index=index)
A = adv[:, 0]
assert A[0] > 0 and all(A[1:8] < 0), A[:8]                       # process term never flips mixed-group signs
sb = s[8:16]
want_b = [lam * (x - sum(sb) / 8) / (1 / 8 ** 0.5) for x in sb]      # all-wrong: lam*(s - mean s)/sigma_min
assert max(abs(float(a) - w) for a, w in zip(A[8:16], want_b)) < 1e-4, (A[8:16], want_b)
assert float(A[8:16].abs().max()) <= lam * 8 ** 0.5 + 1e-6           # |A| <= lam / sigma_min
assert float(A[16:].abs().max()) < 1e-6                           # all-correct, equal S: zero
print("adv ok: mixed", [round(float(x), 3) for x in A[:3]], "all-wrong max", round(float(A[8:16].max()), 4))
