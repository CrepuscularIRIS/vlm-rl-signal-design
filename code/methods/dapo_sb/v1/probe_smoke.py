"""GPU smoke for the DAPO-SB probe on root states (original image only). usage: python probe_smoke.py <model> <n>"""
import collections
import math
import sys

import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

sys.path.insert(0, "$PROJECT/grpo_sb")
import caption_env as CENV  # noqa: E402
import caption_v51 as proto  # noqa: E402
import stop_baseline as S  # noqa: E402
from verl.utils.dataset import process_image  # noqa: E402

model, n = sys.argv[1], int(sys.argv[2])
P = AutoProcessor.from_pretrained(model, trust_remote_code=True)
df = pd.read_parquet("$ROOT/datasets/grpo_caption_v51_4k/train.parquet").sample(n, random_state=7)
reqs, gts = [], []
for _, t in df.iterrows():
    env = CENV.CropEnv(t["images"][0])
    im = process_image(env.images[0], 1024, 589824)
    reqs.append(S.build_probe_request(P, proto, t["env_head"], [im], 1024, 589824, "<answer"))
    reqs.append(S.build_probe_request(P, proto, t["env_head"], [im], 1024, 589824, "<answer>"))
    gts.append(S._gt_letter(t["reward_model"]))
llm = LLM(model=model, trust_remote_code=True, max_model_len=16384, gpu_memory_utilization=0.6, limit_mm_per_prompt={"image": 10})
sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20, seed=0)
outs = llm.generate(reqs, sp, use_tqdm=False)
gi, pi = S.letter_token_ids(P.tokenizer, ">"), S.letter_token_ids(P.tokenizer, "")
gtok = P.tokenizer.encode(">", add_special_tokens=False)[0]
lps = [{int(t): x.logprob for t, x in o.outputs[0].logprobs[0].items()} for o in outs]
mass, b, hit, top = [], [], 0, collections.Counter()
for j, g in enumerate(gts):
    d = S.combine(lps[2 * j], lps[2 * j + 1], gi, pi, gtok)
    m = d.pop("_mass"); d.pop("_p_gt_sep")
    mass.append(m)
    b.append(d.get(g, 0.0))
    t = max(d, key=d.get) if d else "-"
    top[t] += 1
    hit += t == g
print(f"n={n} letter mass mean {sum(mass)/n:.3f} min {min(mass):.3f} | P(gt) mean {sum(b)/n:.3f} | argmax acc {hit/n:.3f} | argmax letters {dict(top)}")
print("sampled first tokens (prefix <answer):", collections.Counter(P.tokenizer.decode(o.outputs[0].token_ids) for o in outs[0::2]).most_common(6))
if len(sys.argv) > 3:
    shown = 0
    for j in range(len(gts)):
        if P.tokenizer.decode(outs[2*j].outputs[0].token_ids) == ">" and shown < 2:
            shown += 1
            for name, o in (("<answer", outs[2*j]), ("<answer>", outs[2*j+1])):
                top5 = sorted(o.outputs[0].logprobs[0].items(), key=lambda kv: -kv[1].logprob)[:6]
                print(name, "gt", gts[j], [(repr(P.tokenizer.decode([int(t)])), int(t), round(math.exp(x.logprob), 3)) for t, x in top5])
    print("encode(>)", P.tokenizer.encode(">", add_special_tokens=False), "tail of req2 ids", reqs[1]["prompt_token_ids"][-3:])
