"""DAPO-SB wiring (papers/Advantage.md, 11-04): pre-turn visual stopping probe + stop-relative advantage.

Probe b_{i,k}: before assistant turn k, the behavior policy (rollout engine weights = theta_old) sees ONLY the question
with its options and the images actually presented before that turn (original + first k crops, as presented), then is
forced to answer: assistant prefix "<answer" (the training text tokenizes "<answer>D" as '<' 'answer' '>D'); the next-token distribution over '>A'..'>J' is read (top-20 raw
logprobs, one forward, no sampling). The model's own thinking text is never included. Ground truth is never in the
prompt; it is used only on the driver to pick b = P(gt letter).

Advantage (lambda = CAPTION_SB_LAMBDA, default 1): stop_relative_advantage.build_advantages (Pro reference code,
sha16 a4f3374f) on DAPO-retained mixed groups, broadcast to the tokens of each real assistant turn; receipt/image tokens
stay 0. Replaces DAPO's group-mean baseline; sigma_q = group sample std of Y (as before).
"""
from __future__ import annotations

import copy
import json
import math
import os
import re

LETTERS = "ABCDEFGHIJ"


def enabled() -> bool:
    return os.environ.get("CAPTION_SB", "0") == "1"


# ---------------- rollout side ----------------
def _question(head: str) -> str:
    return head.split("\nOriginal image")[0].strip()


def build_probe_request(proc, proto, head, images, min_px, max_px, suffix="<answer"):
    """images: already-processed PIL views presented before the turn (original first)."""
    from caption_contracts import grid_list, validate_images, raw_image_prompt
    content = []
    for j, _ in enumerate(images):
        content += [{"type": "text", "text": ("Original image:" if j == 0 else f"Zoomed view {j}:")}, {"type": "image"}]
    content += [{"type": "text", "text": "\n" + _question(head) +
                 "\nAnswer with the option's letter from the given choices directly."}]
    messages = [{"role": "system", "content": proto.SYSTEM}, {"role": "user", "content": content}]
    # training text tokenizes "<answer>D" as '<' 'answer' '>D': stop at "<answer" and read the '>X' token
    text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) + suffix
    enc = proc(images=list(images), text=[text], add_special_tokens=False, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    grid = grid_list(enc["image_grid_thw"])
    validate_images(ids, grid, proc)
    return {"prompt_token_ids": raw_image_prompt(ids, grid, proc), "multi_modal_data": {"image": list(images)}}


def letter_token_ids(tokenizer, prefix=">"):
    """Token ids of prefix+'A'..'J' (single tokens in the Qwen3-VL tokenizer; verified 11-04 for '>X' and 'X')."""
    out = {}
    for L in LETTERS:
        ids = tokenizer.encode(prefix + L, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"'{prefix}{L}' is not a single token; probe read-out would be wrong")
        out[ids[0]] = L
    return out


def combine(lp1, lp2, gt_ids, plain_ids, gt_tok):
    """P(X) = P('>X' | ..<answer) + P('>' | ..<answer) * P('X' | ..<answer>). lp*: {token_id: logprob} top-20."""
    d = {}
    for tid, x in lp1.items():
        L = gt_ids.get(int(tid))
        if L:
            d[L] = d.get(L, 0.0) + math.exp(x)
    p_gt = math.exp(lp1[gt_tok]) if gt_tok in lp1 else 0.0
    for tid, x in lp2.items():
        L = plain_ids.get(int(tid))
        if L:
            d[L] = d.get(L, 0.0) + p_gt * math.exp(x)
    d["_mass"] = sum(v for k, v in d.items() if k != "_mass")
    d["_p_gt_sep"] = p_gt
    return d


def run_probes(engine, base_sampling, proc, proto, states, source_mm, min_px, max_px):
    """Returns per-trajectory list (one per assistant turn) of {letter: prob, '_mass', '_p_gt_sep'}.
    Two requests per distinct state: prefix "<answer" (read '>X' and the bare '>') and "<answer>" (read 'X')."""
    gt_ids = letter_token_ids(proc.tokenizer, ">")
    plain_ids = letter_token_ids(proc.tokenizer, "")
    gt_tok = proc.tokenizer.encode(">", add_special_tokens=False)
    gt_tok = gt_tok[0] if len(gt_tok) == 1 else -1
    reqs, cache, plan = [], {}, []
    for i, st in enumerate(states):
        n_turns = sum(1 for _, a in st["segments"] if a)
        head = source_mm[i]["env"]["head"]
        row = []
        for k in range(max(1, n_turns)):
            imgs = st["images"][:min(k + 1, len(st["images"]))]
            key = (source_mm[i]["images"][0], head, tuple(tuple(b) for b in st["env"].boxes[1:len(imgs)]))
            if key not in cache:
                cache[key] = len(reqs)
                reqs.append(build_probe_request(proc, proto, head, imgs, min_px, max_px, "<answer"))
                reqs.append(build_probe_request(proc, proto, head, imgs, min_px, max_px, "<answer>"))
            row.append(cache[key])
        plan.append(row)
    sp = copy.deepcopy(base_sampling)
    sp.n, sp.max_tokens, sp.temperature, sp.top_p, sp.logprobs, sp.seed = 1, 1, 0.0, 1.0, 20, 0
    outs = engine.generate(prompts=reqs, sampling_params=[sp] * len(reqs), use_tqdm=False) if reqs else []
    lps = [{int(t): float(getattr(x, "logprob", x)) for t, x in (o.outputs[0].logprobs or [{}])[0].items()} for o in outs]
    dists = {j: combine(lps[j], lps[j + 1], gt_ids, plain_ids, gt_tok) for j in range(0, len(lps), 2)}
    return [[dists[j] for j in row] for row in plan], len(reqs)


def turn_spans(segments):
    """[(start, end, turn_id)] in response-token coordinates for assistant segments."""
    spans, pos, t = [], 0, 0
    for toks, is_assistant in segments:
        if is_assistant:
            spans.append((pos, pos + len(toks), t))
            t += 1
        pos += len(toks)
    return spans


# ---------------- driver side ----------------
def _gt_letter(gt):
    if isinstance(gt, dict):
        gt = gt.get("ground_truth", "")
    m = re.search(r"[A-J]", str(gt))
    return m.group(0) if m else ""


def install_advantage():
    """Wrap ray_trainer.compute_advantage: replace advantages with stop-relative ones (install BEFORE update_ledger)."""
    import torch
    import verl.trainer.ray_trainer as RT
    from stop_relative_advantage import build_advantages, broadcast_to_tokens
    if getattr(RT.compute_advantage, "_sb", False):
        return
    original = RT.compute_advantage
    lam = float(os.environ.get("CAPTION_SB_LAMBDA", "1"))
    log_path = os.environ.get("CAPTION_SB_LOG")
    step = {"n": 0}

    def compute_advantage(data, *args, **kwargs):
        data = original(data, *args, **kwargs)
        mm = data.non_tensor_batch["multi_modal_data"]
        uids = [str(u) for u in data.non_tensor_batch["uid"]]
        gts = [_gt_letter(g) for g in data.non_tensor_batch["ground_truth"]]
        scores = data.batch["token_level_rewards"].sum(-1).tolist()
        y = [1 if s >= 1.0 - 1e-6 else 0 for s in scores]
        b = []
        for m, g in zip(mm, gts):
            probe = m.get("caption_probe") or [{}]
            b.append([min(1.0, max(0.0, float(p.get(g, 0.0)))) for p in probe])
        res = build_advantages(y, uids, b, lam=lam, group_size=int(os.environ.get("CAPTION_SB_G", "8")))
        resp_len = data.batch["advantages"].shape[1]
        tok_ids = []
        for i, m in enumerate(mm):
            ids = [-1] * resp_len
            for a, e, t in m.get("caption_turn_spans") or []:
                for p in range(a, min(e, resp_len)):
                    ids[p] = min(t, len(b[i]) - 1)
            tok_ids.append(ids)
        tok_adv = broadcast_to_tokens(res.turn_advantages, tok_ids)
        adv = torch.tensor(tok_adv, dtype=data.batch["advantages"].dtype, device=data.batch["advantages"].device)
        adv = adv * data.batch["response_mask"]
        old = data.batch["advantages"]
        data.batch["advantages"] = adv
        data.batch["returns"] = adv.clone()
        step["n"] += 1
        if log_path:
            groups = {}
            for u, yy, bb in zip(uids, y, b):
                groups.setdefault(u, []).append((yy, bb[0]))
            b0_gap = [sum(x for x, _ in g) / len(g) - g[0][1] for g in groups.values()]
            flat = [x for r in res.raw_turn_advantages for x in r]
            rec = {"call": step["n"], "lam": lam, "n_traj": len(y), "n_groups": len(groups),
                   "mean_b0": sum(r[0] for r in b) / len(b), "mean_acc": sum(y) / len(y),
                   "mean_groupacc_minus_b0": sum(b0_gap) / len(b0_gap),
                   "turns_per_traj": sum(len(r) for r in b) / len(b),
                   "mean_raw_adv": sum(flat) / len(flat), "share_raw_adv_pos": sum(x > 0 for x in flat) / len(flat),
                   "mean_letter_mass": sum(p.get("_mass", 0.0) for m in mm for p in (m.get("caption_probe") or [{}])) /
                   max(1, sum(len(m.get("caption_probe") or [{}]) for m in mm)),
                   "adv_abs_mean_sb": float(adv.abs()[data.batch["response_mask"].bool()].mean()),
                   "adv_abs_mean_dapo": float(old.abs()[data.batch["response_mask"].bool()].mean())}
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[sb] call={rec['call']} b0={rec['mean_b0']:.3f} acc={rec['mean_acc']:.3f} "
                  f"gap={rec['mean_groupacc_minus_b0']:+.3f} mass={rec['mean_letter_mass']:.3f} "
                  f"|A| sb={rec['adv_abs_mean_sb']:.3f} dapo={rec['adv_abs_mean_dapo']:.3f}", flush=True)
        return data

    compute_advantage._sb = True
    RT.compute_advantage = compute_advantage
    print(f"[caption] advantage -> stop-relative (DAPO-SB, lambda={lam})", flush=True)
