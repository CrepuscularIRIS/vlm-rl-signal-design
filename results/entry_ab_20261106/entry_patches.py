"""Update-entry variants on GRPO (2026-11-05): change only WHICH groups enter the update.

Reward (V*Y), advantage (group z-score, sample std), n=8, 128 groups and 4 actor updates per step are unchanged.

CAPTION_ENTRY=replay  zero-variance groups of this step are replaced by mixed groups kept from the previous step
                      (age 1, each replayed at most once, generation order). fit() recomputes their old_log_probs
                      under the current policy and their advantages from the stored rewards.
CAPTION_ENTRY=pilot   after the first round (128 questions x 8), every mixed question gets one more independent group
                      of 8 (new uid, own z-score). Update = first-round mixed + second-round mixed (capped at 128),
                      padded to 128 with first-round zero-variance groups (zero advantage, no gradient).
unset                 stock GRPO path; install() changes nothing.

Per-step facts are appended to CAPTION_ENTRY_LOG (jsonl).
"""
from __future__ import annotations

import copy
import json
import os
import uuid
from collections import defaultdict

import numpy as np

MODES = ("replay", "pilot")
MIXED_LOW, MIXED_HIGH = 0.01, 0.99  # same bounds as the DAPO filter on 0/1 rewards


def unique_in_order(uids) -> list:
    """Group ids in first-appearance order."""
    seen, order = set(), []
    for u in uids:
        if u not in seen:
            seen.add(u)
            order.append(u)
    return order


def mixed_uids(uids, scores) -> list:
    """Groups whose mean reward is strictly between the bounds, in first-appearance order."""
    by = defaultdict(list)
    for u, s in zip(uids, scores):
        by[u].append(float(s))
    return [u for u in unique_in_order(uids) if MIXED_LOW < np.mean(by[u]) < MIXED_HIGH]


def plan_replay(fresh_order: list, fresh_mixed: list, pool_order: list, target: int) -> tuple[list, list]:
    """Keep all fresh groups except the last r zero-variance ones; append r pool groups. Total stays target."""
    mixed = set(fresh_mixed)
    zero = [u for u in fresh_order if u not in mixed]
    r = max(0, min(len(zero), len(pool_order), target - len(fresh_mixed)))
    drop = set(zero[len(zero) - r:]) if r else set()
    keep = [u for u in fresh_order if u not in drop]
    return keep, pool_order[:r]


def plan_pilot(first_order: list, first_mixed: list, second_mixed: list, target: int) -> tuple[list, list]:
    """First-round mixed + second-round mixed up to target, padded with first-round zero-variance groups."""
    assert len(first_mixed) <= target
    take_second = second_mixed[:target - len(first_mixed)]
    n_pad = target - len(first_mixed) - len(take_second)
    mixed = set(first_mixed)
    pad = set([u for u in first_order if u not in mixed][:n_pad])
    keep_first = [u for u in first_order if u in mixed or u in pad]
    assert len(keep_first) + len(take_second) == target, (len(keep_first), len(take_second), target)
    return keep_first, take_second


def select(batch, uid_list: list):
    """Rows of the given groups, in uid_list order."""
    pos = defaultdict(list)
    for i, u in enumerate(batch.non_tensor_batch["uid"]):
        pos[u].append(i)
    return batch[[i for u in uid_list for i in pos[u]]]


def _log(record: dict) -> None:
    path = os.environ.get("CAPTION_ENTRY_LOG")
    if path:
        with open(path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    print("[entry] " + json.dumps(record), flush=True)


def _score(trainer, batch, metrics: dict, prefix: str) -> list:
    """One reward call; stores token_level_scores so fit() does not score again."""
    import ray
    import verl.trainer.ray_trainer as RT
    reward_tensor, reward_metrics = ray.get(trainer.reward_fn.compute_reward.remote(batch))
    batch.batch["token_level_scores"] = reward_tensor
    metrics.update({f"{prefix}/{k}": v for k, v in RT.reduce_metrics(reward_metrics).items()})
    return reward_tensor.sum(-1).tolist()


def _generate_round(trainer, n: int, target: int, keep_prompts: bool):
    """Stock GRPO generation: 32-question chunks until target groups. Returns batch and prompt-side copies."""
    from verl.protocol import DataProto
    chunks, prompt_sides, gen_sides = [], [], []
    meta_info = {"min_pixels": trainer.config.data.min_pixels, "max_pixels": trainer.config.data.max_pixels,
                 "video_fps": trainer.config.data.video_fps}
    while sum(len(c) for c in chunks) // n < target:
        try:
            batch_dict = next(trainer.data_iterator)
        except StopIteration:
            trainer.data_iterator = iter(trainer.train_dataloader)
            batch_dict = next(trainer.data_iterator)
        new_batch = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
        new_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))],
                                                     dtype=object)
        gen_batch = new_batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"],
                                  non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                                  meta_info_keys=["min_pixels", "max_pixels", "video_fps"])
        if keep_prompts:
            prompt_sides.append(copy.deepcopy(new_batch))
            gen_sides.append(copy.deepcopy(gen_batch))
        output = trainer.actor_rollout_ref_wg.generate_sequences(gen_batch)
        chunks.append(new_batch.repeat(repeat_times=n, interleave=True).union(output))
    batch = DataProto.concat(chunks)
    return batch[: target * n], prompt_sides, gen_sides


def make_batch_data(self, metrics: dict):
    """Replacement for RayPPOTrainer._make_batch_data under CAPTION_ENTRY."""
    from verl.protocol import DataProto
    mode = os.environ["CAPTION_ENTRY"]
    assert not self.config.algorithm.online_filtering, "entry variants run on the GRPO path (no online filtering)"
    n, target = self.config.worker.rollout.n, self.config.data.rollout_batch_size
    batch, prompt_sides, gen_sides = _generate_round(self, n, target, keep_prompts=(mode == "pilot"))
    uids = list(batch.non_tensor_batch["uid"])
    order = unique_in_order(uids)
    assert len(order) == target and len(batch) == target * n
    fresh_mixed = mixed_uids(uids, _score(self, batch, metrics, "reward"))
    record = {"step": self.global_step, "mode": mode, "fresh_groups": len(order), "fresh_mixed": len(fresh_mixed)}

    if mode == "replay":
        pool = getattr(self, "_entry_pool", None)
        pool_order = unique_in_order(pool.non_tensor_batch["uid"]) if pool is not None else []
        keep, replay = plan_replay(order, fresh_mixed, pool_order, target)
        out = select(batch, keep)
        if replay:
            out = DataProto.concat([out, select(pool, replay)])
        self._entry_pool = select(batch, fresh_mixed) if fresh_mixed else None
        record.update(pool=len(pool_order), replayed=len(replay), reward_calls=1)
    else:
        prompts, gens = DataProto.concat(prompt_sides), DataProto.concat(gen_sides)
        mixed = set(fresh_mixed)
        idx = [i for i, u in enumerate(prompts.non_tensor_batch["uid"]) if u in mixed]
        second_mixed, second, record_pad = [], None, 0
        if idx:
            p2, g2 = prompts[idx], gens[idx]
            p2.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in idx], dtype=object)
            # dispatch needs a multiple of the worker count; padded prompts are generated and then dropped
            from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
            g2_padded, pad = pad_dataproto_to_divisor(g2, self.actor_rollout_ref_wg.world_size)
            out2 = unpad_dataproto(self.actor_rollout_ref_wg.generate_sequences(g2_padded), pad_size=pad * n)
            assert len(out2) == len(idx) * n, (len(out2), len(idx), n)
            record_pad = pad
            second = p2.repeat(repeat_times=n, interleave=True).union(out2)
            second_mixed = mixed_uids(list(second.non_tensor_batch["uid"]),
                                      _score(self, second, metrics, "entry_second"))
        keep_first, take_second = plan_pilot(order, fresh_mixed, second_mixed, target)
        out = select(batch, keep_first)
        if take_second:
            out = DataProto.concat([out, select(second, take_second)])
        record.update(second_questions=len(idx), second_pad_dropped=record_pad,
                      second_mixed=len(second_mixed), second_taken=len(take_second),
                      padded_zero=target - len(fresh_mixed) - len(take_second), reward_calls=1 + bool(idx))

    out_order = unique_in_order(out.non_tensor_batch["uid"])
    assert len(out_order) == target and len(out) == target * n, (len(out_order), len(out))
    record["update_groups"] = len(out_order)
    record["update_mixed"] = len(mixed_uids(list(out.non_tensor_batch["uid"]),
                                            out.batch["token_level_scores"].sum(-1).tolist()))
    metrics.update({f"entry/{k}": v for k, v in record.items() if isinstance(v, (int, float)) and k != "step"})
    _log(record)
    return out


def install() -> None:
    mode = os.environ.get("CAPTION_ENTRY", "")
    if not mode:
        return
    if mode not in MODES:
        raise ValueError(f"CAPTION_ENTRY must be one of {MODES}, got {mode!r}")
    from verl.trainer.ray_trainer import RayPPOTrainer
    if getattr(RayPPOTrainer._make_batch_data, "_entry", None) == mode:
        return
    make_batch_data._entry = mode
    RayPPOTrainer._make_batch_data = make_batch_data
    print(f"[entry] installed mode={mode}", flush=True)


if __name__ == "__main__":  # CPU self-check of the planning rules
    order = [f"g{i}" for i in range(8)]
    keep, rep = plan_replay(order, ["g0", "g2"], ["p0", "p1", "p2"], 8)
    assert keep == ["g0", "g1", "g2", "g3", "g4"] and rep == ["p0", "p1", "p2"]           # 3 zero groups replaced
    keep, rep = plan_replay(order, ["g0"], [], 8)
    assert keep == order and rep == []                                                    # step 1: nothing to replay
    keep, rep = plan_replay(order, order, ["p0"], 8)
    assert keep == order and rep == []                                                    # all mixed: no room
    k1, k2 = plan_pilot(order, ["g1", "g5"], ["s0", "s1", "s2"], 8)
    assert k2 == ["s0", "s1", "s2"] and k1 == ["g0", "g1", "g2", "g3", "g5"]              # pad 3 zero groups
    k1, k2 = plan_pilot(order, order[:6], ["s0", "s1", "s2"], 8)
    assert k1 == order[:6] and k2 == ["s0", "s1"]                                         # cap at target
    assert mixed_uids(["a"] * 2 + ["b"] * 2 + ["c"] * 2, [1, 0, 1, 1, 0, 0]) == ["a"]
    print("entry_patches selfcheck ok")
