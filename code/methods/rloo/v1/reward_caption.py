"""Terminal reward R = Y * (1 - lambda * N) (mentor 10-16 form; lambda from CAPTION_CROP_LAMBDA, required).

Y = exactly one legal <answer> A-J letter equal to gold (same rule as grpo_v52); N = executed crops. No format bonus,
abstention bonus or process reward. Input must contain generated assistant tokens, not observation receipts.
"""
import os
import re

REWARD_NAME = "caption_correctness_cropcost"
REWARD_TYPE = "batch"
_ANS = re.compile(r"<answer>\s*([^<>]*?)\s*</answer>", re.S)
_LETTER = re.compile(r"[A-J]")


def crop_lambda() -> float:
    return float(os.environ["CAPTION_CROP_LAMBDA"])


def _norm_answer(text: str) -> str | None:
    matches = _ANS.findall(text)
    # A trajectory terminates on its first answer; multiple answers are invalid.
    if len(matches) != 1:
        return None
    answer = matches[0].strip().upper()
    return answer if _LETTER.fullmatch(answer) else None


def correct_one(solution_str: str, ground_truth: str) -> int:
    gold = str(ground_truth).strip().upper()
    return int(bool(_LETTER.fullmatch(gold)) and _norm_answer(solution_str) == gold)


def compute_score(reward_inputs: list[dict]) -> list[float]:
    lam = crop_lambda()
    rewards = []
    for item in reward_inputs:
        gold = item.get("ground_truth", item.get("reward_model", {}))
        if isinstance(gold, dict):
            gold = gold.get("ground_truth", "")
        y = correct_one(item.get("response", ""), str(gold))
        rewards.append(y * (1.0 - lam * int(item["crops"])))
    return rewards
