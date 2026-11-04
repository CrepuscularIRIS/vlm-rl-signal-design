"""Protocol v5.2i = v5.1 with ONE rule change: Step 1's WORD records an intent only (user 10-27, both arms).

v5.1 binds the header WORD to the action (locate/re_examine/fallback -> <grounding>, commit/abstain -> <answer>,
abstain -> UNCLEAR). Under that binding the post-caption prefix h cannot legally take both actions, so the
answer-vs-crop fork at h would measure format legality, not task value. For expected_step == 1 only, the three
WORD-derived errors are dropped; the real action is the final <answer> or <grounding>. Later steps keep v5.1 exactly.

Unlike the 10-22 v52_dl version, SYSTEM states the rule (user 10-27: "同步明确 SYSTEM 规则"): one sentence is appended
to the behaviour paragraph. Datasets still carry the v5.1 SYSTEM; the adapter checks that and encodes this SYSTEM.
"""
from __future__ import annotations

import caption_v51 as _v51
from caption_v51 import *  # noqa: F401,F403  (parse_step, receipt_text, HINTS, ...)
from caption_v51 import MAX_CROPS
from caption_v51 import check_step as _check_step_v51

PROTOCOL = "v5.2i"
SYSTEM_V51 = _v51.SYSTEM
_ANCHOR = "Grounding uses locate, re_examine, or fallback. Answering uses commit or abstain. Geometry alone does not determine a behavior."
STEP1_RULE = " At Step 1 only, WORD records your initial intent; the Step 1 action may still be either <grounding> or <answer>."
if SYSTEM_V51.count(_ANCHOR) != 1:
    raise RuntimeError("v5.1 SYSTEM anchor sentence not found exactly once")
SYSTEM = SYSTEM_V51.replace(_ANCHOR, _ANCHOR + STEP1_RULE)
STEP1_RELAXED = ("behavior_requires_grounding", "behavior_requires_answer", "abstain_without_UNCLEAR")


def check_step(p: dict, expected_step: int, crops_used: int, available: list[str], budget: int = MAX_CROPS) -> list[str]:
    errors = _check_step_v51(p, expected_step, crops_used, available, budget)
    if expected_step != 1:
        return errors
    return [e for e in errors if e not in STEP1_RELAXED]
