"""CPU-testable advantage core for a proposed Stop-Relative DAPO experiment.

This is NOT an integrated trainer. It does not score images or run rollouts.
Inputs must be produced from the frozen behavior checkpoint, BEFORE each
assistant turn. Ground-truth answers must never be inserted into actor inputs.

Version 1 deliberately accepts only complete mixed-outcome groups, preserving
DAPO's dynamic-sampling criterion. It keeps the sample-standard-deviation
scaling used in the supplied repository. It does not claim an unbiased
estimator after outcome filtering, sample normalization, or PPO clipping.

lambda=1: empirical outcome minus a pre-turn stopping baseline.
lambda=0: local potential differences WITH the terminal residual.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class AdvantageResult:
    raw_turn_advantages: tuple[tuple[float, ...], ...]
    turn_advantages: tuple[tuple[float, ...], ...]
    td_residuals: tuple[tuple[float, ...], ...]
    group_sample_stds: tuple[float, ...]


def build_advantages(
    outcomes: Sequence[int],
    group_ids: Sequence[str],
    stop_probabilities: Sequence[Sequence[float]],
    *,
    lam: float = 1.0,
    group_size: int = 8,
    eps: float = 1e-6,
) -> AdvantageResult:
    """Compute per-turn advantages; every row includes its last assistant turn.

    outcomes: original legal-and-correct binary task reward.
    group_ids: a unique ID per GENERATED GROUP, not just a repeated question ID.
    stop_probabilities[i][k]: frozen probe baseline BEFORE assistant turn k.
    Last turn is terminal, including invalid/truncated endings (outcome=0).
    Tool/observation tokens are not model actions and receive no advantage.
    No gradient, whitening, clipping, or positive-only gating is applied here.
    """
    n = len(outcomes)
    if n == 0 or len(group_ids) != n or len(stop_probabilities) != n:
        raise ValueError("Inputs must have the same nonzero trajectory count.")
    if not math.isfinite(lam) or not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be finite and lie in [0, 1].")
    if group_size < 2 or not math.isfinite(eps) or eps <= 0:
        raise ValueError("group_size must be >=2 and eps must be positive.")
    groups: dict[str, list[int]] = defaultdict(list)
    for i, (y, group) in enumerate(zip(outcomes, group_ids)):
        if y not in (0, 1):
            raise ValueError("Outcomes must be the unchanged binary reward.")
        if not isinstance(group, str):
            raise TypeError("group_ids must be strings.")
        groups[group].append(i)
    stds = [0.0] * n
    for indices in groups.values():
        if len(indices) != group_size:
            raise ValueError("Use intact generated groups before minibatch splitting.")
        mean = sum(outcomes[i] for i in indices) / group_size
        if not 0.0 < mean < 1.0:
            raise ValueError("Version 1 requires DAPO-retained mixed groups.")
        std = math.sqrt(
            sum((outcomes[i] - mean) ** 2 for i in indices) / (group_size - 1)
        )
        for i in indices:
            stds[i] = std
    raw_rows, scaled_rows, delta_rows = [], [], []
    for i, probs in enumerate(stop_probabilities):
        b = tuple(float(x) for x in probs)
        if not b or any(not math.isfinite(x) or not 0 <= x <= 1 for x in b):
            raise ValueError("Every turn needs a finite pre-turn probability in [0,1].")
        # Absorbing terminal-state potential is zero; terminal task reward is Y.
        delta = [b[k + 1] - b[k] for k in range(len(b) - 1)]
        delta.append(float(outcomes[i]) - b[-1])
        raw = [0.0] * len(b)
        running = 0.0
        for k in range(len(b) - 1, -1, -1):
            running = delta[k] + lam * running
            raw[k] = running
        raw_rows.append(tuple(raw))
        scaled_rows.append(tuple(x / (stds[i] + eps) for x in raw))
        delta_rows.append(tuple(delta))
    return AdvantageResult(
        tuple(raw_rows), tuple(scaled_rows), tuple(delta_rows), tuple(stds)
    )


def broadcast_to_tokens(
    turn_advantages: Sequence[Sequence[float]],
    token_turn_ids: Sequence[Sequence[int]],
) -> tuple[tuple[float, ...], ...]:
    """Map advantages to real assistant tokens; -1 masks all non-action tokens.

    Turn IDs must come from the actual rollout serializer. Do not infer them
    from model-written 'Step N' text. Do not include tool returns or probe text.
    """
    if len(turn_advantages) != len(token_turn_ids):
        raise ValueError("Trajectory counts do not match.")
    rows = []
    for values, ids in zip(turn_advantages, token_turn_ids):
        row = []
        for k in ids:
            if not isinstance(k, int):
                raise TypeError("Turn IDs must be integers.")
            if k == -1:
                row.append(0.0)
            elif 0 <= k < len(values):
                row.append(float(values[k]))
            else:
                raise ValueError("Token turn ID is out of range.")
        rows.append(tuple(row))
    return tuple(rows)


def _self_test() -> None:
    import random
    rng = random.Random(42)
    y = [1, 0, 1, 0, 1, 0, 1, 0]
    groups = ["batch0/group0"] * 8
    probs = [[rng.random() for _ in range(rng.randint(1, 9))] for _ in y]
    r = build_advantages(y, groups, probs, lam=1)
    for yi, bi, ai, di in zip(y, probs, r.raw_turn_advantages, r.td_residuals):
        assert math.isclose(sum(di), yi - bi[0], abs_tol=1e-12)
        for b, a in zip(bi, ai):
            assert math.isclose(a, yi - b, abs_tol=1e-12)
    r0 = build_advantages(y, groups, probs, lam=0)
    assert r0.raw_turn_advantages == r0.td_residuals
    assert math.isclose(r.group_sample_stds[0], math.sqrt(2 / 7), abs_tol=1e-12)
    # Four desired limiting directions, before the shared positive scale.
    four = [1-.05, 1-.95, 0-.05, 0-.95]
    assert four[0] > four[1] > 0 > four[2] > four[3]
    masked = broadcast_to_tokens([[.5, -.2]], [[-1, 0, 0, -1, 1]])
    assert masked == ((0.0, .5, .5, 0.0, -.2),)
    try:
        build_advantages([0] * 8, groups, [[.1]] * 8)
    except ValueError:
        pass
    else:
        raise AssertionError("Homogeneous group was incorrectly accepted.")
    print("PASS: lambda=1 identity; telescoping with terminal residual; lambda=0;")
    print("sample std; four limiting directions; token masking; homogeneous guard.")
    p = [.1, .1, .9, .9]
    print("Toy path (not empirical data): pre-turn probabilities =", p)
    for yi in (1, 0):
        yi_rows = [yi, 1-yi] * 4
        for lam in (0., .5, 1.):
            out = build_advantages(yi_rows, groups, [p] * 8, lam=lam)
            print(f"Y={yi}, lambda={lam}: raw A={out.raw_turn_advantages[0]}")


if __name__ == "__main__":
    _self_test()
