# Apache-2.0; adapted by extraction from dy-liang/SAPO commit 2217a459b948e5fd339c44cba0158c0367e2fb6e
# Original Copyright 2026. Functions below are unchanged. See OFFICIAL_LICENSE.
from __future__ import annotations
from collections import defaultdict
from typing import Sequence
import torch

def compute_arq_gae_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    old_v_values: torch.Tensor,
    old_q_values: torch.Tensor,
    response_mask: torch.Tensor,
    trajectory_ids: Sequence[object],
    turn_indices: torch.Tensor,
    gamma: float,
    lam: float,
    unique_turn_mask: torch.Tensor | None = None,
    invalid_action_mask: torch.Tensor | None = None,
    invalid_action_penalty_coef: float = 0.0,
    normalize: bool = True,
    epsilon: float = 1e-6,
    q_target_mode: str = "td0",
    q_lambda: float | None = None,
    advantage_mode: str = "gae",
    q_advantage_mix_coef: float = 0.2,
) -> dict[str, torch.Tensor]:
    """Compute turn-level V/Q targets and the configured policy advantage.
    """
    batch_size = rewards.shape[0]
    resolved_q_lambda = lam if q_lambda is None else q_lambda

    with torch.no_grad():
        rewards = rewards.detach().to(torch.float32)
        dones = dones.detach().to(torch.bool)
        old_v_values = old_v_values.detach().to(torch.float32)
        old_q_values = old_q_values.detach().to(torch.float32)
        turn_indices = turn_indices.detach().to(torch.long)
        if unique_turn_mask is not None:
            unique_turn_mask = unique_turn_mask.detach().to(device=rewards.device, dtype=torch.bool)

        for name, tensor in {
            "rewards": rewards,
            "old_v_values": old_v_values,
            "old_q_values": old_q_values,
        }.items():
            if not torch.all(torch.isfinite(tensor)):
                raise ValueError(f"ARQ {name} contains NaN or Inf")

        key_to_rows: dict[tuple[str, int], list[int]] = defaultdict(list)
        trajectory_to_keys: dict[str, list[tuple[int, tuple[str, int]]]] = defaultdict(list)
        for row, (trajectory_id, turn_index) in enumerate(zip(trajectory_ids, turn_indices.tolist())):
            key_to_rows[(str(trajectory_id), int(turn_index))].append(row)

        canonical_rows = []
        key_to_canonical: dict[tuple[str, int], int] = {}
        for key, rows in key_to_rows.items():
            if unique_turn_mask is None:
                canonical_row = rows[0]
            else:
                canonical_candidates = [row for row in rows if bool(unique_turn_mask[row].item())]
                canonical_row = canonical_candidates[0]
            key_to_canonical[key] = canonical_row
            canonical_rows.append(canonical_row)
            trajectory_to_keys[key[0]].append((key[1], key))

        gae_advantages = torch.zeros_like(rewards)
        td_errors = torch.zeros_like(rewards)
        v_targets = torch.zeros_like(rewards)
        q_targets = torch.zeros_like(rewards)
        truncated = torch.zeros_like(dones)

        for trajectory_id, indexed_keys in trajectory_to_keys.items():
            indexed_keys.sort(key=lambda item: item[0])

            next_gae = torch.zeros((), dtype=torch.float32, device=rewards.device)
            next_q_return = torch.zeros((), dtype=torch.float32, device=rewards.device)
            for position in range(len(indexed_keys) - 1, -1, -1):
                _, key = indexed_keys[position]
                row = key_to_canonical[key]
                has_successor = position + 1 < len(indexed_keys)
                is_done = bool(dones[row].item())
                if has_successor:
                    next_key = indexed_keys[position + 1][1]
                    next_row = key_to_canonical[next_key]
                    next_value = old_v_values[next_row]
                    continuation = 1.0
                else:
                    next_value = torch.zeros((), dtype=torch.float32, device=rewards.device)
                    continuation = 0.0
                    truncated[row] = not is_done

                td0_target = rewards[row] + gamma * continuation * next_value
                td_error = td0_target - old_v_values[row]
                gae = td_error + gamma * lam * continuation * next_gae
                v_target = old_v_values[row] + gae

                if q_target_mode == "td0":
                    q_target = td0_target
                elif q_target_mode == "lambda_return":
                    q_target = rewards[row] + gamma * continuation * (
                        (1.0 - resolved_q_lambda) * next_value
                        + resolved_q_lambda * next_q_return
                    )
                elif q_target_mode == "monte_carlo":
                    q_target = rewards[row] + gamma * continuation * next_q_return
                else:  # sarsa_td0
                    next_q_value = (
                        old_q_values[next_row]
                        if has_successor
                        else torch.zeros((), dtype=torch.float32, device=rewards.device)
                    )
                    q_target = rewards[row] + gamma * continuation * next_q_value

                q_targets[row] = q_target
                td_errors[row] = td_error
                gae_advantages[row] = gae
                v_targets[row] = v_target
                next_gae = gae
                if q_target_mode in {"lambda_return", "monte_carlo"}:
                    next_q_return = q_target

        resolved_unique_turn_mask = torch.zeros(batch_size, dtype=torch.bool, device=rewards.device)
        resolved_unique_turn_mask[
            torch.as_tensor(canonical_rows, dtype=torch.long, device=rewards.device)
        ] = True

        # Map canonical results back onto padding duplicates.
        for key, rows in key_to_rows.items():
            canonical_row = key_to_canonical[key]
            for duplicate_row in rows:
                if duplicate_row == canonical_row:
                    continue
                gae_advantages[duplicate_row] = gae_advantages[canonical_row]
                td_errors[duplicate_row] = td_errors[canonical_row]
                v_targets[duplicate_row] = v_targets[canonical_row]
                q_targets[duplicate_row] = q_targets[canonical_row]
                truncated[duplicate_row] = truncated[canonical_row]

        q_advantages = old_q_values - old_v_values
        if advantage_mode == "gae":
            raw_policy_advantage = gae_advantages.clone()
        elif advantage_mode == "qv":
            raw_policy_advantage = q_advantages.clone()
        else:  # gae_qv_mix
            raw_policy_advantage = (
                (1.0 - q_advantage_mix_coef) * gae_advantages
                + q_advantage_mix_coef * q_advantages
            )

        if invalid_action_mask is not None:
            invalid_action_mask = invalid_action_mask.detach().to(
                device=raw_policy_advantage.device, dtype=raw_policy_advantage.dtype
            )
            raw_policy_advantage = (
                raw_policy_advantage - invalid_action_penalty_coef * invalid_action_mask
            )

        normalized_advantage = raw_policy_advantage
        if normalize:
            stats_values = raw_policy_advantage[resolved_unique_turn_mask]
            mean = stats_values.mean()
            variance = stats_values.var(unbiased=False)
            normalized_advantage = (raw_policy_advantage - mean) * torch.rsqrt(variance + epsilon)

        token_advantage = normalized_advantage.unsqueeze(-1) * response_mask
        token_returns = v_targets.unsqueeze(-1) * response_mask

    return {
        "advantages": token_advantage,
        "returns": token_returns,
        "turn_advantages": normalized_advantage,
        "raw_turn_advantages": raw_policy_advantage,
        "gae_advantages": gae_advantages,
        "td_errors": td_errors,
        "v_targets": v_targets,
        "q_targets": q_targets,
        "q_advantages": q_advantages,
        "unique_turn_mask": resolved_unique_turn_mask,
        "truncated": truncated,
    }

def compute_arq_value_loss(
    current_probs: torch.Tensor,
    old_probs: torch.Tensor,
    target_probs: torch.Tensor,
    cliprange: float,
    sample_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn-level PPO-style clipped regression loss for bounded V or Q.

    ``sample_mask`` excludes shape-only duplicates inserted by ``adjust_batch``
    without changing the physical batch size required by distributed workers.
    """
    old_probs = old_probs.detach().to(torch.float32)
    target_probs = target_probs.detach().to(torch.float32)
    current_probs = current_probs.to(torch.float32)
    clipped_probs = old_probs + torch.clamp(current_probs - old_probs, -cliprange, cliprange)
    loss_unclipped = (current_probs - target_probs).square()
    loss_clipped = (clipped_probs - target_probs).square()
    loss_per_turn = 0.5 * torch.maximum(loss_unclipped, loss_clipped)
    clipped = (torch.abs(current_probs - old_probs) > cliprange).to(torch.float32)
    if sample_mask is None:
        loss = loss_per_turn.mean()
        clip_fraction = clipped.mean()
    else:
        if sample_mask.shape != current_probs.shape:
            raise ValueError("ARQ value sample_mask must have one value per turn")
        sample_mask = sample_mask.to(device=current_probs.device, dtype=torch.float32)
        denominator = sample_mask.sum().clamp_min(1.0)
        loss = (loss_per_turn * sample_mask).sum() / denominator
        clip_fraction = (clipped * sample_mask).sum() / denominator
    return loss, clip_fraction

def compute_arq_q_loss(
    current_q_probs: torch.Tensor,
    current_v_probs: torch.Tensor,
    old_q_probs: torch.Tensor,
    old_v_probs: torch.Tensor,
    q_target_probs: torch.Tensor,
    q_advantage_target_probs: torch.Tensor,
    cliprange: float,
    loss_mode: str = "absolute",
    sample_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute absolute-Q or action-residual Q regression.

    Residual mode supervises ``Q(s, a) - stopgrad(V(s))``.  It therefore asks
    the post-action readout to explain the selected target's increment over the
    pre-action baseline without sending a direct Q-loss gradient through the V
    output.  The shared backbone can still transfer information indirectly.
    """
    if loss_mode == "absolute":
        return compute_arq_value_loss(
            current_probs=current_q_probs,
            old_probs=old_q_probs,
            target_probs=q_target_probs,
            cliprange=cliprange,
            sample_mask=sample_mask,
        )
    if loss_mode == "advantage_residual":
        return compute_arq_value_loss(
            current_probs=current_q_probs - current_v_probs.detach(),
            old_probs=old_q_probs - old_v_probs,
            target_probs=q_advantage_target_probs,
            cliprange=cliprange,
            sample_mask=sample_mask,
        )
    raise ValueError(
        f"Unsupported ARQ q_loss_mode {loss_mode!r}; "
        "expected 'absolute' or 'advantage_residual'"
    )
