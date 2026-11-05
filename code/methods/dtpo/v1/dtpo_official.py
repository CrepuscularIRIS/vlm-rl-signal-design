import torch
from collections import defaultdict
class _Functional:
    @staticmethod
    def masked_mean(values, mask):
        return (values * mask).sum() / mask.sum()
verl_F = _Functional()

def _calculate_group_relative_scores(scores: torch.Tensor, index: torch.Tensor, epsilon: float):
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
    return scores

def compute_DTPO_advantage(
    turn_level_rewards: torch.Tensor,
    outcome_level_rewards: torch.Tensor,
    turn_mask: torch.Tensor,
    eos_mask: torch.Tensor,
    index: torch.Tensor,
    lambda_coeff: float = 0.0,
    epsilon: float = 1e-6):
    """
    Compute advantage for DTPO using turn-level credit assignment.

    Args:
        turn_level_rewards (torch.Tensor): 
            Rewards for the first turn's actions (e.g., tool use). Shape: (bs, response_length)
        outcome_level_rewards (torch.Tensor):
            Rewards for the final outcome of the trajectory. Shape: (bs, response_length)
        turn_mask (torch.Tensor):
            Mask indicating which turn each token belongs to. 
            e.g., 1 for turn 1, 2 for turn 2. Shape: (bs, response_length)
        eos_mask (torch.Tensor):
            Mask to ignore padding tokens. Shape: (bs, response_length)
        index (torch.Tensor):
            Tensor of prompt indices to group responses. Shape: (bs,)
        lambda_coeff (float):
            Coefficient to weight the turn advantage in the first turn.
        epsilon (float):
            Small value for numerical stability in division.

    Returns:
        advantages (torch.Tensor): 
            The final computed advantages for each token. Shape: (bs, response_length)
        returns (torch.Tensor): 
            Same as advantages, to match the common function signature. Shape: (bs, response_length)
    """
    response_length = turn_level_rewards.shape[-1]
    turn_scores = turn_level_rewards.sum(dim=-1)       # (bs,)
    outcome_scores = outcome_level_rewards.sum(dim=-1) # (bs,)
    with torch.no_grad():
        # compute advantage
        adv_outcome = _calculate_group_relative_scores(outcome_scores, index, epsilon) # (bs,)
        adv_turn = _calculate_group_relative_scores(turn_scores, index, epsilon) # (bs,)

        adv_turn_expanded = adv_turn.unsqueeze(-1).expand(-1, response_length) # (bs,) -> (bs, response_length)
        adv_outcome_expanded = adv_outcome.unsqueeze(-1).expand(-1, response_length) # (bs,) -> (bs, response_length)
        
        
        adv_for_turn1 = lambda_coeff * adv_turn_expanded + adv_outcome_expanded

        adv_for_turn2 = adv_outcome_expanded
        
        advantages = torch.zeros_like(turn_level_rewards)
        
        advantages = torch.where(turn_mask == 1, adv_for_turn1, advantages)
        
        advantages = torch.where(turn_mask == 2, adv_for_turn2, advantages)
        
        advantages = advantages * eos_mask

    return advantages, advantages, adv_turn, adv_outcome

def compute_policy_loss(old_log_prob,
                        log_prob,
                        advantages,
                        eos_mask,
                        cliprange=None,
                        cliprange_low=None,
                        cliprange_high=None,
                        turn_mask=None,
                        control_level=None,
                        control_alpha=1):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_low: (float)
            The lower clip range used in PPO.
        cliprange_high: (float)
            The higher clip range used in PPO.
    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            the fraction of policy gradient loss being clipped
        ppo_kl: (float)
            the estimated KL divergence between the latest updating policy and the old sampling policy
    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    # pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange_low, 1.0 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    token_loss = torch.max(pg_losses, pg_losses2)
    
    if control_level == 'tool_level':
        if turn_mask is None:
            pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
        else:
            max_turn_per_sequence, _ = torch.max(turn_mask, dim=1, keepdim=True)
            is_valid_token = (turn_mask > 0)
            # is_last_turn_mask: (bs, response_length)
            is_last_turn_mask = is_valid_token & (turn_mask == max_turn_per_sequence)

            # is_intermediate_turn_mask (bs, response_length)
            is_intermediate_turn_mask = is_valid_token & (turn_mask < max_turn_per_sequence)

            if (is_intermediate_turn_mask * eos_mask).sum() == 0:
                intermediate_turn_loss = torch.tensor(0.0)
            else:
                intermediate_turn_loss = verl_F.masked_mean(token_loss, is_intermediate_turn_mask * eos_mask)

            last_turn_loss = verl_F.masked_mean(token_loss, is_last_turn_mask * eos_mask)

            pg_loss = control_alpha * intermediate_turn_loss + last_turn_loss
    else:
        pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl
