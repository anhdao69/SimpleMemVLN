"""ActiveVLN GRPO components, ported from arvillion/ActiveVLN @ 3a0c63b.

References (paths relative to the ActiveVLN checkout):
- reward: vlnce_server/env.py L395-404, reward_type ``weighted_success_ndtw`` with
  success_reward_base=15, ndtw_reward_base=0;
- advantage: verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage (L219-273);
- policy loss: core_algos.py::compute_policy_loss (L616-688), clip 0.2/0.28, dual clip 3.0;
- aggregation: dp_actor.py::update_policy (L499-541), token-mean per micro-batch of two
  trajectories, divided by the number of micro-batches in the mini-batch;
- budget: ray_trainer.py::compute_episode_step_drop_and_limit, min(2 * len(gt), 120).
"""
import math

import torch

SUCCESS_DISTANCE = 3.0


def episode_reward(*, success, distance, success_reward_base=15.0, ndtw_reward_base=0.0, ndtw=None):
    """R = 15 * 1[STOP within 3 m] * (1 - d/3); budget exhaustion/format error give 0."""
    reward = 0.0
    if success:
        if not distance < SUCCESS_DISTANCE:
            raise ValueError('Success requires distance < 3 m')
        reward += success_reward_base * (1.0 - distance / SUCCESS_DISTANCE)
    if ndtw_reward_base:
        reward += ndtw_reward_base * float(ndtw)
    return reward


def activevln_budget(n_star, macro_len, factor=2.0, cap=120):
    """ActiveVLN's step budget min(factor*len(gt), cap) counts merged macro actions.

    SimpleMem emits one 0.25 m / 15-degree primitive per step, so the macro budget
    is converted with the episode's own primitives-per-macro ratio N*/M (exact,
    because ActiveVLN's GT expands to exactly our primitive GT for all 4000 episodes).
    """
    if n_star <= 0 or macro_len <= 0:
        raise ValueError('Expert lengths must be positive')
    return int(math.ceil(min(factor * macro_len, cap) * n_star / macro_len))


def grpo_advantages(rewards, epsilon=1e-6):
    """(R - mean) / (unbiased std + eps) within one group; a singleton group gets 0."""
    scores = torch.as_tensor(rewards, dtype=torch.float32)
    if scores.numel() == 1:
        return torch.zeros_like(scores)
    return (scores - scores.mean()) / (scores.std(unbiased=True) + epsilon)


def policy_loss_terms(logp, old_logp, advantage, clip_low=0.2, clip_high=0.28, clip_c=3.0):
    """Per-token PPO dual-clip loss as in verl compute_policy_loss; returns (loss, clipped flags)."""
    negative_approx_kl = torch.clamp(logp - old_logp, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    losses1 = -advantage * ratio
    losses2 = -advantage * torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
    clip1 = torch.maximum(losses1, losses2)
    losses3 = -advantage * clip_c
    clip2 = torch.minimum(losses3, clip1)
    loss = torch.where(advantage < 0, clip2, clip1)
    clipped = (losses2 > losses1) | ((advantage < 0) & (losses3 < clip1))
    return loss, clipped


def micro_batches(trajectory_keys, size=2):
    """Group trajectories into verl-style micro-batches of ``size`` in batch order."""
    keys = list(trajectory_keys)
    return [keys[i:i + size] for i in range(0, len(keys), size)]


def loss_weights(token_counts, micro_batch_ids, world_size, num_micro_batches_global):
    """Coefficient of each trajectory's summed token loss on its own rank.

    verl: loss_mb = sum_tokens(l) / tokens(mb) / grad_accum on each of G GPUs, gradients
    averaged over GPUs => total = (1/num_mb_global) * sum_mb sum_tokens(l) / tokens(mb).
    DeepSpeed also averages over ranks, so each local term is scaled by world_size.
    """
    per_mb = {}
    for count, mb in zip(token_counts, micro_batch_ids):
        per_mb[mb] = per_mb.get(mb, 0) + count
    return [world_size / (num_micro_batches_global * per_mb[mb]) for mb in micro_batch_ids]
