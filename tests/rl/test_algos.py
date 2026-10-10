import math

import pytest
import torch

from qwen_vl.rl.algos import (activevln_budget, episode_reward, grpo_advantages, loss_weights,
                              micro_batches, policy_loss_terms)


def test_reward_matches_activevln_weighted_success():
    assert episode_reward(success=True, distance=0.0) == 15.0
    assert episode_reward(success=True, distance=1.5) == pytest.approx(7.5)
    assert episode_reward(success=True, distance=2.999) == pytest.approx(15 * (1 - 2.999 / 3))
    assert episode_reward(success=False, distance=0.5) == 0.0  # budget/format failure
    assert episode_reward(success=False, distance=3.0) == 0.0
    with pytest.raises(ValueError):
        episode_reward(success=True, distance=3.0)  # Habitat success is strict d < 3


def test_grpo_advantage_unbiased_std_and_zero_variance():
    rewards = [15.0, 0.0, 7.5, 0.0]
    t = torch.tensor(rewards)
    expected = (t - t.mean()) / (t.std(unbiased=True) + 1e-6)
    assert torch.allclose(grpo_advantages(rewards), expected)
    assert torch.equal(grpo_advantages([3.0] * 4), torch.zeros(4))
    assert torch.equal(grpo_advantages([5.0]), torch.zeros(1))


def test_budget_converts_macro_units_exactly():
    assert activevln_budget(n_star=60, macro_len=30) == 120          # 2*30 macros * 2 prim/macro
    assert activevln_budget(n_star=171, macro_len=90) == math.ceil(120 * 171 / 90)  # cap binds
    with pytest.raises(ValueError):
        activevln_budget(0, 3)


def test_dual_clip_policy_loss_reduces_to_policy_gradient_on_policy():
    logp = torch.tensor([-0.5, -1.0, -2.0], requires_grad=True)
    adv = torch.tensor([1.0, -2.0, 0.5])
    loss, clipped = policy_loss_terms(logp, logp.detach(), adv)
    loss.sum().backward()
    assert torch.allclose(logp.grad, -adv)
    assert not clipped.any()


def test_dual_clip_policy_loss_clips_like_verl():
    old = torch.zeros(4)
    logp = torch.log(torch.tensor([1.5, 0.5, 5.0, 1.1]))
    adv = torch.tensor([1.0, -1.0, -1.0, 1.0])
    loss, clipped = policy_loss_terms(logp, old, adv)
    assert loss[0] == pytest.approx(-1.28)        # clip-higher bound for positive advantage
    assert loss[1] == pytest.approx(0.8)          # lower bound for negative advantage
    assert loss[2] == pytest.approx(3.0)          # dual clip caps -A*ratio at -A*3
    assert loss[3] == pytest.approx(-1.1, rel=1e-5)
    assert clipped.tolist() == [True, True, True, False]


def test_loss_weights_reproduce_verl_micro_batch_token_mean():
    # 2 ranks x 4 trajectories, micro-batches of 2 => 4 global micro-batches.
    torch.manual_seed(0)
    tokens = [3, 5, 2, 7, 4, 4, 6, 1]
    grads = [torch.randn(n) for n in tokens]                  # d loss_token / d theta (scalar theta per token)
    reference = 0.0
    for mb in micro_batches(range(8)):
        reference += sum(grads[i].sum() for i in mb) / sum(tokens[i] for i in mb)
    reference /= 4
    total = 0.0
    for rank in (0, 1):
        idx = list(range(4 * rank, 4 * rank + 4))
        mbs = [i // 2 for i in idx]
        weights = loss_weights([tokens[i] for i in idx], mbs, world_size=2, num_micro_batches_global=4)
        total += sum(w * grads[i].sum() for w, i in zip(weights, idx))
    assert torch.allclose(total / 2, torch.as_tensor(reference))  # DeepSpeed averages ranks
