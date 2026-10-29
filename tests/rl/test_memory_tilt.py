"""S8N memory twins (rl/memory_tilt.py): common-random-number twins, antithetic tilt gradient, scheduling."""
import numpy as np
import pytest
import torch

from qwen_vl.rl.memory_tilt import TiltLearner, pair_stats
from qwen_vl.rl.rollout import run_episode
from qwen_vl.rl.train_grpo import rollout_rank_pool
from test_fork_rollout import FORK, REWARD, FakeEnv, FakeSession

LAYERS = [3, 7]
CFG = dict(twins=1, sigma_consensus=0.25, sigma_birth=0.05)


class FakeTiltSession(FakeSession):
    """FakeSession whose policy, from step 3 on, depends on the retention rule (like a changed memory)."""

    def _logp(self, t, rgb, temperature):
        logp = super()._logp(t, rgb, temperature)
        if t >= 3 and self.tilt is not None:
            logits = logp * temperature
            logits[1] = logits[1] + 40.0 * float(sum(self.tilt.birth))
            logp = torch.log_softmax(logits / temperature, -1)
        return logp


def rollout(tilt, fork=None, seed=11, session=None):
    return run_episode(session or FakeTiltSession(), FakeEnv(), dict(episode_id='e1', N_star=8), temperature=1.2,
                       budget=20, seed=seed, sample_idx=0, reward_cfg=REWARD, device='cpu', fork=fork, tilt=tilt)


def test_twins_share_the_action_stream_of_their_main():
    learner = TiltLearner(CFG, LAYERS)
    zero = learner.tilt()
    for seed in range(8):
        main = rollout(zero, fork=FORK, seed=seed)
        twin = rollout(zero, seed=seed)
        # the main's forks consume no randomness of its own action stream
        assert main['actions'] == twin['actions'] and main['reward'] == twin['reward']
        assert main['memory_tilt'] == twin['memory_tilt'] == [0.0] * 4


def test_twins_differ_only_after_the_memory_rule_matters():
    learner = TiltLearner(CFG, LAYERS)
    diverged = 0
    for seed in range(20):
        eps = learner.epsilon(0, 1, 'e1', 0, 0)
        plus, minus = rollout(learner.tilt(eps, +1.0), seed=seed), rollout(learner.tilt(eps, -1.0), seed=seed)
        n = min(3, len(plus['actions']), len(minus['actions']))
        assert plus['actions'][:n] == minus['actions'][:n]          # identical before step 3
        diverged += plus['actions'] != minus['actions']
        np.testing.assert_allclose(plus['memory_tilt'], -np.asarray(minus['memory_tilt']))
    assert diverged > 0


def test_epsilon_is_deterministic_and_per_sample():
    learner = TiltLearner(CFG, LAYERS)
    a = learner.epsilon(1, 2, 'e1', 0, 0)
    np.testing.assert_array_equal(a, learner.epsilon(1, 2, 'e1', 0, 0))
    for other in (learner.epsilon(1, 2, 'e1', 0, 1), learner.epsilon(1, 2, 'e1', 1, 0),
                  learner.epsilon(1, 3, 'e1', 0, 0), learner.epsilon(1, 2, 'e2', 0, 0)):
        assert not np.array_equal(a, other)
    assert a.shape == (4,)


def test_antithetic_gradient_recovers_the_return_slope():
    """R(w) = c . w: the twin estimate is unbiased for c / scale; Adam then climbs along c."""
    learner = TiltLearner(CFG, LAYERS)
    c, scale = np.asarray([2.0, -1.0, 30.0, 0.0]), 4.0
    R = lambda w: float(c @ w)
    pairs = []
    for k in range(4000):
        eps = learner.epsilon(0, 0, 'e', 0, k)
        pairs.append((R(learner.w + learner.sigma * eps), R(learner.w - learner.sigma * eps), eps))
    g = learner.gradient(pairs, scale)
    np.testing.assert_allclose(g, c / scale, atol=0.06 * np.abs(c / scale).max())
    for _ in range(20):
        learner.step(c / scale)
    assert learner.w[0] > 0 and learner.w[1] < 0 and learner.w[2] > 0 and learner.w[3] == 0
    assert np.all(learner.gradient([], scale) == 0) and np.all(learner.gradient(pairs, 0.0) == 0)
    copy_ = TiltLearner(CFG, LAYERS)
    copy_.load(learner.state())
    np.testing.assert_array_equal(copy_.w, learner.w)
    assert copy_.t == learner.t


def test_pair_stats():
    m = dict(actions=[0, 0, 1, 3], reward=15.0, success=True)
    same = dict(actions=[0, 0, 1, 3], reward=15.0, success=True, num_steps=4)
    other = dict(actions=[0, 2, 0, 0, 3], reward=0.0, success=False, num_steps=5)
    s = pair_stats([(m, same), (m, other)])
    assert s['twin_pairs'] == 2 and s['twin_diverged'] == 0.5 and s['twin_first_divergence'] == 1
    assert s['twin_success_flip'] == 0.5 and s['twin_reward_gap_abs'] == 7.5 and s['twin_env_steps'] == 9


class FakePool:
    """Runs tasks in submission order; reward depends on the seed and on the tilt (like a simulator)."""

    def __init__(self):
        self.tasks, self.log = [], []

    def submit(self, key, item, **kw):
        self.tasks.append((key, item, kw))
        self.log.append((key, kw))

    def get(self):
        key, item, kw = self.tasks.pop(0)
        rng = np.random.default_rng(kw['seed'] % 2 ** 32)
        birth = sum(kw['tilt']['birth']) if kw.get('tilt') else 0.0
        success = item['episode_id'] != 'hard' and rng.random() < 0.5 + birth
        return key, dict(success=bool(success), reward=15.0 * success, num_steps=5, seed=kw['seed'],
                         actions=[0, 3], uid=f"{item['episode_id']}:{kw['sample_idx']}")


def pool_cfg():
    return dict(rollout=dict(temperature=1.2, group_size=4, budget_factor=2.0, budget_cap=120,
                             dynamic_sampling=dict(enabled=True, max_sample_attempts=2)),
                run=dict(seed=5), reward=REWARD)


def test_twins_are_scheduled_for_final_groups_only():
    items = [dict(episode_id=e, N_star=10, activevln_merged_len=8) for e in ('a', 'hard', 'b')]
    learner, pool = TiltLearner(CFG, LAYERS), FakePool()
    groups, history, twins = rollout_rank_pool(pool, items, 3, pool_cfg(), tilt=learner)
    twin_tasks = [(key, kw) for key, kw in pool.log if isinstance(key[2], tuple)]
    assert len(twin_tasks) == 3 and len(twins) == 3                 # one per final group
    assert len(history[1]) == 3                                       # 'hard' was re-rolled twice
    for main, twin, eps in twins:
        assert main['seed'] == twin['seed'] and main['attempt'] == twin['attempt']
    for key, kw in twin_tasks:
        i, attempt, (_, k) = key
        assert attempt == groups[i][0]['attempt'] and k == 0 and kw['fork'] is None and not kw['keep_frames']
        main_kw = next(m for mk, m in pool.log if mk == (i, attempt, 0))
        np.testing.assert_allclose(np.add(main_kw['tilt']['birth'], kw['tilt']['birth']), 0.0, atol=1e-15)
    # samples without a twin act with the unperturbed rule
    assert all(kw['tilt'] == learner.tilt() for key, kw in pool.log if not isinstance(key[2], tuple) and key[2] > 0)
    # without a learner nothing changes: no tilt, two return values
    plain_pool = FakePool()
    assert len(rollout_rank_pool(plain_pool, items, 3, pool_cfg())) == 2
    assert all(kw['tilt'] is None for _, kw in plain_pool.log)


def test_independent_twins_ablation_draws_other_seeds():
    items = [dict(episode_id=e, N_star=10, activevln_merged_len=8) for e in ('a', 'b')]
    crn, plain = TiltLearner(CFG, LAYERS), TiltLearner(dict(CFG, common_random_numbers=False), LAYERS)
    assert crn.crn and not plain.crn
    _, _, paired = rollout_rank_pool(FakePool(), items, 3, pool_cfg(), tilt=crn)
    _, _, independent = rollout_rank_pool(FakePool(), items, 3, pool_cfg(), tilt=plain)
    _, _, again = rollout_rank_pool(FakePool(), items, 3, pool_cfg(), tilt=TiltLearner(dict(CFG, common_random_numbers=False), LAYERS))
    assert all(m['seed'] == t['seed'] for m, t, _ in paired)
    assert all(m['seed'] != t['seed'] for m, t, _ in independent)
    assert [t['seed'] for _, t, _ in independent] == [t['seed'] for _, t, _ in again]
    # the perturbation directions are the same: only the action random stream of the twin changes
    for (_, _, e1), (_, _, e2) in zip(paired, independent):
        np.testing.assert_array_equal(e1, e2)


def test_invalid_config():
    with pytest.raises(ValueError):
        TiltLearner(dict(CFG, sigma_birth=0.0), LAYERS)
    with pytest.raises(ValueError):
        TiltLearner(dict(CFG, init=[0.0]), LAYERS)
