"""grpo_fork rollouts and record assembly on a fake streaming session and simulator (CPU only)."""
import copy

import numpy as np
import pytest
import torch

from qwen_vl.contracts import HABITAT_IDS
from qwen_vl.rl.rollout import run_episode

REWARD = dict(success_reward_base=15.0)
TO_CLASS = {v: k for k, v in enumerate(HABITAT_IDS)}


class FakeEnv:
    """Corridor: forward moves 1 m toward a goal 6 m away; turns change only the heading."""

    def __init__(self, goal=6.0, render_salt=0):
        self.goal, self.salt = goal, render_salt
        self.resets = 0

    def _obs(self):
        rgb = np.full((480, 640, 3), (int(self.x * 10) + 7 * self.heading + self.salt) % 256, dtype=np.uint8)
        return dict(distance_to_goal=abs(self.goal - self.x), position=[self.x, 0.0, 0.0],
                    bearing_error=0.0, rgb=rgb, episode_over=False)

    def reset(self, episode_id):
        self.x, self.heading, self.resets = 0.0, 0, self.resets + 1
        return dict(self._obs(), scene_id='scene', instruction='go', episode_id=episode_id)

    def step(self, action):
        cls = TO_CLASS[action]
        if cls == 0:
            self.x += 1.0
        elif cls in (1, 2):
            self.heading = (self.heading + (1 if cls == 1 else -1)) % 4
        obs = self._obs()
        if cls == 3:
            d = abs(self.goal - self.x)
            obs.update(episode_over=True, metrics=dict(success=float(d < 3.0), spl=0.5 if d < 3.0 else 0.0))
        return obs


class FakeSession:
    """Policy whose action distribution depends on the full history (a running 'memory' tensor)."""

    def reset(self, uid, instruction):
        self.memory, self.next_step, self.valid = torch.zeros(4), 0, True

    def prepare_step(self, step_id, generator):     # promotion before the snapshot (no memory here)
        pass

    def snapshot(self):
        return dict(memory=self.memory.clone(), next_step=self.next_step)

    def restore(self, snapshot, episode_uid=None):
        self.memory, self.next_step, self.valid = snapshot['memory'].clone(), snapshot['next_step'], True

    def _logp(self, t, rgb, temperature):
        assert t == self.next_step and self.valid
        self.memory = 0.9 * self.memory + torch.tensor([float(rgb[0, 0, 0]) / 255, 1.0, t / 10, 0.0])
        logits = torch.tensor([1.0, 0.2 * self.memory[0], 0.1, -2.0 + 0.6 * t])
        return torch.log_softmax(logits / temperature, -1)

    def _commit(self, cls):
        self.memory[3] += cls
        self.next_step += 1

    def sample_step(self, t, rgb, temperature, generator):
        logp = self._logp(t, rgb, temperature)
        cls = int(torch.multinomial(logp.exp(), 1, generator=generator))
        self._commit(cls)
        entropy = float(-(logp.exp() * logp).sum())
        return dict(valid=True, class_id=cls, complete=True, tokens=[100 + cls, 9], logps=[float(logp[cls]), 0.0],
                    entropies=[entropy, 0.0], greedy_tokens=[100, 9], stop_logp=float(logp[3]))

    def forced_step(self, t, rgb, temperature, cls):
        logp = self._logp(t, rgb, temperature)
        self._commit(cls)
        return float(logp[cls])


def episode(fork=None, env=None, seed=7):
    return run_episode(FakeSession(), env or FakeEnv(), dict(episode_id='e1', N_star=8), temperature=1.2,
                       budget=20, seed=seed, sample_idx=0, reward_cfg=REWARD, device='cpu', fork=fork)


FORK = dict(max_forks=2, branches=2, min_entropy=0.0, window=3)


def test_main_trajectory_is_unchanged_by_forking():
    for seed in range(5):
        plain, forked = episode(seed=seed), episode(dict(FORK), seed=seed)
        for key in ('actions', 'reward', 'rollout_logps', 'step_d', 'step_stop_logp'):
            assert plain[key] == forked[key]


def test_branches_continue_from_the_exact_history():
    found = False
    for seed in range(10):
        e = episode(dict(FORK), seed=seed)
        for b in e['_branches']:
            found = True
            t = b['fork_t']
            assert 1 <= t < e['num_steps'] and t in e['fork_points']
            # the branch's first decision sees the same pose as the main's decision t
            assert b['step_d'][0] == e['step_d'][t]
            assert np.array_equal(b['_frames'][0], e['_frames'][t])
            assert len(b['actions']) == b['num_steps'] == len(b['_responses'])
            # teacher-forcing main prefix + branch suffix through a fresh session reproduces the branch
            # log-probs exactly: the restored memory is the main's memory at decision t
            session, env = FakeSession(), FakeEnv()
            obs = env.reset('e1')
            session.reset('u', 'go')
            forced = []
            for k, a in enumerate(e['actions'][:t] + b['actions']):
                forced.append(session.forced_step(k, obs['rgb'], 1.2, a))
                if a != 3:
                    obs = env.step(HABITAT_IDS[a])
            assert forced[:t] == e['rollout_logps'][0:2 * t:2]
            assert forced[t:] == b['rollout_logps'][0::2]
        assert len({b['branch'] for b in e['_branches']}) <= FORK['branches']
    assert found


def test_replay_divergence_is_detected():
    class Drifting(FakeEnv):
        def reset(self, episode_id):
            self.salt = self.resets       # renders differently after the first reset
            return super().reset(episode_id)

    with pytest.raises(RuntimeError, match='Replay diverged'):
        for seed in range(10):
            e = episode(dict(FORK, max_forks=1), env=Drifting(), seed=seed)
            assert not e['_branches']     # only forkless episodes may pass


def test_fork_placement_is_causal():
    """Fork points depend only on pre-drawn targets and entropies, never on the sampled actions."""
    e = episode(dict(FORK, min_entropy=0.0, window=0), seed=3)
    # window 0 and threshold 0: every fork sits exactly on a target -> independent of actions
    for b in e['_branches']:
        assert b['fork_t'] in e['fork_points']
    high = episode(dict(FORK, min_entropy=1e9), seed=3)
    assert high['_branches'] == [] and high['fork_points'] == []


def test_fork_prepare_builds_full_branch_records():
    from qwen_vl.rl.train_grpo import fork_prepare
    cfg = dict(algorithm=dict(scale_floor=0.05, epsilon=1e-6), reward=REWARD)
    group = [episode(dict(FORK, max_forks=1, branches=1), seed=s) for s in (1, 2)]
    for i, e in enumerate(group):
        e['uid'] = f'u{i}'
        e['advantage'] = 0.7 if i == 0 else -0.7
    snapshot = copy.deepcopy([{k: v for k, v in e.items() if k != '_frames'} for e in group])
    items, stats = fork_prepare([group], {'u0': 0.25, 'u1': 0.5}, cfg)
    assert stats['mains'] == 2 and stats['branches'] == sum(len(e['_branches']) for e in snapshot)
    for record, w in items:
        if record.get('is_branch'):
            parent = next(e for e in group if record['uid'].startswith(e['uid'] + ':'))
            t = int(record['uid'].split(':fork')[1].split(':')[0])
            assert w == {'u0': 0.25, 'u1': 0.5}[parent['uid']]
            assert record['_c_logpi'][:t] == [0.0] * t and len(record['_c_logpi']) == len(record['_responses'])
            assert len(record['rollout_logps']) == sum(len(r['tokens']) for r in record['_responses'])
            assert record['response_tokens'] == len(record['rollout_logps'])
            assert record['_responses'][:t] == parent['_responses'][:t]
        else:
            assert len(record['_c_logpi']) == record['num_steps']
            points = record['fork_points']
            if points:
                assert record['_c_logpi'][:points[0]] == [record['advantage']] * points[0]


def test_fork_prepare_batch_scale():
    """fork_scale: batch divides sibling contrasts by the step scale instead of the group std."""
    from qwen_vl.rl.train_grpo import fork_prepare
    group = [episode(dict(FORK, max_forks=1, branches=1), seed=s) for s in range(6)]
    for i, e in enumerate(group):
        e['uid'], e['advantage'] = f'u{i}', 0.0
    cfg = dict(algorithm=dict(scale_floor=0.05, epsilon=1e-6), reward=REWARD)
    group_items, _ = fork_prepare([copy.deepcopy(group)], {e['uid']: 1.0 for e in group}, cfg)
    batch_items, _ = fork_prepare([copy.deepcopy(group)], {e['uid']: 1.0 for e in group},
                                  dict(cfg, algorithm=dict(cfg['algorithm'], fork_scale='batch')), step_scale=4.0)
    rewards = np.asarray([e['reward'] for e in group])
    group_scale = max(rewards.std(ddof=1) + 1e-6, 0.75)
    pairs = [(g, b) for (g, _), (b, _) in zip(group_items, batch_items)]
    assert pairs and len(group_items) == len(batch_items)
    for g, b in pairs:
        if g.get('is_branch') or g.get('fork_points'):
            np.testing.assert_allclose(np.asarray(b['_c_logpi'][-1]) * 4.0, np.asarray(g['_c_logpi'][-1]) * group_scale)
    with pytest.raises(ValueError, match='step scale'):
        fork_prepare([copy.deepcopy(group)], {e['uid']: 1.0 for e in group},
                     dict(cfg, algorithm=dict(cfg['algorithm'], fork_scale='batch')))


class FakeMemorySession(FakeSession):
    """FakeSession with a MemFork-like memory decision drawn in prepare_step (t >= 3) that changes the
    policy, and a pre-decision entropy that depends only on the history."""

    def reset(self, uid, instruction):
        super().reset(uid, instruction)
        self.uid, self.decisions, self.log = uid, [], getattr(self, 'log', [])

    def memory_entropy(self, t):
        assert all(s != t for s, _ in self.decisions), 'entropy must be read before m_t is drawn'
        return None if t < 3 else 0.2 + float(self.memory[0] % 1.0)

    def prepare_step(self, t, generator):
        if t < 3 or any(s == t for s, _ in self.decisions):
            return
        m = int(torch.randint(2, (), generator=generator))
        self.decisions.append((t, m))
        self.memory[1] += 0.7 * m
        self.log.append((self.uid, t, m))

    def snapshot(self):
        assert all(s != self.next_step for s, _ in self.decisions), 'snapshot must precede m_t'
        return dict(super().snapshot(), decisions=list(self.decisions))

    def restore(self, snapshot, episode_uid=None):
        super().restore(snapshot, episode_uid)
        self.decisions, self.uid = list(snapshot['decisions']), episode_uid


MEMORY_FORK = dict(max_forks=2, branches=2, min_entropy=0.5, window=3, placement='memory', min_t=3)


def test_memory_forks_snapshot_before_the_memory_decision_and_redraw_it():
    differs = 0
    for seed in range(12):
        session = FakeMemorySession()
        item = dict(episode_id='e1', N_star=12)
        plain = run_episode(FakeMemorySession(), FakeEnv(), item, temperature=1.2, budget=20, seed=seed,
                            sample_idx=0, reward_cfg=REWARD, device='cpu')
        e = run_episode(session, FakeEnv(), item, temperature=1.2, budget=20, seed=seed, sample_idx=0,
                        reward_cfg=REWARD, device='cpu', fork=dict(MEMORY_FORK))
        assert e['actions'] == plain['actions'] and e['rollout_logps'] == plain['rollout_logps']
        main = {t: m for uid, t, m in session.log if uid == e['uid']}
        for b in e['_branches']:
            t = b['fork_t']
            assert t >= 3 and t in e['fork_points']
            drawn = {s: m for uid, s, m in session.log if uid == f"{e['uid']}:fork{t}:{b['branch']}"}
            assert t in drawn                                  # the branch re-drew m_t itself
            differs += drawn[t] != main[t]
    assert differs > 0
