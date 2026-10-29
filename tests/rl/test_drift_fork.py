"""S8N event-boundary fork placement (``fork.trigger: drift | drift_entropy``) on the fake session and simulator."""
import json
import math

import numpy as np
import pytest
import torch

from qwen_vl.rl.rollout import event_score, position_bucket, run_episode
from test_fork_rollout import FORK, REWARD, FakeEnv, FakeSession


class FakeDriftSession(FakeSession):
    """FakeSession whose responses carry per-layer S8N drift priorities of the observation o_t.

    The drift depends only on the observation (rendered from the pose) and the history, never on a_t.
    """

    def sample_step(self, t, rgb, temperature, generator):
        before = self.memory.clone()
        result = super().sample_step(t, rgb, temperature, generator)
        level = 0.05 + float(rgb[0, 0, 0] % 17) / 17 + 0.01 * float(before[1])
        result['drift'] = [level, 2 * level]
        return result


def episode(fork, seed=7):
    return run_episode(FakeDriftSession(), FakeEnv(), dict(episode_id='e1', N_star=10), temperature=1.2,
                       budget=20, seed=seed, sample_idx=0, reward_cfg=REWARD, device='cpu', fork=fork)


def test_event_score_is_the_mean_log_layer_drift():
    assert event_score(None) is None
    assert math.isclose(event_score([0.5, 2.0]), 0.0, abs_tol=1e-12)
    assert math.isclose(event_score([math.e, math.e ** 3]), 2.0)
    for bad in ([], [0.0, 1.0], [-1.0], [float('nan')], [[1.0]]):
        with pytest.raises(ValueError):
            event_score(bad)


def test_drift_forks_fire_only_at_high_drift_decisions():
    seen = 0
    for seed in range(12):
        e = episode(dict(FORK, trigger='drift', min_drift=-0.5, min_entropy=None), seed=seed)
        scores = [event_score(d) for d in e['step_drift']]
        assert all(len(d) == 2 for d in e['step_drift'])
        for t in e['fork_points']:
            seen += 1
            assert scores[t] >= -0.5
        assert len({b['fork_t'] for b in e['_branches']}) == len(e['fork_points'])
    assert seen > 0


def test_drift_forks_leave_the_main_trajectory_unchanged():
    for seed in range(5):
        plain = episode(None, seed=seed)
        for trigger in ('drift', 'drift_entropy'):
            forked = episode(dict(FORK, trigger=trigger, min_drift=-1.0, min_entropy=0.0), seed=seed)
            for key in ('actions', 'reward', 'rollout_logps', 'step_d', 'step_drift'):
                assert plain[key] == forked[key]


def test_drift_entropy_needs_both_signals():
    for seed in range(8):
        both = episode(dict(FORK, trigger='drift_entropy', min_drift=-1.0, min_entropy=1e9), seed=seed)
        assert both['fork_points'] == []
        both = episode(dict(FORK, trigger='drift_entropy', min_drift=1e9, min_entropy=0.0), seed=seed)
        assert both['fork_points'] == []


def test_drift_trigger_without_drift_is_rejected():
    with pytest.raises(ValueError, match='S8N'):
        for seed in range(10):
            run_episode(FakeSession(), FakeEnv(), dict(episode_id='e1', N_star=10), temperature=1.2, budget=20,
                        seed=seed, sample_idx=0, reward_cfg=REWARD, device='cpu',
                        fork=dict(FORK, trigger='drift', min_drift=0.0))
    with pytest.raises(ValueError, match='trigger'):
        episode(dict(FORK, trigger='surprise', min_drift=0.0))


def test_drift_threshold_is_a_quantile_of_the_previous_step(tmp_path):
    from qwen_vl.rl.train_grpo import fork_threshold
    (tmp_path / 'rollouts').mkdir()
    rows = [dict(step_drift=[[1.0, 1.0], [math.e, math.e], None], step_entropy=[0.1, 0.3, 0.5]),
            dict(step_drift=[[math.e ** 2, math.e ** 2]], step_entropy=[0.7])]
    (tmp_path / 'rollouts' / 'step_0004_rank0.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    cfg = dict(drift_quantile=0.5, initial_min_drift=0.25, entropy_quantile=0.5, initial_min_entropy=0.1)
    assert fork_threshold(tmp_path, 5, cfg, signal='drift') == pytest.approx(1.0)
    assert fork_threshold(tmp_path, 5, cfg) == pytest.approx(0.4)
    assert fork_threshold(tmp_path, 4, cfg, signal='drift') == 0.25
    assert fork_threshold(tmp_path, 4, cfg) == 0.1


def test_position_buckets_double_in_width():
    assert [position_bucket(t) for t in range(16)] == [0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4]


def test_position_thresholds_apply_per_bucket():
    """A per-bucket threshold list fires exactly where the decision's own bucket threshold holds."""
    lists = ([9.0, 9.0, -9.0], [-9.0, 9.0, 9.0, 9.0], [9.0, -9.0])
    seen = 0
    for seed in range(12):
        for thresholds in lists:
            e = episode(dict(FORK, trigger='drift', min_drift=thresholds, min_entropy=None), seed=seed)
            for t in e['fork_points']:
                seen += 1
                assert thresholds[min(position_bucket(t), len(thresholds) - 1)] < 0
    assert seen > 0
    for seed in range(5):    # a list of one equals the scalar threshold
        scalar = episode(dict(FORK, trigger='drift', min_drift=-0.5, min_entropy=None), seed=seed)
        listed = episode(dict(FORK, trigger='drift', min_drift=[-0.5], min_entropy=None), seed=seed)
        assert scalar['fork_points'] == listed['fork_points']


def test_position_thresholds_are_bucket_quantiles_of_the_previous_step(tmp_path):
    from qwen_vl.rl.train_grpo import MIN_BUCKET, fork_threshold
    (tmp_path / 'rollouts').mkdir()
    rng = np.random.default_rng(0)
    # Event scores fall with t like S8N's (every bucket has >= MIN_BUCKET decisions here).
    rows = []
    for _ in range(40):
        n = int(rng.integers(5, 7)) if len(rows) % 4 else 40
        rows.append(dict(step_drift=[[math.exp(-0.1 * t + rng.normal(0, 0.05))] * 2 for t in range(n)]))
    (tmp_path / 'rollouts' / 'step_0002_rank0.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    cfg = dict(drift_quantile=0.5, initial_min_drift=[0.0, -1.0], drift_by_position=True)
    got = fork_threshold(tmp_path, 3, cfg, signal='drift')
    pairs = [(t, event_score(d)) for r in rows for t, d in enumerate(r['step_drift'])]
    by = lambda b: [v for t, v in pairs if position_bucket(t) == b]
    assert len(got) == position_bucket(39) + 1
    for b, threshold in enumerate(got):
        if len(by(b)) >= MIN_BUCKET:
            assert threshold == pytest.approx(np.quantile(by(b), 0.5))
    assert got == sorted(got, reverse=True)                     # later buckets get lower thresholds
    assert fork_threshold(tmp_path, 2, cfg, signal='drift') == [0.0, -1.0]
    assert fork_threshold(tmp_path, 3, dict(cfg, drift_by_position=False), signal='drift') == pytest.approx(
        np.quantile([v for _, v in pairs], 0.5))


def test_sparse_buckets_inherit_the_previous_threshold(tmp_path):
    from qwen_vl.rl.train_grpo import MIN_BUCKET, fork_threshold
    (tmp_path / 'rollouts').mkdir()
    rows = [dict(step_drift=[[math.exp(-t)] * 2 for t in range(3)]) for _ in range(MIN_BUCKET)]
    rows.append(dict(step_drift=[[math.exp(-t)] * 2 for t in range(8)]))   # buckets 2 and 3: one episode only
    (tmp_path / 'rollouts' / 'step_0000_rank0.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    got = fork_threshold(tmp_path, 1, dict(drift_quantile=0.5, initial_min_drift=[0.0], drift_by_position=True),
                         signal='drift')
    assert got[0] == pytest.approx(0.0) and got[1] == pytest.approx(-1.5)
    assert got[2] == got[3] == got[1]


def test_calibration_reports_the_step_one_thresholds(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    from qwen_vl.rl.train_grpo import fork_threshold
    (tmp_path / 'rollouts').mkdir()
    rows = [dict(step_drift=[[math.exp(-0.2 * t + 0.1 * k)] * 2 for t in range(20)]) for k in range(20)]
    (tmp_path / 'rollouts' / 'step_0000_rank0.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    script = Path(__file__).resolve().parents[2] / 'scripts' / 'rl' / 'calibrate_drift.py'
    got = json.loads(subprocess.run([sys.executable, str(script), str(tmp_path)], check=True,
                                    capture_output=True, text=True).stdout)
    cfg = dict(drift_quantile=0.5, initial_min_drift=None)
    assert got['pooled'] == pytest.approx(fork_threshold(tmp_path, 1, cfg, signal='drift'))
    assert got['by_position'] == pytest.approx(
        fork_threshold(tmp_path, 1, dict(cfg, drift_by_position=True), signal='drift'))
    assert len(got['by_position']) == 5
