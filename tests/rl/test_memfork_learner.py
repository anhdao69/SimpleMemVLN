"""MemFork learner: calibration, exact export round trip, credited loss and fork-branch prefixes."""
import numpy as np
import torch

from qwen_vl.rl import memfork
from qwen_vl.rl.hico import memory_coefficients
from qwen_vl.stream.retention import RetentionPolicy


def _decisions(n, steps=None, seed=0, pool=5, layers=8):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        step = steps[i] if steps else 8 + i
        feats = rng.normal(-3.0, 0.4, size=(pool, layers))
        out.append(dict(step=step, pool=list(range(step - 8, step - 3)), features=feats.tolist(),
                        drop=int(rng.integers(pool)), logq=0.0, entropy=1.0))
    return out


def _cfg():
    return {'memory_policy': dict(enabled=True, hidden=16, lr=1e-2, log_kappa=0.0)}


def test_calibration_sets_normalisation_and_moderate_entropy():
    policy, _ = memfork.make_policy(_cfg(), 8, seed=1)
    result = memfork.calibrate(policy, _decisions(200), logit_spread=1.5)
    assert 0.3 < result['initial_entropy'] < np.log(5)
    np.testing.assert_allclose(policy.shift.numpy(), result['shift'])
    assert abs(float(policy.log_kappa) - result['log_kappa']) < 1e-12


def test_export_round_trip_is_exact_and_initialisation_is_rank_independent():
    a, _ = memfork.make_policy(_cfg(), 8, seed=3)
    b, _ = memfork.make_policy(_cfg(), 8, seed=3)
    for x, y in zip(a.state_dict().values(), b.state_dict().values()):
        assert torch.equal(x, y)
    with torch.no_grad():
        a.net[2].weight.normal_()
        a.log_kappa.fill_(0.123456789012345)
    c = RetentionPolicy.from_export(a.export())
    for (k, x), y in zip(a.state_dict().items(), c.state_dict().values()):
        assert x.dtype == torch.float64 and torch.equal(x, y), k


def test_update_loss_matches_credited_score_function_and_skips_branch_prefix():
    policy, optimizer = memfork.make_policy(_cfg(), 8, seed=0)
    memfork.calibrate(policy, _decisions(50, seed=9), 1.5)
    main = dict(num_steps=14, advantage=0.0, _c_logpi=[0.5] * 10 + [-2.0] * 4,
                memory=dict(decisions=_decisions(6, steps=[8, 9, 10, 11, 12, 13], seed=1)))
    # branch forked at t=10: decisions at steps <= 10 are the main's (prefix), not credited here
    branch = dict(_c_logpi=[0.0] * 10 + [1.5] * 3,
                  memory=dict(decisions=_decisions(4, steps=[9, 10, 11, 12], seed=2)))
    items = [(main, 1.0), (branch, 1.0)]
    expected = 0.0
    with torch.no_grad():
        for record in (main, branch):
            cs = memory_coefficients([d['step'] for d in record['memory']['decisions']], record['_c_logpi'])
            for d, c in zip(record['memory']['decisions'], cs):
                logq = policy.drop_log_probs(torch.tensor(d['features'], dtype=torch.float64))[d['drop']]
                expected -= c * float(logq)
    before = [p.detach().clone() for p in policy.parameters()]
    stats = memfork.update(policy, optimizer, items, n_main_global=4, grad_clip=1.0, device='cpu')
    assert abs(stats['memory_loss'] - expected / 4) < 1e-12
    assert stats['memory_credited_decisions'] == 6 + 2           # branch prefix decisions (9, 10) excluded
    assert any(not torch.equal(b, p) for b, p in zip(before, policy.parameters()))


def test_rollout_stats_and_metrics():
    records = [dict(memory=dict(decisions=_decisions(3, seed=4))), dict(memory=None)]
    merged = memfork.merged_metrics([memfork.rollout_stats(records), memfork.rollout_stats([])])
    assert merged['memory_decisions'] == 3 and 0 <= merged['memory_candidate_drop'] <= 1
