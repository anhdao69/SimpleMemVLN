"""G7b: exact expected-gradient check of the production HiCo-PO coefficient functions.

Toy POMDP (plans/RL_technical.md Appendix E): corridor 0..6, goal 4, success radius 1.6, actions
forward/back/STOP, horizon 6, truncation gives 0, logits indexed by (t, x). Every trajectory is
enumerated; E[estimator] must equal the finite-difference gradient of the expected return.
"""
import numpy as np
import pytest
import torch

from qwen_vl.rl.hico import (STOP, GeometricBaseline, action_terms, batch_scale, coefficients,
                             rloo_baselines, stop_values)

H, L, GOAL, RADIUS = 6, 6, 4, 1.6
F, B = 0, 1
S = 2            # toy STOP index; mapped to hico.STOP when calling production code
TOY_TO_HICO = {F: 0, B: 1, S: STOP}


def q_stop(x):
    d = abs(x - GOAL)
    return float(d < RADIUS) * (0.5 + 0.5 * (1 - d / RADIUS))


def pi(theta, t, x):
    z = np.exp(theta[t, x] - theta[t, x].max())
    return z / z.sum()


def move(x, a):
    return min(L, x + 1) if a == F else max(0, x - 1)


def trajectories(theta):
    out = []

    def rec(t, x, p, hist):
        if t == H:
            out.append((p, hist, 0.0))
            return
        probs = pi(theta, t, x)
        for a in (F, B, S):
            if a == S:
                out.append((p * probs[S], hist + [(t, x, S)], q_stop(x)))
            else:
                rec(t + 1, move(x, a), p * probs[a], hist + [(t, x, a)])

    rec(0, 0, 1.0, [])
    return out


def expected_return(theta):
    return sum(p * r for p, _, r in trajectories(theta))


def finite_difference(theta, eps=1e-6):
    grad = np.zeros_like(theta)
    for index in np.ndindex(theta.shape):
        up, down = theta.copy(), theta.copy()
        up[index] += eps
        down[index] -= eps
        grad[index] = (expected_return(up) - expected_return(down)) / (2 * eps)
    return grad


def episode_gradient(theta, hist, reward, baseline_fn, analytic_stop):
    """Gradient w.r.t. the logits built only from the production coefficients()."""
    actions = [TOY_TO_HICO[a] for _, _, a in hist]
    q = np.asarray([q_stop(x) for _, x, _ in hist])
    b = np.asarray([baseline_fn(t, x) for t, x, _ in hist])
    c_logpi, c_sigma = coefficients(actions, reward, q, b, 1.0, analytic_stop)
    grad = np.zeros_like(theta)
    eye = np.eye(3)
    for k, (t, x, a) in enumerate(hist):
        probs = pi(theta, t, x)
        grad[t, x] += c_logpi[k] * (eye[a] - probs) + c_sigma[k] * probs[S] * (eye[S] - probs)
    return grad


def expected_estimator(theta, baseline_fn, analytic_stop):
    total = np.zeros_like(theta)
    for p, hist, reward in trajectories(theta):
        total += p * episode_gradient(theta, hist, reward, baseline_fn, analytic_stop)
    return total


@pytest.fixture(scope='module')
def problem():
    theta = np.random.default_rng(0).normal(0, 1, (H, L + 1, 3))
    return theta, finite_difference(theta)


def relative_error(estimate, truth):
    return np.abs(estimate - truth).max() / np.abs(truth).max()


@pytest.mark.parametrize('analytic_stop', [True, False])
@pytest.mark.parametrize('baseline', ['zero', 'constant', 'state', 'poor_state'])
def test_estimators_are_unbiased(problem, analytic_stop, baseline):
    theta, truth = problem
    fn = dict(zero=lambda t, x: 0.0, constant=lambda t, x: 0.3,
              state=lambda t, x: q_stop(x) * 0.8 + 0.05 * t,
              poor_state=lambda t, x: 0.1 + 0.7 * (x % 2))[baseline]
    assert relative_error(expected_estimator(theta, fn, analytic_stop), truth) < 1e-6


def test_rloo_group_baseline_is_unbiased(problem):
    """G = 2 RLOO: b(tau1) = R(tau2). The estimator is affine in b, so E = E[U] - E[R] E[W]."""
    theta, truth = problem
    trajs = trajectories(theta)
    u = sum(p * episode_gradient(theta, h, r, lambda t, x: 0.0, True) for p, h, r in trajs)
    w = sum(p * (episode_gradient(theta, h, r, lambda t, x: 0.0, True)
                 - episode_gradient(theta, h, r, lambda t, x: 1.0, True)) for p, h, r in trajs)
    mean_reward = sum(p * r for p, _, r in trajs)
    assert relative_error(u - mean_reward * w, truth) < 1e-6
    assert np.allclose(rloo_baselines([1.0, 3.0, 5.0]), [4.0, 3.0, 2.0])


def test_wrong_estimators_fail(problem):
    """Power check: double-counting STOP, or crediting the stop branch with the episode return, is biased."""
    theta, truth = problem

    def double_count(theta_, hist, reward, baseline_fn, analytic_stop):
        g = episode_gradient(theta_, hist, reward, baseline_fn, True)
        eye = np.eye(3)
        for t, x, a in hist:
            if a == S:  # also keep a log-pi term at the taken STOP
                g[t, x] += (reward - baseline_fn(t, x)) * (eye[S] - pi(theta_, t, x))
        return g

    def return_as_stop_value(theta_, hist, reward, baseline_fn, analytic_stop):
        g = np.zeros_like(theta_)
        eye = np.eye(3)
        for t, x, a in hist:
            probs = pi(theta_, t, x)
            g[t, x] += (reward - 0.3) * probs[S] * (eye[S] - probs)  # R instead of the exact Q_S(t)
            if a != S:
                g[t, x] += (reward - 0.3) * (eye[a] - probs)
        return g

    for wrong in (double_count, return_as_stop_value):
        estimate = sum(p * wrong(theta, h, r, lambda t, x: 0.3, True) for p, h, r in trajectories(theta))
        assert relative_error(estimate, truth) > 1e-2


def test_stop_values_and_scale():
    assert stop_values([0.0, 1.5, 2.999, 3.0, 9.0]).tolist() == pytest.approx([15.0, 7.5, 15 * (1 - 2.999 / 3), 0.0, 0.0])
    assert batch_scale([5.0, 5.0], floor=0.75) == 0.75
    assert batch_scale([0.0, 15.0], floor=0.75) == pytest.approx(np.std([0.0, 15.0], ddof=1))


def test_coefficients_shapes_and_stop_rule():
    c_logpi, c_sigma = coefficients([0, 1, 3], reward=6.0, q_stop=[0.0, 3.0, 6.0], baseline=2.0, scale=2.0,
                                    analytic_stop=True)
    assert c_logpi.tolist() == [2.0, 2.0, 0.0]          # no log-pi term at the taken STOP
    assert c_sigma.tolist() == [-1.0, 0.5, 2.0]
    c_logpi, c_sigma = coefficients([0, 3], reward=6.0, q_stop=[0.0, 6.0], baseline=[1.0, 2.0], scale=1.0,
                                    analytic_stop=False)
    assert c_logpi.tolist() == [5.0, 4.0] and c_sigma.tolist() == [0.0, 0.0]


def test_action_terms_sum_tokens_and_read_first_token():
    logp = torch.log(torch.tensor([0.5, 0.9, 0.8, 0.7, 0.6]))
    stop = torch.log(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]))
    index = torch.tensor([0, 0, 1, 1, 2])
    logpi, sigma = action_terms(logp, stop, index, 3)
    assert torch.allclose(logpi, torch.log(torch.tensor([0.45, 0.56, 0.6])))
    assert torch.allclose(sigma, torch.tensor([0.1, 0.3, 0.5]))


def test_geometric_baseline_is_out_of_sample_and_bounded():
    rng = np.random.default_rng(0)
    critic = GeometricBaseline(history=2, iterations=200, min_rows=50)
    for step in range(3):
        z = rng.normal(size=(100, 10)).astype(np.float32)
        critic.add(step, z, 15 * (z[:, 0] > 0))
    critic.fit(current_step=2)   # may use steps 0..1 only
    assert critic.diagnostics['rows'] == 100   # history=2 keeps steps 1,2; step 2 is excluded
    pred = critic.predict(rng.normal(size=(20, 10)))
    assert pred.min() >= 0 and pred.max() <= 15
    assert critic.diagnostics['heldout_r2'] > 0.3


def test_loss_gradient_matches_analytic_form():
    """G7: d loss / d logits of -(c_logpi*logpi + c_sigma*sigma) with tempered softmax."""
    torch.manual_seed(0)
    vocab, temperature, stop_token = 7, 1.2, 5
    logits = torch.randn(4, vocab, dtype=torch.float64, requires_grad=True)
    tokens = torch.tensor([2, 3, 5, 1])            # decision 0: tokens (2,3); 1: (5); 2: (1)
    index = torch.tensor([0, 0, 1, 2])
    logp_all = torch.log_softmax(logits / temperature, -1)
    logp = logp_all.gather(1, tokens[:, None]).squeeze(1)
    logpi, sigma = action_terms(logp, logp_all[:, stop_token], index, 3)
    c_logpi = torch.tensor([0.7, 0.0, -1.3], dtype=torch.float64)
    c_sigma = torch.tensor([0.4, 2.0, -0.6], dtype=torch.float64)
    (-(c_logpi * logpi + c_sigma * sigma).sum()).backward()
    p = torch.softmax(logits.detach() / temperature, -1)
    eye = torch.eye(vocab, dtype=torch.float64)
    expected = torch.zeros_like(logits)
    first = [0, 2, 3]
    for row, (tok, dec) in enumerate(zip(tokens.tolist(), index.tolist())):
        expected[row] -= c_logpi[dec] * (eye[tok] - p[row]) / temperature
    for dec, row in enumerate(first):
        expected[row] -= c_sigma[dec] * p[row, stop_token] * (eye[stop_token] - p[row]) / temperature
    assert torch.allclose(logits.grad, expected, atol=1e-12)


def test_grpo_stop_coefficients_keep_grpo_moves_and_integrate_stop():
    from qwen_vl.rl.hico import grpo_stop_coefficients
    group = [10.0, 0.0, 5.0, 5.0]
    mean, std = 5.0, float(np.std(group, ddof=1))
    c_logpi, c_sigma = grpo_stop_coefficients([0, 1, 3], 10.0, [0.0, 4.0, 10.0], group)
    a = (10.0 - mean) / (std + 1e-6)
    assert np.allclose(c_logpi, [a, a, 0.0])                     # GRPO advantage, no log-pi at the STOP
    assert np.allclose(c_sigma, (np.array([0.0, 4.0, 10.0]) - mean) / (std + 1e-6))
    # zero-variance group: GRPO's moves vanish, exact STOP values still give signal (floored scale)
    c_logpi, c_sigma = grpo_stop_coefficients([0, 0], 0.0, [6.0, 0.0], [0.0, 0.0, 0.0, 0.0], stop_scale_floor=0.75)
    assert np.allclose(c_logpi, 0.0) and np.allclose(c_sigma, [8.0, 0.0])
