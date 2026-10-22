"""MemFork joint retention/action estimator: exact enumeration on a toy memory POMDP.

Decision t: a memory decision m_t ~ q_phi(. | x_t) (t >= 1, taken before the fork snapshot), then an
action a_t ~ pi_theta(. | x_t, m_t) whose distribution depends on what was kept. Terminal reward only.
The fork rule may use m_t (it precedes the snapshot) but never a_t. Credit as in the trainer:

    actions:  rl/hico.fork_coefficients (root advantage before the first fork, sibling contrasts after);
    memory:   m_s gets the coefficient of action s - 1 (``memory_coefficients``): the shared m_t at a
              fork point is root-credited, later memory decisions get their segment's sibling credit.

E[estimator] over the main trajectory and its branch must equal grad J for theta and phi. Crediting the
shared m_t with the sibling contrast (the counterfactual-memory mistake) must be detectably biased.
"""
import itertools
import math

import numpy as np
import pytest

from qwen_vl.rl.hico import fork_coefficients, memory_coefficients

T = 3


def sig(z):
    return 1.0 / (1.0 + math.exp(-z))


def p_mem(phi, x):
    return sig(phi[0] + phi[1] * x)


def p_act(theta, x, m):
    return sig(theta[0] + theta[1] * x + theta[2] * (2 * m - 1))


def reward(x, mems):
    return (x - 1.5) ** 2 + 0.7 * sum(mems) + (1.0 if x >= 2 and mems and mems[-1] else 0.0)


def suffix(theta, phi, t0, x0, mems0, m_at_t0=None):
    """All continuations from decision t0 (memory decision at t0 given if already taken)."""
    out = []

    def rec(t, x, p, mems, steps):
        if t == T:
            out.append((p, x, mems, steps))
            return
        options_m = [None] if t == 0 else ([m_at_t0] if (t == t0 and m_at_t0 is not None) else [0, 1])
        for m in options_m:
            pm = 1.0 if m is None or (t == t0 and m_at_t0 is not None) else (
                p_mem(phi, x) if m == 1 else 1 - p_mem(phi, x))
            mm = 0 if m is None else m
            for a in (0, 1):
                pa = p_act(theta, x, mm) if a == 1 else 1 - p_act(theta, x, mm)
                rec(t + 1, x + a, p * pm * pa, mems + ([m] if m is not None else []),
                    steps + [(t, x, m, a)])

    rec(t0, x0, 1.0, list(mems0), [])
    return out


def expected_return(theta, phi):
    return sum(p * reward(x, mems) for p, x, mems, _ in suffix(theta, phi, 0, 0, []))


def finite_difference(theta, phi, eps=1e-6):
    g = []
    for vec, i in [(0, i) for i in range(3)] + [(1, i) for i in range(2)]:
        up = [list(theta), list(phi)]
        down = [list(theta), list(phi)]
        up[vec][i] += eps
        down[vec][i] -= eps
        g.append((expected_return(*up) - expected_return(*down)) / (2 * eps))
    return np.asarray(g)


def score_action(theta, x, m, a):
    p = p_act(theta, x, m)
    return np.asarray([(a - p), (a - p) * x, (a - p) * (2 * m - 1), 0.0, 0.0])


def score_memory(phi, x, m):
    q = p_mem(phi, x)
    return np.asarray([0.0, 0.0, 0.0, (m - q), (m - q) * x])


def fork_point(steps):
    """Causal: first t >= 1 with (x_t + m_t) odd -- uses m_t (pre-snapshot), never a_t."""
    for t, x, m, a in steps:
        if t >= 1 and (x + m) % 2 == 1:
            return t
    return None


def expected_estimator(theta, phi, wrong_memory_credit=False, baseline=0.4):
    total = np.zeros(5)
    for p, x, mems, steps in suffix(theta, phi, 0, 0, []):
        r = reward(x, mems)
        t = fork_point(steps)
        mem_steps = [s for s in steps if s[2] is not None]
        if t is None:
            g = sum(score_action(theta, sx, sm or 0, sa) for _, sx, sm, sa in steps) * (r - baseline)
            g = g + sum(score_memory(phi, sx, sm) for _, sx, sm, _ in mem_steps) * (r - baseline)
            total += p * g
            continue
        tx, tm = steps[t][1], steps[t][2]
        prefix_mems = [s[2] for s in steps[:t] if s[2] is not None]
        for pb, xb, memsb, stepsb in suffix(theta, phi, t, tx, prefix_mems, m_at_t0=tm):
            rb = reward(xb, memsb)
            main_c, branch_c = fork_coefficients(T, [t], r, [[rb]], r - baseline, 1.0)
            c_mem = memory_coefficients([s[0] for s in mem_steps], main_c)
            if wrong_memory_credit:
                c_mem = [main_c[s] if s == t else c for s, c in zip([s[0] for s in mem_steps], c_mem)]
            g = sum(c * score_action(theta, sx, sm or 0, sa) for c, (_, sx, sm, sa) in zip(main_c, steps))
            g = g + sum(c * score_memory(phi, sx, sm) for c, (_, sx, sm, _) in zip(c_mem, mem_steps))
            cb = branch_c[0][0]
            full_c = [0.0] * t + [cb] * (T - t)
            branch_mem = [s for s in stepsb if s[2] is not None and s[0] > t]
            cb_mem = memory_coefficients([s[0] for s in branch_mem], full_c)
            g = g + sum(c * score_action(theta, sx, sm or 0, sa) for c, (_, sx, sm, sa) in zip(full_c[t:], stepsb))
            g = g + sum(c * score_memory(phi, sx, sm) for c, (_, sx, sm, _) in zip(cb_mem, branch_mem))
            total += p * pb * g
    return total


@pytest.fixture
def params():
    return [0.3, -0.4, 0.8], [-0.2, 0.6]


def test_joint_fork_estimator_is_unbiased(params):
    theta, phi = params
    truth = finite_difference(theta, phi)
    estimate = expected_estimator(theta, phi)
    assert np.abs(truth[3:]).max() > 0.05          # memory decisions matter in the toy
    np.testing.assert_allclose(estimate, truth, rtol=1e-5, atol=1e-7)


def test_sibling_credit_for_the_shared_memory_decision_is_biased(params):
    theta, phi = params
    truth = finite_difference(theta, phi)
    wrong = expected_estimator(theta, phi, wrong_memory_credit=True)
    assert np.abs(wrong[3:] - truth[3:]).max() > 1e-3


def test_memory_coefficients_index_the_preceding_action():
    main = np.asarray([0.5, 0.5, -1.0, -1.0, 2.0])
    assert memory_coefficients([1, 2, 3, 4], main) == [0.5, 0.5, -1.0, -1.0]
    with pytest.raises(ValueError):
        memory_coefficients([0], main)
    with pytest.raises(ValueError):
        memory_coefficients([5], main)
