"""HiCo-PO estimators: analytic STOP branch + out-of-sample geometric state baseline.

Refined from plans/RL_technical.md (v2a/v2/v3) and plans/RL_survey.md section 7; derivation and
conditions in plans/RL/research/hico_po_report.md.

Per visited decision t of episode i (terminal-only reward R_i, gamma = 1):

    loss_i = - sum_t [ c_logpi[t] * log pi_T(a_t | h_t) + c_sigma[t] * sigma_t ]

  * rloo        (v2a): c_logpi = (R_i - b)/s for every decision (STOP included), c_sigma = 0.
  * rloo_stop   (v2) : c_sigma = (Q_S(t) - b)/s at every decision;
                       c_logpi = (R_i - b)/s only where a_t != STOP.
  * critic_stop (v3) : as v2 with b = b(z_t), a state baseline fit only on previous RL steps.

sigma_t = pi_T(first response token = STOP | h_t) and Q_S(t) is the exact return of stopping at the
pose of decision t (STOP does not move the agent and ends the episode). For any baseline that does
not depend on episode i's own actions, E[estimator] equals the policy gradient (tests/rl/
test_hico_estimator.py enumerates a toy POMDP with these production functions).
"""
import math

import numpy as np
import torch

STOP = 3
FEATURES = ('d', 'd_over_d0', 'path_over_d0', 't_over_nstar', 'within_3m', 'cos_bearing',
            'sin_bearing', 'd0', 'min_d_over_d0', 'bearing_missing')


def stop_values(distances, success_reward_base=15.0):
    """Exact return of STOP at each decision under ActiveVLN's reward 15*(1-d/3)*1[d<3]."""
    d = np.asarray(distances, dtype=np.float64)
    return np.where(d < 3.0, success_reward_base * (1.0 - d / 3.0), 0.0)


def rloo_baselines(rewards):
    """Leave-one-out mean of the other group members (independent of the episode's own actions)."""
    r = np.asarray(rewards, dtype=np.float64)
    if r.size < 2:
        raise ValueError('RLOO needs at least two samples per group')
    return (r.sum() - r) / (r.size - 1)


def batch_scale(rewards, floor):
    """One positive scalar per RL step (a step size; conditional unbiasedness holds given it)."""
    r = np.asarray(rewards, dtype=np.float64)
    spread = float(r.std(ddof=1)) if r.size > 1 else 0.0
    return max(spread, floor)


def coefficients(actions, reward, q_stop, baseline, scale, analytic_stop):
    """Per-decision loss coefficients (c_logpi, c_sigma); ``baseline`` is a scalar or per-decision."""
    actions = list(actions)
    n = len(actions)
    q_stop = np.asarray(q_stop, dtype=np.float64)
    b = np.broadcast_to(np.asarray(baseline, dtype=np.float64), (n,))
    if q_stop.shape != (n,):
        raise ValueError('One stop value per decision is required')
    if not scale > 0:
        raise ValueError('Scale must be positive')
    c_logpi = (reward - b) / scale
    c_sigma = np.zeros(n)
    if analytic_stop:
        c_sigma = (q_stop - b) / scale
        # The STOP branch is integrated analytically; only sampled non-STOP outcomes keep log-pi terms.
        c_logpi = np.where(np.asarray([a == STOP for a in actions]), 0.0, c_logpi)
    return c_logpi.astype(np.float64), c_sigma.astype(np.float64)


def grpo_stop_coefficients(actions, reward, q_stop, group_rewards, epsilon=1e-6, stop_scale_floor=0.75):
    """GRPO with the STOP branch integrated analytically (estimator ``grpo_stop``).

    Movement decisions keep GRPO's group-normalised advantage (R - mean_g)/(std_g + eps) on log pi; the
    STOP branch at every decision gets (Q_S(t) - mean_g)/max(std_g + eps, floor) on sigma_t, replacing the
    log-pi term of a taken STOP. The floor only matters in (near) zero-variance groups, where GRPO's own
    movement terms vanish but exact STOP values still carry signal.
    """
    r = np.asarray(group_rewards, dtype=np.float64)
    mean, std = float(r.mean()), float(r.std(ddof=1)) if r.size > 1 else 0.0
    advantage = (reward - mean) / (std + epsilon)
    q_stop = np.asarray(q_stop, dtype=np.float64)
    stop = np.asarray([a == STOP for a in actions])
    c_logpi = np.where(stop, 0.0, advantage)
    c_sigma = (q_stop - mean) / max(std + epsilon, stop_scale_floor)
    return c_logpi, c_sigma


def fork_coefficients(num_decisions, fork_points, main_reward, branch_rewards, root_advantage, scale):
    """Per-decision (coefficient * weight) for a main trajectory and its forked branches (``grpo_fork``).

    ``fork_points`` are increasing decision indices t_1 < ... < t_K of the main trajectory, chosen by a
    causal rule (only information available before a_t); ``branch_rewards[k]`` holds the returns of the
    M_k continuations sampled from the same history h_{t_k}. With terminal-only reward:

      * main decisions s < t_1 keep the root (GRPO) advantage, weight 1;
      * main decisions s in [t_k, t_{k+1}) get (R_main - mean(branch_rewards[k])) / scale;
      * branch b at t_k gets (R_b - mean(the other members of {main} + branches at t_k)) / scale on its
        own decisions s >= t_k (its prefix is the main's and is not trained here);
      * every decision at or after t_k carries the weight prod_{j <= k} 1 / (1 + M_j).

    The 1 + M_k suffixes from h_{t_k} are i.i.d. samples of the same conditional suffix gradient, so
    averaging them keeps E[estimator] equal to the policy gradient (the root part's baseline aside),
    while the sibling baseline is conditioned on the state h_{t_k} (tests/rl/test_hico_estimator.py).
    Returns (main coefficients, [[branch coefficient per suffix decision] per branch] per fork).
    """
    if list(fork_points) != sorted(set(fork_points)) or any(not 0 <= t < num_decisions for t in fork_points):
        raise ValueError('Fork points must be increasing decisions of the main trajectory')
    if len(branch_rewards) != len(fork_points) or any(len(b) == 0 for b in branch_rewards):
        raise ValueError('Every fork point needs at least one branch')
    if not scale > 0:
        raise ValueError('Scale must be positive')
    main = np.full(num_decisions, float(root_advantage))
    branches, weight = [], 1.0
    for k, (t, rewards) in enumerate(zip(fork_points, branch_rewards)):
        weight /= 1 + len(rewards)
        end = fork_points[k + 1] if k + 1 < len(fork_points) else num_decisions
        main[t:end] = weight * (main_reward - float(np.mean(rewards))) / scale
        members = [main_reward] + list(rewards)
        branches.append([weight * (r - (sum(members) - r) / (len(members) - 1)) / scale for r in rewards])
    return main, branches


def episode_features(episode):
    """Baseline features z_t (none depends on a_t), one row per decision."""
    d = np.asarray(episode['step_d'], dtype=np.float64)
    n = d.size
    d0 = max(float(episode['d0']), 1e-3)
    nstar = max(float(episode.get('N_star') or 1), 1.0)
    path = np.asarray(episode['step_path'], dtype=np.float64)
    bearing = [b if b is not None and math.isfinite(b) else None for b in episode['step_bearing']]
    cos_b = np.asarray([math.cos(b) if b is not None else 0.0 for b in bearing])
    sin_b = np.asarray([math.sin(b) if b is not None else 0.0 for b in bearing])
    missing = np.asarray([1.0 if b is None else 0.0 for b in bearing])
    t = np.arange(n, dtype=np.float64)
    return np.stack([d, d / d0, path / d0, t / nstar, (d < 3.0).astype(np.float64), cos_b, sin_b,
                     np.full(n, d0), np.minimum.accumulate(d) / d0, missing], axis=1)


class GeometricBaseline:
    """b(z) ~ E[R | z]: small CPU MLP fit on (z_t, R) rows of the previous ``history`` RL steps only."""

    def __init__(self, history=4, hidden=64, iterations=400, lr=3e-3, r_max=15.0, seed=0, min_rows=500):
        self.history, self.hidden, self.iterations, self.lr = history, hidden, iterations, lr
        self.r_max, self.seed, self.min_rows = r_max, seed, min_rows
        self.buffer = []          # [(rl_step, Z, y)]
        self.model = None
        self.diagnostics = {}

    def add(self, step, z, y):
        self.buffer.append((int(step), np.asarray(z, np.float32), np.asarray(y, np.float32)))
        latest = max(s for s, _, _ in self.buffer)
        self.buffer = [row for row in self.buffer if row[0] > latest - self.history]

    def _net(self, width):
        return torch.nn.Sequential(torch.nn.Linear(width, self.hidden), torch.nn.Tanh(),
                                   torch.nn.Linear(self.hidden, self.hidden), torch.nn.Tanh(),
                                   torch.nn.Linear(self.hidden, 1))

    def fit(self, current_step):
        rows = [(z, y) for s, z, y in self.buffer if s < current_step]
        if not rows or sum(len(y) for _, y in rows) < self.min_rows:
            self.model = None
            self.diagnostics = dict(fitted=False, rows=sum(len(y) for _, y in rows))
            return self.diagnostics
        z = torch.from_numpy(np.concatenate([r[0] for r in rows]))
        y = torch.from_numpy(np.concatenate([r[1] for r in rows])) / self.r_max
        generator = torch.Generator().manual_seed(self.seed + current_step)
        order = torch.randperm(len(y), generator=generator)
        held = order[: max(1, len(y) // 10)]
        train = order[len(held):]
        mean, std = z[train].mean(0), z[train].std(0).clamp_min(1e-6)
        torch.manual_seed(self.seed + current_step)
        net = self._net(z.shape[1])
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        zt = (z - mean) / std
        for _ in range(self.iterations):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(net(zt[train]).squeeze(-1), y[train])
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            pred = net(zt[held]).squeeze(-1).clamp(0, 1)
            resid = float(((pred - y[held]) ** 2).mean())
            var = float(y[held].var()) if len(held) > 1 else 0.0
        self.model = dict(state=net.state_dict(), mean=mean, std=std, width=z.shape[1])
        self.diagnostics = dict(fitted=True, rows=int(len(y)), heldout_r2=1 - resid / var if var > 0 else None,
                                train_mse=float(loss.detach()) * self.r_max ** 2)
        return self.diagnostics

    def predict(self, z):
        if self.model is None:
            return None
        net = self._net(self.model['width'])
        net.load_state_dict(self.model['state'])
        with torch.no_grad():
            zt = (torch.as_tensor(np.asarray(z, np.float32)) - self.model['mean']) / self.model['std']
            return (net(zt).squeeze(-1).clamp(0, 1) * self.r_max).double().numpy()

    def state(self):
        return dict(buffer=self.buffer, model=self.model, diagnostics=self.diagnostics)

    def load(self, state):
        self.buffer, self.model, self.diagnostics = state['buffer'], state['model'], state['diagnostics']


def action_terms(logp_tokens, stop_logp_tokens, action_index, num_actions):
    """Per-decision log pi_T(a_t) (sum of its sampled tokens) and sigma_t (first-token STOP prob)."""
    logpi = torch.zeros(num_actions, dtype=logp_tokens.dtype, device=logp_tokens.device)
    logpi = logpi.index_add(0, action_index, logp_tokens)
    first = torch.ones_like(action_index, dtype=torch.bool)
    first[1:] = action_index[1:] != action_index[:-1]
    if int(first.sum()) != num_actions:
        raise ValueError('Every decision needs at least one response token')
    sigma = stop_logp_tokens[first].exp()
    return logpi, sigma


def memory_coefficients(decision_steps, action_coefficients, snapshot_before_memory=False):
    """MemFork: coefficient of each memory decision m_s (taken before observation s, s >= 1).

    * Snapshots after the memory decision (``fork.placement: action``): m_s belongs to the history shared by
      every branch at s, i.e. to the trajectory segment of a_{s-1}, and takes that coefficient (the root
      advantage up to the first fork point inclusive, the segment's sibling contrast after it, zero on
      a branch's replayed prefix).
    * Snapshots before the memory decision (``fork.placement: memory``): branches at s re-draw m_s, which then
      belongs to the segment of a_s and takes its coefficient.

    tests/rl/test_memfork_estimator.py enumerates the joint estimator for both placements.
    """
    c = list(action_coefficients)
    offset = 0 if snapshot_before_memory else 1
    out = []
    for s in decision_steps:
        if not 1 <= int(s) < len(c):
            raise ValueError(f'Memory decision {s} outside decisions 1..{len(c) - 1}')
        out.append(float(c[int(s) - offset]))
    return out
