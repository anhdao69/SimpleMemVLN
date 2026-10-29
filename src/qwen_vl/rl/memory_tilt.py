"""S8N memory twins: an RL-learned per-layer frame retention rule (plans/RL/research/s8n_event_rl_design.md §8).

The rule is ``stream/sweep_retention.RetentionTilt`` with parameters w (per attention layer: consensus
a_l and birth b_l; w = 0 is S8N). Each RL step perturbs it for the first ``twins`` samples of every
final group: main sample k acts with w + sigma * eps and its *twin* re-runs the same episode with the
same seed (the same action random stream, no forks) and w - sigma * eps. The two trajectories are
identical until a memory difference changes an action, so their return gap is caused by the
memory rule alone (common random numbers; M5's MemFork had no such control and its retention credit
was drowned by action noise). The antithetic estimate

    g_k = E[(R(w + sigma eps) - R(w - sigma eps)) / (2 scale) * eps_k / sigma_k]  =  d E[R] / d w_k / scale

(smoothed by the Gaussian perturbation) drives Adam ascent on w, identical on every rank. Twins are
never trained on: the action policy's update is unchanged (mains and their F-B branches only).
Evaluation exports w and runs it greedily (deterministic top-F under the learned priority).
"""
import numpy as np

from qwen_vl.rl.rollout import rollout_seed
from qwen_vl.stream.sweep_retention import RetentionTilt


class TiltLearner:
    def __init__(self, cfg, layers):
        self.layers = [int(l) for l in layers]
        n = len(self.layers)
        self.twins = int(cfg.get('twins', 1))
        self.crn = bool(cfg.get('common_random_numbers', True))   # False: ablation, twins draw independent seeds
        self.sigma = np.asarray([cfg['sigma_consensus']] * n + [cfg['sigma_birth']] * n, dtype=np.float64)
        if not (self.sigma > 0).all():
            raise ValueError('memory_tilt sigmas must be positive')
        self.lr = cfg.get('lr_scale', 0.2) * self.sigma
        self.betas = tuple(cfg.get('betas', (0.9, 0.999)))
        self.w = np.asarray(cfg.get('init') or [0.0] * (2 * n), dtype=np.float64)
        if self.w.shape != (2 * n,):
            raise ValueError(f'memory_tilt init must have {2 * n} values (consensus then birth, layer order)')
        self.m = np.zeros(2 * n)
        self.v = np.zeros(2 * n)
        self.t = 0

    def epsilon(self, seed, step, episode_id, attempt, sample_idx):
        """Standard normal direction of one (step, episode, attempt, sample); layout independent."""
        rng = np.random.default_rng(rollout_seed(seed, 'memory-tilt', step, episode_id, attempt, sample_idx))
        return rng.standard_normal(self.w.shape)

    def tilt(self, eps=None, sign=1.0):
        w = self.w if eps is None else self.w + sign * self.sigma * eps
        return RetentionTilt.from_vector(self.layers, w).to_dict()

    def gradient(self, pairs, scale):
        """``pairs``: (R_plus, R_minus, eps) of every twin pair of the step (all ranks)."""
        if not pairs or not scale > 0:
            return np.zeros_like(self.w)
        terms = [(rp - rm) / (2.0 * scale) * np.asarray(eps) / self.sigma for rp, rm, eps in pairs]
        return np.mean(terms, axis=0)

    def step(self, g):
        """One Adam ascent step on w."""
        b1, b2 = self.betas
        self.t += 1
        self.m = b1 * self.m + (1 - b1) * g
        self.v = b2 * self.v + (1 - b2) * g * g
        m_hat, v_hat = self.m / (1 - b1 ** self.t), self.v / (1 - b2 ** self.t)
        delta = self.lr * m_hat / (np.sqrt(v_hat) + 1e-12)
        self.w = self.w + delta
        return float(np.linalg.norm(g)), float(np.abs(delta).max())

    def state(self):
        return dict(w=self.w.tolist(), m=self.m.tolist(), v=self.v.tolist(), t=self.t)

    def load(self, state):
        self.w, self.m, self.v = (np.asarray(state[k], dtype=np.float64) for k in ('w', 'm', 'v'))
        self.t = int(state['t'])

    def summary(self):
        n = len(self.layers)
        return dict(tilt_consensus=self.w[:n].tolist(), tilt_birth=self.w[n:].tolist(),
                    tilt_consensus_mean=float(self.w[:n].mean()), tilt_birth_mean=float(self.w[n:].mean()))


def pair_stats(pairs):
    """Behaviour of the step's twin pairs: (main record, twin record) of every pair (all ranks)."""
    if not pairs:
        return dict(twin_pairs=0)
    diverged = [m['actions'] != t['actions'] for m, t in pairs]
    first = [next((i for i, (a, b) in enumerate(zip(m['actions'], t['actions'])) if a != b),
                  min(len(m['actions']), len(t['actions']))) for (m, t), d in zip(pairs, diverged) if d]
    return dict(twin_pairs=len(pairs), twin_diverged=float(np.mean(diverged)),
                twin_first_divergence=float(np.mean(first)) if first else None,
                twin_reward_gap_abs=float(np.mean([abs(m['reward'] - t['reward']) for m, t in pairs])),
                twin_success_flip=float(np.mean([m['success'] != t['success'] for m, t in pairs])),
                twin_env_steps=int(sum(t['num_steps'] for _, t in pairs)))
