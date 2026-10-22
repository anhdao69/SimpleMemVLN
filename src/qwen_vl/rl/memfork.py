"""MemFork learner side: the retention policy's calibration, update and metrics.

The retention policy q_phi (stream/retention.py) is a ~300-parameter float64 module replicated on every
rank (same initialisation, identical all-reduced gradients and Adam steps), sent to the rollout helpers
as a CPU state dict with each RL step's tasks. Its loss is the score-function term of the joint
policy gradient (plans/RL/research/memfork_design.md):

    L_phi = - (1 / N_main) * sum_trajectories sum_decisions c_d * log q_phi(drop_d | pool_d),

with c_d = rl/hico.memory_coefficients (the coefficient of the action preceding the decision: GRPO's
advantage, or GRPO+F-B's root/sibling credit). Pool features are the rollout's recorded values (the
exact behaviour-policy inputs), so no language-model forward is needed.
"""
import math

import numpy as np
import torch
import torch.distributed as dist

from qwen_vl.rl.hico import memory_coefficients
from qwen_vl.stream.retention import RetentionPolicy


def make_policy(cfg, layers, seed):
    m = cfg['memory_policy']
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        policy = RetentionPolicy(layers, m.get('hidden', 32), m.get('log_kappa', 0.0)).double()
    optimizer = torch.optim.Adam(policy.parameters(), lr=m['lr'], betas=tuple(m.get('betas', (0.9, 0.999))))
    return policy, optimizer


def payload(policy, mode='sample'):
    return dict(export=policy.export(), mode=mode)


def decisions_of(records):
    return [d for r in records if r.get('memory') for d in r['memory']['decisions']]


def calibrate(policy, decisions, logit_spread):
    """Input normalisation and initial kappa from greedy-retention episodes (training items only).

    shift/scale: per-layer mean/std of all pool features; kappa = logit_spread / sigma, sigma the mean
    within-pool std of the prior term mean_l x_l, so initial drop logits differ by ~logit_spread std.
    """
    pools = np.asarray([d['features'] for d in decisions], dtype=np.float64)        # [N, P, L]
    if pools.ndim != 3 or len(pools) < 10:
        raise RuntimeError(f'Memory calibration needs >= 10 decisions, got {len(pools)}')
    flat = pools.reshape(-1, pools.shape[-1])
    shift, scale = flat.mean(0), np.maximum(flat.std(0), 1e-6)
    sigma = float(pools.mean(-1).std(-1, ddof=0).mean())
    if not sigma > 0:
        raise RuntimeError('Degenerate memory features')
    with torch.no_grad():
        policy.shift.copy_(torch.from_numpy(shift))
        policy.scale.copy_(torch.from_numpy(scale))
        policy.log_kappa.fill_(math.log(logit_spread / sigma))
    probs = policy.drop_log_probs(torch.from_numpy(pools)).exp().detach().numpy()
    entropy = float(-(probs * np.log(np.maximum(probs, 1e-300))).sum(-1).mean())
    return dict(decisions=int(len(pools)), pool_size=int(pools.shape[1]), shift=shift.tolist(),
                scale=scale.tolist(), within_pool_sigma=sigma, log_kappa=float(policy.log_kappa),
                initial_entropy=entropy, initial_candidate_drop=float((probs.argmax(-1) == pools.shape[1] - 1).mean()))


def action_coefficients(record):
    if '_c_logpi' in record:
        return record['_c_logpi']
    return [record['advantage']] * record['num_steps']


def update(policy, optimizer, items, n_main_global, grad_clip, device):
    """One Adam step of q_phi on this RL step's credited memory decisions (all ranks)."""
    features, drops, coefficients = [], [], []
    for record, _ in items:
        memory = record.get('memory')
        if not memory or not memory['decisions']:
            continue
        cs = memory_coefficients([d['step'] for d in memory['decisions']], action_coefficients(record))
        for d, c in zip(memory['decisions'], cs):
            if c != 0.0:
                features.append(d['features'])
                drops.append(d['drop'])
                coefficients.append(c)
    policy.train()
    optimizer.zero_grad(set_to_none=False)
    loss_value = 0.0
    if features:
        pools = torch.tensor(features, dtype=torch.float64)                         # [N, P, L]
        logq = policy.drop_log_probs(pools).gather(-1, torch.tensor(drops).view(-1, 1)).squeeze(-1)
        loss = -(torch.tensor(coefficients, dtype=torch.float64) * logq).sum() / n_main_global
        loss.backward()
        loss_value = float(loss.detach())
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in policy.parameters()]
    flat = torch.cat([g.reshape(-1) for g in grads]).to(device)
    stats = torch.tensor([loss_value, len(features), float(np.abs(coefficients).sum()) if coefficients else 0.0],
                         dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(flat)
        dist.all_reduce(stats)
    offset = 0
    for p in policy.parameters():
        n = p.numel()
        p.grad = flat[offset:offset + n].view_as(p).cpu().clone()
        offset += n
    norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip))
    optimizer.step()
    policy.eval()
    credited = int(stats[1])
    return dict(memory_loss=float(stats[0]), memory_credited_decisions=credited,
                memory_c_abs=float(stats[2]) / max(1, credited), memory_grad_norm=norm,
                memory_log_kappa=float(policy.log_kappa))


def rollout_stats(records):
    """Behaviour of this rank's sampled memory decisions (main trajectories)."""
    decisions = decisions_of(records)
    if not decisions:
        return dict(decisions=0, entropy=0.0, candidate_drop=0, oldest_drop=0, drop_age=0.0)
    return dict(decisions=len(decisions), entropy=float(sum(d['entropy'] for d in decisions)),
                candidate_drop=sum(d['drop'] == len(d['pool']) - 1 for d in decisions),
                oldest_drop=sum(d['drop'] == 0 for d in decisions),
                drop_age=float(sum(d['step'] - d['pool'][d['drop']] for d in decisions)))


def merged_metrics(gathered):
    total = lambda k: sum(g[k] for g in gathered)
    n = max(1, total('decisions'))
    return dict(memory_decisions=total('decisions'), memory_entropy=total('entropy') / n,
                memory_candidate_drop=total('candidate_drop') / n, memory_oldest_drop=total('oldest_drop') / n,
                memory_drop_age=total('drop_age') / n)


def state(policy, optimizer, calibration):
    return dict(policy=policy.state_dict(), optimizer=optimizer.state_dict(), calibration=calibration)


def load(policy, optimizer, payload_):
    policy.load_state_dict(payload_['policy'])
    optimizer.load_state_dict(payload_['optimizer'])
    return payload_.get('calibration')
