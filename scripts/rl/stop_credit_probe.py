"""Estimator probe on logged rollouts: STOP-logit gradient SNR and near-goal stop calibration.

python scripts/rl/stop_credit_probe.py RUN_DIR [--first K] [--last K] [--out OUT.json]

Uses the per-decision records HiCo runs log (step_d, step_stop_logp, actions, reward, groups), so all
estimators are compared on *identical* rollouts. For each episode, the gradient of the estimator
w.r.t. the STOP first-token logit, summed over decisions with d_t < 3 m, is (temperature T):
  REINFORCE-type term on the taken action:  c_logpi * (1[a_t = STOP] - sigma_t) / T
  analytic STOP term:                       c_sigma * sigma_t * (1 - sigma_t) / T
GRPO uses its group-normalised advantage as c_logpi (per-decision form; token weights ignored).
v3's baseline is re-fit offline exactly as in training (previous 4 RL steps only).
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from qwen_vl.rl.algos import grpo_advantages
from qwen_vl.rl.hico import (GeometricBaseline, batch_scale, coefficients, episode_features, rloo_baselines,
                             stop_values)

T = 1.2
R_MAX = 15.0


def load_steps(run):
    steps = defaultdict(list)
    for path in sorted((Path(run) / 'rollouts').glob('step_*_rank*.jsonl')):
        step = int(path.name.split('_')[1])
        steps[step] += [json.loads(line) for line in path.read_text().splitlines()]
    return dict(sorted(steps.items()))


def stop_logit_grad(episode, c_logpi, c_sigma):
    d = np.asarray(episode['step_d'])
    sigma = np.exp(np.asarray(episode['step_stop_logp'], dtype=np.float64))
    taken = np.asarray([a == 3 for a in episode['actions']], dtype=np.float64)
    g = (np.asarray(c_logpi) * (taken - sigma) + np.asarray(c_sigma) * sigma * (1 - sigma)) / T
    return float(g[d < 3.0].sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run')
    p.add_argument('--out')
    p.add_argument('--n-star', default='data/rl/activevln_r2r4k.jsonl')
    args = p.parse_args()
    n_star = {json.loads(l)['episode_id']: json.loads(l)['N_star'] for l in open(args.n_star)}
    steps = load_steps(args.run)
    critic = GeometricBaseline(history=4, iterations=400, lr=3e-3, r_max=R_MAX, seed=20261010, min_rows=2000)
    grads = defaultdict(list)
    near = dict(passed_failed=[], succeeded=[])
    for step, episodes in steps.items():
        for e in episodes:
            e['N_star'] = n_star[e['episode_id']]
        groups = defaultdict(list)
        for e in episodes:
            groups[(e['episode_id'], e.get('attempt', 0))].append(e)
        rewards = [e['reward'] for e in episodes]
        scale = batch_scale(rewards, 0.05 * R_MAX)
        critic.fit(step)
        for group in groups.values():
            if len(group) < 2:
                continue
            adv = grpo_advantages([e['reward'] for e in group]).numpy()
            rloo = rloo_baselines([e['reward'] for e in group])
            for e, a_grpo, b in zip(group, adv, rloo):
                q = stop_values(e['step_d'], R_MAX)
                n = len(q)
                grads['grpo'].append(stop_logit_grad(e, np.full(n, a_grpo), np.zeros(n)))
                for name, analytic, base in (('rloo_v2a', False, b), ('rloo_stop_v2', True, b)):
                    c_l, c_s = coefficients(e['actions'], e['reward'], q, np.full(n, base), scale, analytic)
                    grads[name].append(stop_logit_grad(e, c_l, c_s))
                pred = critic.predict(episode_features(e))
                if pred is not None:
                    c_l, c_s = coefficients(e['actions'], e['reward'], q, pred, scale, True)
                    grads['critic_stop_v3'].append(stop_logit_grad(e, c_l, c_s))
                d = np.asarray(e['step_d'])
                if (d < 3).any():
                    sig = np.exp(np.asarray(e['step_stop_logp']))[d < 3]
                    key = 'succeeded' if e['success'] else 'passed_failed'
                    near[key].append(dict(max_sigma=float(sig.max()), mean_sigma=float(sig.mean()),
                                          decisions_within_3m=int((d < 3).sum())))
        critic.add(step, np.concatenate([episode_features(e) for e in episodes]),
                   np.concatenate([np.full(len(e['step_d']), e['reward']) for e in episodes]))
    result = dict(run=args.run, steps=len(steps), estimators={})
    for name, g in grads.items():
        g = np.asarray(g)
        nonzero = g[g != 0]
        result['estimators'][name] = dict(
            episodes=len(g), mean=float(g.mean()), std=float(g.std(ddof=1)),
            snr=float(abs(g.mean()) / (g.std(ddof=1) / math.sqrt(len(g)))) if g.std() > 0 else None,
            fraction_nonzero=float(len(nonzero) / len(g)))
    for key, rows in near.items():
        if rows:
            m = np.asarray([r['max_sigma'] for r in rows])
            result[f'near_goal_{key}'] = dict(episodes=len(rows), max_sigma_median=float(np.median(m)),
                                              frac_max_sigma_above_0_1=float((m > 0.1).mean()),
                                              frac_max_sigma_below_0_01=float((m < 0.01).mean()))
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
