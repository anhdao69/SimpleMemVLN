"""H1 of plans/RL/research/s8n_event_rl_design.md: does the S8N event score at a fork predict how
informative the fork is? Reads the rollout logs of a grpo_fork run on S8N (``step_drift``,
``step_entropy``, ``fork_summary`` of every main trajectory).

Per fork (main decision t, one branch): informative = R_branch != R_main, gap = |R_branch - R_main|;
event = rl/rollout.event_score(step_drift[t]); entropy = first-token entropy at t. Reports P(informative)
and mean gap by event-score quintile (bootstrap 95% CI over episodes), Spearman correlations, and a
logistic regression of informative on standardised (event, entropy, t / N*) to see whether the event
score adds information beyond entropy.
The raw score falls with t (||S_before|| grows over the episode), so the same tables are also given for
event_pos = event - median event score of all main decisions of the same training step in the same
rl/rollout.position_bucket, with a logistic regression on (event_pos, entropy, log(1 + t)).

Usage: python scripts/rl/analyze_event_forks.py RUN_DIR [--steps A:B] [--out report.json]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from qwen_vl.rl.rollout import event_score, position_bucket  # noqa: E402


def ranks(x):
    order = np.argsort(x, kind='mergesort')
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for value in np.unique(x):                       # average ranks of ties
        tie = x == value
        r[tie] = r[tie].mean()
    return r


def spearman(x, y):
    if len(x) < 3:
        return None
    a, b = ranks(np.asarray(x, float)), ranks(np.asarray(y, float))
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def logistic(X, y, iterations=200, l2=1e-3):
    """Newton-Raphson logistic regression with an intercept; returns (coefficients, standard errors)."""
    X = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X.shape[1])
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-X @ w))
        H = X.T @ (X * (p * (1 - p))[:, None]) + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - l2 * w)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    p = 1 / (1 + np.exp(-X @ w))
    cov = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]) + l2 * np.eye(X.shape[1]))
    return w, np.sqrt(np.diag(cov))


def load(run, lo, hi):
    rows, pooled = [], {}
    for path in sorted((Path(run) / 'rollouts').glob('step_*_rank*.jsonl')):
        step = int(path.name.split('_')[1])
        if not lo <= step < hi:
            continue
        for line in path.read_text().splitlines():
            e = json.loads(line)
            drift, entropy = e.get('step_drift') or [], e.get('step_entropy') or []
            scores = [event_score(d) for d in drift]
            for t, v in enumerate(scores):
                if v is not None:
                    pooled.setdefault((step, position_bucket(t)), []).append(v)
            for f in e.get('fork_summary') or []:
                t = f['fork_t']
                if t >= len(scores) or scores[t] is None:
                    continue
                own = np.asarray([s for s in scores if s is not None])
                rows.append(dict(step=step, episode=e['episode_id'], uid=e['uid'], t=t,
                                 n_star=float(e.get('N_star') or e['num_steps']), event=scores[t],
                                 event_in_episode=float((own < scores[t]).mean()),
                                 entropy=entropy[t] if t < len(entropy) and entropy[t] is not None else float('nan'),
                                 main_reward=e['reward'], branch_reward=f['reward'],
                                 informative=float(f['reward'] != e['reward']),
                                 gap=abs(f['reward'] - e['reward'])))
    medians = {k: float(np.median(v)) for k, v in pooled.items()}
    for r in rows:
        r['event_pos'] = r['event'] - medians[(r['step'], position_bucket(r['t']))]
    return rows


def bootstrap(rows, key, groups, seed=0, draws=2000):
    episodes = sorted({r['episode'] for r in rows})
    by_episode = {ep: [r for r in rows if r['episode'] == ep] for ep in episodes}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        sample = [r for ep in rng.choice(episodes, len(episodes)) for r in by_episode[ep]]
        out.append([np.mean([r[key] for r in sample if g(r)] or [np.nan]) for g in groups])
    return np.nanpercentile(np.asarray(out), [2.5, 97.5], axis=0)


def quintiles(rows, key):
    """P(informative) and mean gap by quintile of ``key`` with bootstrap 95% CIs over episodes."""
    value = np.asarray([r[key] for r in rows])
    edges = np.quantile(value, np.linspace(0, 1, 6))
    bins = [lambda r, a=edges[k], b=edges[k + 1], last=k == 4: a <= r[key] <= b if last else a <= r[key] < b
            for k in range(5)]
    ci_inf, ci_gap = bootstrap(rows, 'informative', bins), bootstrap(rows, 'gap', bins, seed=1)
    table = []
    for k, inside in enumerate(bins):
        sel = [r for r in rows if inside(r)]
        table.append(dict(quintile=k + 1, range=[float(edges[k]), float(edges[k + 1])], forks=len(sel),
                          p_informative=float(np.mean([r['informative'] for r in sel])),
                          p_informative_ci=[float(ci_inf[0][k]), float(ci_inf[1][k])],
                          mean_gap=float(np.mean([r['gap'] for r in sel])),
                          mean_gap_ci=[float(ci_gap[0][k]), float(ci_gap[1][k])],
                          mean_t=float(np.mean([r['t'] for r in sel]))))
    return table


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run')
    p.add_argument('--steps', default='0:100000')
    p.add_argument('--out')
    args = p.parse_args()
    lo, hi = map(int, args.steps.split(':'))
    rows = load(args.run, lo, hi)
    if not rows:
        raise SystemExit('no fork records with drift')
    finite = [r for r in rows if math.isfinite(r['entropy'])]
    fits = {}
    for name, features in [('raw', lambda r: [r['event'], r['entropy'], r['t'] / r['n_star']]),
                           ('position', lambda r: [r['event_pos'], r['entropy'], math.log1p(r['t'])])]:
        X = np.asarray([features(r) for r in finite])
        X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
        w, se = logistic(X, np.asarray([r['informative'] for r in finite]))
        fits[name] = dict(coefficients=w.tolist(), standard_errors=se.tolist(), z=(w / se).tolist())
    fits['raw']['features'] = ['intercept', 'event', 'entropy', 't_over_nstar']
    fits['position']['features'] = ['intercept', 'event_pos', 'entropy', 'log1p_t']
    pick = lambda key: [r[key] for r in rows]
    report = dict(run=str(args.run), steps=[lo, hi], forks=len(rows), episodes=len({r['episode'] for r in rows}),
                  p_informative=float(np.mean(pick('informative'))),
                  by_event_quintile=quintiles(rows, 'event'),
                  by_event_pos_quintile=quintiles(rows, 'event_pos'),
                  spearman=dict(event_informative=spearman(pick('event'), pick('informative')),
                                event_gap=spearman(pick('event'), pick('gap')),
                                event_pos_informative=spearman(pick('event_pos'), pick('informative')),
                                event_pos_gap=spearman(pick('event_pos'), pick('gap')),
                                entropy_gap=spearman([r['entropy'] for r in finite], [r['gap'] for r in finite]),
                                event_entropy=spearman([r['event'] for r in finite], [r['entropy'] for r in finite]),
                                event_t=spearman(pick('event'), pick('t')),
                                event_pos_t=spearman(pick('event_pos'), pick('t'))),
                  logistic=fits['raw'], logistic_position=fits['position'])
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + '\n')


if __name__ == '__main__':
    main()
