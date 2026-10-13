"""Memory-related behaviour of complete evaluator journals on the same val_unseen episodes (CPU only).

python scripts/rl/analyze_memory.py OUT.json NAME=EVAL_DIR [NAME=EVAL_DIR ...]

The Window8 policy keeps the last 8 observations as KV and everything older only in the recurrent
(Gated DeltaNet) state. From the recorded agent positions and actions this reports, per policy:
  * revisits: positions within ``radius`` of a place visited more than 8 decisions earlier after the
    agent had moved >= ``excursion`` m away from it (the earlier visit is older than the KV window, so
    only the recurrent state can tell the agent it has been there);
  * spins: runs of >= 24 same-direction 15-degree turns (a full 360 degrees without moving);
  * reversals: adjacent TURN_LEFT/TURN_RIGHT pairs per decision;
  * passed goal: came within 3 m (oracle success) but ended farther than 3 m;
  * cap hits: episodes forced to stop at the 500-action limit;
and success conditional on these events, plus the paired transition table for every pair of policies.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

WINDOW, RADIUS, EXCURSION, SPIN = 8, 0.5, 2.0, 24
STOP, FORWARD, LEFT, RIGHT = 0, 1, 2, 3          # Habitat action ids


def load(directory):
    rows = (json.loads(line) for line in (Path(directory) / 'episodes.jsonl').read_text().splitlines())
    return {r['episode_id']: r for r in rows if not r['failed']}


def revisits(points):
    """Indices i whose position returns within RADIUS of some j <= i - WINDOW - 1 after an excursion."""
    p = np.asarray(points, dtype=np.float64)
    n = len(p)
    if n <= WINDOW + 1:
        return []
    dist = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    hits = []
    for i in range(WINDOW + 1, n):
        for j in np.nonzero(dist[i, : i - WINDOW] < RADIUS)[0]:
            if dist[j, j + 1:i].max(initial=0.0) >= EXCURSION:
                hits.append(i)
                break
    return hits


def episode_stats(r):
    acts = [a['executed'] for a in r['actions']]
    m = r['metrics']
    longest, run, last = 0, 0, None
    for a in acts:
        run = run + 1 if a in (LEFT, RIGHT) and a == last else (1 if a in (LEFT, RIGHT) else 0)
        last = a
        longest = max(longest, run)
    rev = revisits(r['trajectory'])
    reversals = sum(1 for a, b in zip(acts, acts[1:]) if {a, b} == {LEFT, RIGHT})
    forward = sum(a == FORWARD for a in acts)
    return dict(success=float(m.get('success', 0)), oracle=float(m.get('oracle_success', 0)),
                steps=len(acts), forward=forward, revisit_steps=len(rev), revisit=bool(rev),
                first_revisit=rev[0] if rev else None, spin=longest >= SPIN, longest_turn_run=longest,
                reversal_rate=reversals / max(1, len(acts)), cap=bool(r.get('forced_stop')),
                passed_goal=m.get('oracle_success', 0) > 0 and not m.get('success', 0))


def rate(rows, key, cond=None):
    rows = [r for r in rows if cond is None or cond(r)]
    return round(float(np.mean([float(r[key]) for r in rows])), 4) if rows else None


def summarize(stats):
    rows = list(stats.values())
    out = dict(n=len(rows), sr=rate(rows, 'success'), osr=rate(rows, 'oracle'),
               mean_steps=round(float(np.mean([r['steps'] for r in rows])), 1))
    for key in ('revisit', 'spin', 'cap', 'passed_goal'):
        out[f'{key}_rate'] = rate(rows, key)
        out[f'sr_given_{key}'] = rate(rows, 'success', lambda r: r[key])
        out[f'sr_given_no_{key}'] = rate(rows, 'success', lambda r: not r[key])
    fails = [r for r in rows if not r['success']]
    out['failures'] = len(fails)
    for key in ('revisit', 'spin', 'cap', 'passed_goal'):
        out[f'share_of_failures_with_{key}'] = rate(fails, key)
    moves = sum(r['forward'] for r in rows)
    out['revisit_steps_per_100_forward'] = round(100 * sum(r['revisit_steps'] for r in rows) / max(1, moves), 2)
    out['reversal_rate'] = round(float(np.mean([r['reversal_rate'] for r in rows])), 4)
    out['never_within_3m_rate'] = round(1 - out['osr'], 4)
    # among never-within-3m failures: share that revisited (lost) vs did not (wrong route, no looping)
    lost = [r for r in rows if not r['oracle']]
    out['never_within_3m_with_revisit'] = rate(lost, 'revisit')
    return out


def paired(a, b):
    ids = sorted(set(a) & set(b))
    gained = [i for i in ids if b[i]['success'] and not a[i]['success']]
    lost = [i for i in ids if a[i]['success'] and not b[i]['success']]
    def share(ids_, stats, key):
        return round(float(np.mean([stats[i][key] for i in ids_])), 4) if ids_ else None
    return dict(n=len(ids), gained=len(gained), lost=len(lost),
                gained_revisit_in_a=share(gained, a, 'revisit'), gained_spin_in_a=share(gained, a, 'spin'),
                gained_passed_goal_in_a=share(gained, a, 'passed_goal'),
                lost_revisit_in_b=share(lost, b, 'revisit'), lost_spin_in_b=share(lost, b, 'spin'),
                lost_passed_goal_in_b=share(lost, b, 'passed_goal'),
                revisit_a=share(ids, a, 'revisit'), revisit_b=share(ids, b, 'revisit'))


def main():
    out_path, specs = sys.argv[1], [s.split('=', 1) for s in sys.argv[2:]]
    stats = {}
    for name, directory in specs:
        stats[name] = {i: episode_stats(r) for i, r in load(directory).items()}
    common = set.intersection(*(set(s) for s in stats.values()))
    stats = {k: {i: v[i] for i in common} for k, v in stats.items()}
    report = dict(params=dict(window=WINDOW, radius=RADIUS, excursion=EXCURSION, spin_turns=SPIN),
                  episodes=len(common), policies={k: summarize(v) for k, v in stats.items()},
                  paired={f'{x}->{y}': paired(stats[x], stats[y]) for x, y in combinations(stats, 2)})
    Path(out_path).write_text(json.dumps(report, indent=1))
    keys = ['sr', 'osr', 'mean_steps', 'revisit_rate', 'sr_given_revisit', 'sr_given_no_revisit',
            'revisit_steps_per_100_forward', 'spin_rate', 'sr_given_spin', 'reversal_rate', 'cap_rate',
            'passed_goal_rate', 'never_within_3m_rate', 'never_within_3m_with_revisit',
            'share_of_failures_with_revisit']
    print('policy'.ljust(14) + ''.join(k[:12].rjust(13) for k in keys))
    for name, s in report['policies'].items():
        print(name.ljust(14) + ''.join(str(s[k]).rjust(13) for k in keys))


if __name__ == '__main__':
    main()
