"""Per-episode failure analysis of two complete evaluator journals on the same val_unseen episodes.

python scripts/rl/analyze_failures.py BASE_DIR NEW_DIR DATASET_GZ GT_GZ OUT.json
Bins episodes by reference length (GT actions incl. STOP) and start-goal geodesic distance, and
reports success, stop distance, path length, action mix and STOP behaviour for both policies.
"""
import gzip
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path


def load(directory):
    return {r['episode_id']: r for r in map(json.loads, (Path(directory) / 'episodes.jsonl').read_text().splitlines())}


def describe(rows):
    n = len(rows)
    ok = [r for r in rows if not r['failed']]
    stopped = [r for r in ok if not r.get('forced_stop')]
    d = [r['metrics']['distance_to_goal'] for r in ok if 'distance_to_goal' in r['metrics']]
    acts = Counter(a['executed'] for r in ok for a in r['actions'])
    total = sum(acts.values()) or 1
    return dict(n=n, sr=round(sum(r['metrics'].get('success', 0) for r in ok) / n, 4),
                spl=round(sum(r['metrics'].get('spl', 0) for r in ok) / n, 4),
                osr=round(sum(r['metrics'].get('oracle_success', 0) for r in ok) / n, 4),
                ne_median=round(st.median(d), 3) if d else None,
                actions_mean=round(sum(len(r['actions']) for r in ok) / max(1, len(ok)), 1),
                forced_stop=sum(bool(r.get('forced_stop')) for r in ok),
                stop_within_3_to_5m=sum(3.0 <= r['metrics'].get('distance_to_goal', 99) < 5.0 for r in stopped),
                early_stop_d_gt_5=sum(r['metrics'].get('distance_to_goal', 0) >= 5.0 for r in stopped),
                action_share={k: round(v / total, 4) for k, v in sorted(acts.items())})


def main():
    base, new, data_gz, gt_gz, out = sys.argv[1:6]
    b, n = load(base), load(new)
    ids = sorted(b)
    if set(ids) != set(n):
        raise SystemExit('Journals cover different episodes')
    with gzip.open(data_gz, 'rt') as f:
        episodes = {str(e['episode_id']): e for e in json.load(f)['episodes']}
    with gzip.open(gt_gz, 'rt') as f:
        gt = json.load(f)
    length = {i: len(gt[i]['actions']) for i in ids}
    geo = {i: episodes[i]['info']['geodesic_distance'] for i in ids}
    result = dict(all=dict(base=describe([b[i] for i in ids]), new=describe([n[i] for i in ids])))
    for name, key, edges in (('gt_actions', length, (0, 40, 60, 80, 1000)),
                             ('geodesic_m', geo, (0, 7, 10, 13, 100))):
        bins = {}
        for lo, hi in zip(edges, edges[1:]):
            sel = [i for i in ids if lo <= key[i] < hi]
            bins[f'[{lo},{hi})'] = dict(base=describe([b[i] for i in sel]), new=describe([n[i] for i in sel]))
        result[name] = bins
    flips = dict(gained=[i for i in ids if n[i]['metrics'].get('success') and not b[i]['metrics'].get('success')],
                 lost=[i for i in ids if b[i]['metrics'].get('success') and not n[i]['metrics'].get('success')])
    lost_reason = Counter('forced_stop' if n[i].get('forced_stop') else
                          'passed_goal' if n[i]['metrics'].get('oracle_success') else
                          'near_miss' if n[i]['metrics'].get('distance_to_goal', 99) < 5 else 'far' for i in flips['lost'])
    gained_from = Counter('forced_stop' if b[i].get('forced_stop') else
                          'passed_goal' if b[i]['metrics'].get('oracle_success') else
                          'near_miss' if b[i]['metrics'].get('distance_to_goal', 99) < 5 else 'far' for i in flips['gained'])
    result['flips'] = dict(gained=len(flips['gained']), lost=len(flips['lost']),
                           base_outcome_of_gained=dict(gained_from), new_outcome_of_lost=dict(lost_reason))
    Path(out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
