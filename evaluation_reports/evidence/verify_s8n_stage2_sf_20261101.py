"""Recompute the S8N stage-2 STOP-fix R2R results from the per-episode CSV and check them against the JSON evidence.

Every number in S8N_STAGE2_STOPFIX_R2R_EVALUATION.md and in the stage-2 rows of ALL_RESULTS.md comes from these two
files. Standard library only:

    python3 verify_s8n_stage2_sf_20261101.py                 # check the evidence next to this script
    python3 verify_s8n_stage2_sf_20261101.py --write CSV OUT # (re)build the computed part of the JSON from a CSV

Per model: SR, SPL, NE, OSR, TL, forced stops, mean steps and SR/OSR. Per pair (same 1,839 episodes): discordant
successes with an exact two-sided McNemar (binomial) p-value, and paired bootstrap intervals for the SR and SPL
differences (10,000 resamples of episodes, seed 0, percentile method) at 95% and at the Bonferroni level for the six
comparisons (1 - 0.05/6).
"""
import csv
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / 'S8N_STAGE2_SF_R2R_20261101_EPISODES.csv'
JSON_PATH = HERE / 'S8N_STAGE2_SF_R2R_20261101.json'
MODELS = ('S8N', 'S8N_DAGGER_SF', 'S8N_DAGGER_SF_REMAINING', 'S8N_DAGGER_SF_FULL')
PAIRS = (('S8N_DAGGER_SF', 'S8N'), ('S8N_DAGGER_SF_REMAINING', 'S8N'), ('S8N_DAGGER_SF_FULL', 'S8N'),
         ('S8N_DAGGER_SF_REMAINING', 'S8N_DAGGER_SF'), ('S8N_DAGGER_SF_FULL', 'S8N_DAGGER_SF'),
         ('S8N_DAGGER_SF_REMAINING', 'S8N_DAGGER_SF_FULL'))
FLOATS = ('success', 'spl', 'distance_to_goal', 'oracle_success', 'trajectory_length')
INTS = ('forced_stop', 'steps')
ROUNDS = 10000
ADJUSTED_ALPHA = 0.05 / len(PAIRS)


def load(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    ids = [r['episode_id'] for r in rows]
    assert len(ids) == len(set(ids)) == 1839, 'expected 1,839 unique R2R val_unseen episodes'
    episodes = {m: [dict({k: float(r[f'{m}.{k}']) for k in FLOATS}, **{k: int(r[f'{m}.{k}']) for k in INTS}) for r in rows]
                for m in MODELS}
    for m in MODELS:
        assert all(e['success'] in (0.0, 1.0) and e['oracle_success'] in (0.0, 1.0) and e['forced_stop'] in (0, 1)
                   for e in episodes[m]), m
    return ids, episodes


def mean(xs):
    return sum(xs) / len(xs)


def summary(e):
    successes = sum(int(x['success']) for x in e)
    out = dict(episodes=len(e), successes=successes, sr=100 * mean([x['success'] for x in e]),
               spl=100 * mean([x['spl'] for x in e]), ne=mean([x['distance_to_goal'] for x in e]),
               osr=100 * mean([x['oracle_success'] for x in e]), tl=mean([x['trajectory_length'] for x in e]),
               forced_stops=sum(x['forced_stop'] for x in e), mean_steps=mean([x['steps'] for x in e]))
    out['sr_over_osr'] = out['sr'] / out['osr']
    return out


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def bootstrap(ref, new, key):
    """Sorted bootstrap means of the paired difference new - ref (percentage points)."""
    rng = random.Random(0)
    n = len(ref)
    diffs = [new[i][key] - ref[i][key] for i in range(n)]
    return sorted(100 * sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(ROUNDS))


def interval(stats, alpha):
    return [stats[int(alpha / 2 * ROUNDS)], stats[int((1 - alpha / 2) * ROUNDS) - 1]]


def pair(ref, new):
    both = sum(x['success'] == 1.0 and y['success'] == 1.0 for x, y in zip(ref, new))
    new_only = sum(x['success'] == 0.0 and y['success'] == 1.0 for x, y in zip(ref, new))
    ref_only = sum(x['success'] == 1.0 and y['success'] == 0.0 for x, y in zip(ref, new))
    sr, spl = bootstrap(ref, new, 'success'), bootstrap(ref, new, 'spl')
    p = mcnemar(new_only, ref_only)
    return dict(sr_diff=100 * (mean([y['success'] for y in new]) - mean([x['success'] for x in ref])),
                spl_diff=100 * (mean([y['spl'] for y in new]) - mean([x['spl'] for x in ref])),
                both_success=both, new_only=new_only, reference_only=ref_only,
                both_failure=len(ref) - both - new_only - ref_only, mcnemar_p=p,
                mcnemar_p_bonferroni=min(1.0, p * len(PAIRS)),
                sr_diff_ci95=interval(sr, 0.05), spl_diff_ci95=interval(spl, 0.05),
                sr_diff_ci_bonferroni=interval(sr, ADJUSTED_ALPHA), spl_diff_ci_bonferroni=interval(spl, ADJUSTED_ALPHA))


def compute(path):
    ids, episodes = load(path)
    return dict(episodes=len(ids), models={m: summary(episodes[m]) for m in MODELS},
                pairs={f'{new} vs {ref}': pair(episodes[ref], episodes[new]) for new, ref in PAIRS},
                bootstrap=dict(rounds=ROUNDS, seed=0, method='percentile, paired resampling of episodes',
                               bonferroni_alpha=ADJUSTED_ALPHA, comparisons=len(PAIRS)))


def same(a, b, where=''):
    if isinstance(a, dict):
        assert set(a) == set(b), (where, sorted(set(a) ^ set(b)))
        for k in a:
            same(a[k], b[k], f'{where}.{k}')
    elif isinstance(a, list):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f'{where}[{i}]')
    elif isinstance(a, float) or isinstance(b, float):
        assert math.isclose(a, b, rel_tol=0, abs_tol=1e-9), (where, a, b)
    else:
        assert a == b, (where, a, b)


def main():
    if sys.argv[1:2] == ['--write']:
        Path(sys.argv[3]).write_text(json.dumps(compute(sys.argv[2]), indent=1) + '\n')
        return
    evidence = json.loads(JSON_PATH.read_text())
    computed = compute(CSV_PATH)
    same(computed, evidence['computed'], 'computed')
    for m in MODELS:
        final = evidence['models'][m]['final_verified']
        got = computed['models'][m]
        assert got['successes'] == final['successes'] and final['failures'] == 0, m
        assert math.isclose(got['sr'], final['sr_percent'], rel_tol=0, abs_tol=1e-9), m
        assert math.isclose(got['spl'], final['spl_percent'], rel_tol=0, abs_tol=1e-9), m
    print(json.dumps(dict(passed=True, episodes=computed['episodes'],
                          models={m: dict(successes=v['successes'], sr=round(v['sr'], 6), spl=round(v['spl'], 6))
                                  for m, v in computed['models'].items()})))


if __name__ == '__main__':
    main()
