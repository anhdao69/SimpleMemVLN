"""Seed-pooled paired comparison of two methods on the same val_unseen episodes.

python scripts/rl/pooled_compare.py OUT.json --a NAME DIR [DIR ...] --b NAME DIR [DIR ...]
Each DIR is a complete evaluator journal (one training seed). Per episode, success and SPL are averaged
over the seeds of each method; the reported effect is mean_B - mean_A with
  * a paired bootstrap 95 % CI over episodes (10,000 resamples),
  * an exact-by-Monte-Carlo paired sign-flip permutation p-value over episodes (100,000 flips),
  * every seed-by-seed delta, so training-seed spread stays visible (2 seeds cannot estimate it).
The test accounts for episode sampling, not for training-seed variance; read it with the seed table.
"""
import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_evals import PROTOCOL_KEYS, episode_metrics, load  # noqa: E402

METRICS = ('success', 'spl', 'osr', 'ne')


def runs(dirs):
    out = []
    for d in dirs:
        summary, contract, rows = load(d)
        out.append((contract, {i: episode_metrics(r) for i, r in rows.items()}))
    return out


def matrix(seeds, ids, key):
    return np.array([[float(m[i][key] or 0.0) for i in ids] for _, m in seeds])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--a', nargs='+', required=True)
    ap.add_argument('--b', nargs='+', required=True)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    name_a, name_b = args.a[0], args.b[0]
    a, b = runs(args.a[1:]), runs(args.b[1:])
    ref = a[0][0]
    for contract, _ in a + b:
        bad = [k for k in PROTOCOL_KEYS if contract.get(k) != ref.get(k)]
        if bad:
            raise SystemExit(f'protocol mismatch: {bad}')
    ids = sorted(ref['episode_ids'])
    rng = np.random.default_rng(args.seed)
    n = len(ids)
    boot = rng.integers(0, n, size=(10_000, n))
    flips = rng.choice([-1.0, 1.0], size=(100_000, n))
    report = dict(a=dict(name=name_a, dirs=args.a[1:]), b=dict(name=name_b, dirs=args.b[1:]), episodes=n,
                  metrics={})
    for key in METRICS:
        ma, mb = matrix(a, ids, key), matrix(b, ids, key)
        d = mb.mean(0) - ma.mean(0)
        entry = dict(a_seeds=[round(float(x), 4) for x in ma.mean(1)],
                     b_seeds=[round(float(x), 4) for x in mb.mean(1)],
                     a_mean=round(float(ma.mean()), 4), b_mean=round(float(mb.mean()), 4),
                     delta=round(float(d.mean()), 4),
                     seed_pair_deltas=[round(float(mb[j].mean() - ma[i].mean()), 4)
                                       for i, j in product(range(len(ma)), range(len(mb)))])
        entry['ci95'] = [round(float(x), 4) for x in np.percentile(d[boot].mean(1), [2.5, 97.5])]
        null = np.abs((flips * d).mean(1))
        entry['p_signflip'] = float((1 + (null >= abs(d.mean()) - 1e-12).sum()) / (1 + len(null)))
        report['metrics'][key] = entry
    Path(args.out).write_text(json.dumps(report, indent=1))
    for key, e in report['metrics'].items():
        print(f"{key:8s} {name_a} {e['a_mean']:.4f} {name_b} {e['b_mean']:.4f} delta {e['delta']:+.4f} "
              f"CI {e['ci95']} p {e['p_signflip']:.4f} seed pairs {e['seed_pair_deltas']}")


if __name__ == '__main__':
    main()
