"""Diagnostic for H1 'raw event beyond t': logistic fits with absolute and relative position controls."""
import math, sys, json
import numpy as np
sys.path.insert(0, sys.argv[1] + '/scripts/rl')
from analyze_event_forks import load, logistic, spearman, position_bucket
rows = [r for r in load(sys.argv[2], 0, int(sys.argv[3])) if math.isfinite(r['entropy'])]
y = np.asarray([r['informative'] for r in rows])
specs = {
    'event+entropy': ['event', 'entropy'],
    'event+entropy+log1p_t': ['event', 'entropy', 'log1p_t'],
    'event+entropy+t/N*+log1p_t': ['event', 'entropy', 't_over_n', 'log1p_t'],
    'event_pos+entropy+t/N*+log1p_t': ['event_pos', 'entropy', 't_over_n', 'log1p_t'],
    'log1p_t+entropy': ['log1p_t', 'entropy'],
}
feat = dict(event=lambda r: r['event'], event_pos=lambda r: r['event_pos'], entropy=lambda r: r['entropy'],
            log1p_t=lambda r: math.log1p(r['t']), t_over_n=lambda r: r['t'] / r['n_star'])
out = {}
for name, cols in specs.items():
    X = np.asarray([[feat[c](r) for c in cols] for r in rows])
    X = (X - X.mean(0)) / X.std(0)
    w, se = logistic(X, y)
    out[name] = {c: dict(coef=round(float(w[i + 1]), 3), z=round(float(w[i + 1] / se[i + 1]), 2)) for i, c in enumerate(cols)}
by_bucket = {}
for r in rows:
    by_bucket.setdefault(position_bucket(r['t']), []).append(r['informative'])
out['p_informative_by_position_bucket'] = {f'b{b} (t {2**b - 1}-{2**(b + 1) - 2})': [len(v), round(float(np.mean(v)), 3)]
                                           for b, v in sorted(by_bucket.items())}
out['forks_with_entropy'] = len(rows)
print(json.dumps(out, indent=1))
