"""Exploratory (not pre-registered): does the S8N event score predict success flips at forks?"""
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[1] + '/scripts/rl')
from analyze_event_forks import load, logistic
run, hi = sys.argv[2], int(sys.argv[3])
rows = [r for r in load(run, 0, hi) if math.isfinite(r['entropy'])]
succ = {}
for p in sorted((Path(run) / 'rollouts').glob('step_*_rank*.jsonl')):
    if int(p.name.split('_')[1]) >= hi:
        continue
    for line in p.read_text().splitlines():
        e = json.loads(line)
        for f in e.get('fork_summary') or []:
            succ[(e['uid'], f['fork_t'])] = (float(e['success']), float(f['success']))
for r in rows:
    m, b = succ[(r['uid'], r['t'])]
    r['flip'] = float(m != b)
    r['main_success'] = m
y = np.asarray([r['flip'] for r in rows])
feat = dict(event=lambda r: r['event'], event_pos=lambda r: r['event_pos'], entropy=lambda r: r['entropy'],
            t_over_n=lambda r: r['t'] / r['n_star'], log1p_t=lambda r: math.log1p(r['t']),
            main_success=lambda r: r['main_success'])
out = dict(forks=len(rows), p_flip=round(float(y.mean()), 3))
for name, cols in {'event_pos+entropy+t/N*+log1p_t': ['event_pos', 'entropy', 't_over_n', 'log1p_t'],
                   'event+entropy+t/N*+log1p_t': ['event', 'entropy', 't_over_n', 'log1p_t'],
                   'event_pos+entropy+t/N*+log1p_t+main_success': ['event_pos', 'entropy', 't_over_n', 'log1p_t', 'main_success']}.items():
    X = np.asarray([[feat[c](r) for c in cols] for r in rows])
    X = (X - X.mean(0)) / X.std(0)
    w, se = logistic(X, y)
    out[name] = {c: [round(float(w[i + 1]), 3), round(float(w[i + 1] / se[i + 1]), 2)] for i, c in enumerate(cols)}
for key in ('event_pos', 'event'):
    v = np.asarray([r[key] for r in rows]); edges = np.quantile(v, np.linspace(0, 1, 6))
    q = np.clip(np.searchsorted(edges, v, side='right') - 1, 0, 4)
    out[f'flip_by_{key}_quintile'] = [round(float(y[q == k].mean()), 3) for k in range(5)]
print(json.dumps(out, indent=1))
