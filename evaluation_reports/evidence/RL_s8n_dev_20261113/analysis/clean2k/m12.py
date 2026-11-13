import json, glob, collections, statistics as st, sys
from pathlib import Path
RUNS = Path('/nvme/public/svln/runs')
def corr(x, y):
    if len(x) < 3: return float('nan')
    mx, my = st.mean(x), st.mean(y); sx = sum((a-mx)**2 for a in x) ** .5; sy = sum((b-my)**2 for b in y) ** .5
    return sum((a-mx)*(b-my) for a, b in zip(x, y)) / (sx*sy) if sx and sy else float('nan')
for run in sys.argv[1:]:
    groups = collections.defaultdict(list)
    for f in glob.glob(str(RUNS/run/'rollouts'/'step_*_rank*.jsonl')):
        step = int(Path(f).name.split('_')[1])
        if step > 124: continue
        for l in open(f):
            r = json.loads(l); r.pop('memory', None); r.pop('step_drift', None)
            groups[(f, r['episode_id'])].append(r)
    # M1: all-success groups: what does the advantage rank?
    A, D, S, T = [], [], [], []
    for g in groups.values():
        if all(x['success'] for x in g):
            for x in g:
                A.append(x['advantage']); D.append(x['final_distance']); S.append(x['num_steps']); T.append(sum(a in (1,2) for a in x['actions'])/x['num_steps'])
    sd = [st.pstdev([x['final_distance'] for x in g]) for g in groups.values() if all(x['success'] for x in g)]
    print(run, 'ALL-SUCCESS groups: corr(adv, final_d) %.2f, corr(adv, steps) %.2f, corr(adv, turn_share) %.2f; median within-group sd of stop distance %.2f m; mean final_d %.2f' % (corr(A, D), corr(A, S), corr(A, T), st.median(sd), st.mean(D)))
    # M2: mixed groups: success minus failure, by s
    for s in (1, 2, 3):
        ds, dt, dpath = [], [], []
        for g in groups.values():
            if sum(x['success'] for x in g) != s or len(g) != 4: continue
            su = [x for x in g if x['success']]; fa = [x for x in g if not x['success']]
            m = lambda xs, f: st.mean(f(x) for x in xs)
            ds.append(m(su, lambda x: x['num_steps']) - m(fa, lambda x: x['num_steps']))
            dt.append(m(su, lambda x: sum(a in (1,2) for a in x['actions'])/x['num_steps']) - m(fa, lambda x: sum(a in (1,2) for a in x['actions'])/x['num_steps']))
        if ds: print('   mixed %dof4 (n=%d): success - failure: steps %+.1f, turn share %+.3f' % (s, len(ds), st.mean(ds), st.mean(dt)))
    # where success and failure diverge in mixed groups: first differing action / length
    fd = []
    for g in groups.values():
        su = [x for x in g if x['success']]; fa = [x for x in g if not x['success']]
        if not su or not fa: continue
        for a in su:
            for b in fa:
                k = next((i for i, (p, q) in enumerate(zip(a['actions'], b['actions'])) if p != q), min(len(a['actions']), len(b['actions'])))
                fd.append(k / max(1, len(a['actions'])))
    print('   success vs failure first divergence at %.2f of the success length (median %.2f)' % (st.mean(fd), st.median(fd)))
