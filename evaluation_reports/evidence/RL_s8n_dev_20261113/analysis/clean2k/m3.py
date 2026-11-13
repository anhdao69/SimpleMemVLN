import json, glob, collections, statistics as st, sys
from pathlib import Path
RUNS = Path('/nvme/public/svln/runs')
def first_fwd(acts): return next((i for i, a in enumerate(acts) if a == 0), len(acts))
for run in sys.argv[1:]:
    groups = collections.defaultdict(list)
    for f in glob.glob(str(RUNS/run/'rollouts'/'step_*_rank*.jsonl')):
        if int(Path(f).name.split('_')[1]) > 124: continue
        for l in open(f):
            r = json.loads(l); r.pop('memory', None); r.pop('step_drift', None)
            groups[(f, r['episode_id'])].append(r)
    mains = [x for g in groups.values() for x in g]
    # hesitant arrival: steps between first d < 1 m and the end, for successes and failures
    def linger(x):
        i = next((i for i, d in enumerate(x['step_d']) if d is not None and d < 1.0), None)
        return None if i is None else x['num_steps'] - i
    pos = [x for x in mains if x['advantage'] > 0]; neg = [x for x in mains if x['advantage'] < 0]
    def share_linger(xs, k=5):
        v = [linger(x) for x in xs]; v = [y for y in v if y is not None]
        return len(v), sum(y >= k for y in v) / max(1, len(v))
    # advantage-weighted "arrive then keep going": sum adv over mains that were within 1 m and kept going >= 5 steps
    w_l = sum(x['advantage'] for x in mains if (linger(x) or 0) >= 5)
    w_tot = sum(abs(x['advantage']) for x in mains)
    np_, sp = share_linger([x for x in pos if x['success']])
    nn_, sn = share_linger([x for x in neg if x['success']])
    # initial orientation signal: adv-weighted number of turns before the first forward
    ft = sum(x['advantage'] * first_fwd(x['actions']) for x in mains) / w_tot
    print('%-22s reached<1m then >=5 more steps: positive-adv successes %.3f (n %d), negative-adv successes %.3f; net adv on "arrive and keep going" %+.3f of |adv|; adv-weighted initial turns %+.2f' % (
        run, sp, np_, sn, w_l / w_tot, ft))
