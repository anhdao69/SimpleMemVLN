"""Signal quality vs the prompt's prior pass rate (from the OTHER runs' samples), on each run's own groups."""
import collections, glob, json, statistics as st, sys
from pathlib import Path
A = '/nvme/public/svln/analysis/hard2k'
W = '/tmp/claude-1001/-nvme-public-SimpleMemVLN/dced37af-913d-422f-b0e4-f110c3584dd3/scratchpad/why'
PRIOR = {'r2r': (1.0455, 0.295), 'rxr': (1.5476, 0.3853)}
obs = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))   # eid -> label -> [n, s]
for f in (f'{A}/local_outcomes.tsv', f'{W}/rewind8k_outcomes_s790.tsv', f'{W}/hard_outcomes.tsv'):
    for line in open(f):
        lab, step, eid, n, s, a, b = line.rstrip('\n').split('\t')
        obs[eid][lab][0] += int(n); obs[eid][lab][1] += int(s)
def p_prior(eid, exclude):
    n = sum(v[0] for k, v in obs[eid].items() if k != exclude); s = sum(v[1] for k, v in obs[eid].items() if k != exclude)
    a, b = PRIOR['rxr' if eid.startswith('rxr-') else 'r2r']
    return (a + s) / (a + b + n), n
RUNS = {'mix1k': 'rewind_sfr_mix1k', 'sfr_hard1k': 'rewind_sfr_hard1k', 'mix1k_hard2k': 'rewind_mix1k_hard2k',
        'activevln8k': 'grpo_sfr_mix8k_aplus', 'B': 'rewind_sfr', 'path_mix1k': 'rewind_path_sfr_mix1k'}
BINS = [(0, .5), (.5, .65), (.65, .75), (.75, .85), (.85, .92), (.92, 1.01)]
allrows = []
for lab, run in RUNS.items():
    groups = collections.defaultdict(list)
    for f in glob.glob(f'/nvme/public/svln/runs/{run}/rollouts/step_*_rank?.jsonl'):
        for l in open(f):
            r = json.loads(l)
            if r.get('is_branch') or r.get('twin'): continue
            groups[(f, str(r['episode_id']))].append(dict(success=r['success'], steps=r['num_steps'], d0=r['d0'],
                turns=sum(a in (1, 2) for a in r['actions']) / r['num_steps'], adv=r['advantage'], fd=r['final_distance']))
    for (f, eid), g in groups.items():
        p, n = p_prior(eid, lab)
        if n == 0: continue
        allrows.append((lab, eid, p, n, g))
print('run          bin(p_prior)   groups  succ  all_s  mixed  | in mixed groups: success-failure steps / d0, turn share')
for lab in RUNS:
    for lo, hi in BINS:
        sel = [g for l, e, p, n, g in allrows if l == lab and lo <= p < hi]
        if len(sel) < 20: continue
        mixed = [g for g in sel if 0 < sum(x['success'] for x in g) < len(g)]
        ds = [st.mean(x['steps'] / max(.5, x['d0']) for x in g if x['success']) - st.mean(x['steps'] / max(.5, x['d0']) for x in g if not x['success']) for g in mixed]
        dt = [st.mean(x['turns'] for x in g if x['success']) - st.mean(x['turns'] for x in g if not x['success']) for g in mixed]
        print('%-12s [%.2f,%.2f) %6d  %.2f  %.2f   %.2f   | %+.2f  %+.3f' % (lab, lo, hi, len(sel),
            st.mean(x['success'] for g in sel for x in g), sum(all(x['success'] for x in g) for g in sel) / len(sel),
            len(mixed) / len(sel), st.mean(ds) if ds else 0, st.mean(dt) if dt else 0))
