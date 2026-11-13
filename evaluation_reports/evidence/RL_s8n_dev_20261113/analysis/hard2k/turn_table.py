"""Turn counts of the mains (no branches, no twins) per R2R episode of an RL run (drift-guard reference table).
Usage: python -I turn_table.py RUN_DIR OUT.tsv   -> rows: episode_id, step, turns, actions, success"""
import glob, json, re, sys
rows = []
for f in sorted(glob.glob(sys.argv[1] + '/rollouts/step_*_rank?.jsonl')):
    step = int(re.search(r'step_(\d+)_rank', f).group(1))
    for l in open(f):
        r = json.loads(l)
        if r.get('is_branch') or r.get('twin') or str(r['episode_id']).startswith('rxr-'): continue
        a = r['actions']; rows.append(f"{r['episode_id']}\t{step}\t{sum(x in (1, 2) for x in a)}\t{len(a)}\t{int(bool(r['success']))}")
open(sys.argv[2], 'w').write('\n'.join(rows) + '\n')
