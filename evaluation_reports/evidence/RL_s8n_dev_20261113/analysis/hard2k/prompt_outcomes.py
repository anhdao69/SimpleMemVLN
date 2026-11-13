"""Per-prompt outcomes of the policy's own samples in RL training rollouts (2026-11-09, hard-2k selection).
Main samples only (no rewind/rescue branches, no twins). The saved group is the last attempt of dynamic sampling:
attempt a > 0 means a all-fail groups of G before it, so samples = G * a + n and successes = s (as prompt_signal.py).
Usage: python -I prompt_outcomes.py OUT.tsv RUN_DIR[:label] ...
Rows: label, step, episode_id, samples, successes, attempt, branch_signal (a rescue branch got credit)"""
import collections, glob, json, os, re, sys
out = sys.argv[1]; rows = []
for arg in sys.argv[2:]:
    d, _, label = arg.partition(':'); label = label or os.path.basename(d.rstrip('/'))
    cfg = json.load(open(f'{d}/run_config.json')); cfg = cfg.get('config', cfg); G = cfg['rollout']['group_size']
    g = collections.defaultdict(list)
    for f in sorted(glob.glob(f'{d}/rollouts/step_*_rank?.jsonl')):
        step = int(re.search(r'step_(\d+)_rank', f).group(1))
        for l in open(f):
            r = json.loads(l)
            if r.get('is_branch') or r.get('twin'): continue
            g[(step, str(r['episode_id']))].append((int(r.get('attempt', 0)), bool(r['success']),
                any(abs(b.get('coefficient', 0.0)) > 0 for b in r.get('fork_summary') or [])))
    for (step, eid), rs in sorted(g.items()):
        a = max(x[0] for x in rs); rs = [x for x in rs if x[0] == a]
        rows.append(f'{label}\t{step}\t{eid}\t{G * a + len(rs)}\t{sum(x[1] for x in rs)}\t{a}\t{int(any(x[2] for x in rs))}')
    print(label, len(g), 'groups, G', G, file=sys.stderr)
open(out, 'w').write('\n'.join(rows) + '\n')
