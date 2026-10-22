"""Memory behaviour of a MemFork run from its training records (no evaluation data).

python scripts/rl/analyze_memfork.py RUN_DIR [--out OUT.json]

Per block of RL steps (main trajectories, all ranks):
  * entropy of the sampled memory decisions, kappa, credited decisions;
  * agreement of the sampled drop with the frame-level M5 prior (argmin of mean_l x_l, ties newest);
  * what is dropped: the newly expired candidate (M5-style rejection) / the oldest frame; age of the
    dropped frame and of the retained frames (decision step - source step);
  * memory forks: how often the branch re-drew a different m_t than the main, and the sibling return
    contrast when it did vs did not (|R_main - R_branch|, success flips).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def prior_drop(features):
    base = np.asarray(features, dtype=np.float64).mean(-1)
    low = np.flatnonzero(base == base.min())
    return int(low.max())


def block_of(step, size):
    return f'{(step // size) * size:03d}-{(step // size) * size + size - 1:03d}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run')
    p.add_argument('--block', type=int, default=25)
    p.add_argument('--out')
    args = p.parse_args()
    run = Path(args.run)
    metrics = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    blocks = defaultdict(lambda: defaultdict(list))
    for m in metrics:
        b = blocks[block_of(m['step'], args.block)]
        for key in ('memory_entropy', 'memory_log_kappa', 'memory_credited_decisions', 'memory_c_abs',
                    'success_rate', 'spl', 'branch_success_rate'):
            if m.get(key) is not None:
                b[key].append(m[key])
    for path in sorted((run / 'rollouts').glob('step_*_rank*.jsonl')):
        step = int(path.name.split('_')[1])
        b = blocks[block_of(step, args.block)]
        for line in path.read_text().splitlines():
            record = json.loads(line)
            decisions = (record.get('memory') or {}).get('decisions', [])
            by_step = {d['step']: d for d in decisions}
            for d in decisions:
                pool = d['pool']
                b['prior_agreement'].append(float(d['drop'] == prior_drop(d['features'])))
                b['candidate_drop'].append(float(d['drop'] == len(pool) - 1))
                b['oldest_drop'].append(float(d['drop'] == 0))
                b['drop_age'].append(d['step'] - pool[d['drop']])
                b['kept_age'] += [d['step'] - s for i, s in enumerate(pool) if i != d['drop']]
            for f in record.get('fork_summary', []):
                main_decision = by_step.get(f['fork_t'])
                drops = dict(map(tuple, f.get('memory_drops') or []))
                if main_decision is None or f['fork_t'] not in drops:
                    continue
                differs = drops[f['fork_t']] != main_decision['pool'][main_decision['drop']]
                key = 'fork_redraw_differs' if differs else 'fork_redraw_same'
                b[key + '_contrast'].append(abs(record['reward'] - f['reward']))
                b[key + '_success_flip'].append(float(record['success'] != f['success']))
                b['fork_redraw_differs'].append(float(differs))
    table = {}
    for name in sorted(blocks):
        table[name] = {k: dict(mean=float(np.mean(v)), n=len(v)) for k, v in sorted(blocks[name].items()) if v}
    text = json.dumps(dict(run=str(run), block=args.block, blocks=table), indent=2)
    if args.out:
        Path(args.out).write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
