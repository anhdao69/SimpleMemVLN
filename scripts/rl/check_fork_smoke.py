"""Pass/fail check of a 2-step smoke run directory (used by smoke_then_queue.sh, chain_m5.sh).

Usage: check_fork_smoke.py RUN_DIR [--no-forks]   (--no-forks for plain GRPO smokes)
"""
import json
import math
import sys
from pathlib import Path

run = Path(sys.argv[1])
expect_forks = '--no-forks' not in sys.argv[2:]
metrics = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
problems = []
if not (run / 'STOPPED').exists():
    problems.append('no STOPPED marker (crashed?)')
if [m['step'] for m in metrics] != [0, 1] or metrics[-1]['global_steps'] != 2:
    problems.append(f"expected 2 optimizer steps, got {[m['step'] for m in metrics]}")
for m in metrics:
    if not math.isfinite(m['grad_norm']):
        problems.append(f"step {m['step']}: non-finite grad norm")
    if m['logp_mismatch_mean'] > 0.02:
        problems.append(f"step {m['step']}: trainer/rollout log-prob mismatch {m['logp_mismatch_mean']:.4f}")
branches = sum(m.get('branches', 0) for m in metrics)
if expect_forks and branches == 0:
    problems.append('no fork happened in the smoke')
records = [json.loads(line) for path in sorted((run / 'rollouts').glob('step_*.jsonl'))
           for line in path.read_text().splitlines()]
summaries = [s for r in records for s in r.get('fork_summary', [])]
if branches and len(summaries) != branches:
    problems.append(f'{branches} branches in metrics but {len(summaries)} in the records')
print(json.dumps(dict(run=str(run), branches=branches, forks=[m.get('forks_per_main') for m in metrics],
                      trained=[m['trained_trajectories'] for m in metrics],
                      mismatch=[round(m['logp_mismatch_mean'], 5) for m in metrics],
                      grad_norm=[m['grad_norm'] for m in metrics],
                      branch_env_steps=[m.get('branch_env_steps') for m in metrics],
                      branch_nonzero=[m.get('branch_nonzero_fraction') for m in metrics],
                      problems=problems)))
sys.exit(1 if problems else 0)
