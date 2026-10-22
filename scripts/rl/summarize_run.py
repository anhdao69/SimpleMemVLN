"""Summarize an RL run's metrics.jsonl (the last record of a step wins: resumed steps are re-run).

python scripts/rl/summarize_run.py RUN_DIR [--window 25] [--out SUMMARY.json]
"""
import argparse
import json
from pathlib import Path

KEYS = ('success_rate', 'oracle_success_rate', 'spl', 'reward_mean', 'mean_steps', 'truncation_rate',
        'format_error_rate', 'all_fail_groups', 'all_success_groups', 'rollout_entropy', 'greedy_agreement',
        'grad_norm', 'logp_mismatch_mean', 'clipfrac', 'step_seconds', 'rollout_seconds', 'train_seconds',
        'train_peak_gib')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run')
    p.add_argument('--window', type=int, default=25)
    p.add_argument('--out')
    args = p.parse_args()
    run = Path(args.run)
    by_step = {}
    lines = (run / 'metrics.jsonl').read_text().splitlines()
    for line in lines:
        record = json.loads(line)
        by_step[record['step']] = record
    steps = [by_step[s] for s in sorted(by_step)]
    if [r['step'] for r in steps] != list(range(len(steps))):
        raise SystemExit('Missing steps in metrics')
    windows = []
    for start in range(0, len(steps), args.window):
        chunk = steps[start:start + args.window]
        windows.append(dict(steps=f'{chunk[0]["step"]}-{chunk[-1]["step"]}',
                            **{k: round(sum(r[k] for r in chunk) / len(chunk), 4) for k in KEYS if k in chunk[0]}))
    totals = dict(steps=len(steps), records=len(lines), rerun_records=len(lines) - len(steps),
                  trajectories=sum(r['trajectories'] for r in steps),
                  discarded_trajectories=sum(r['discarded_trajectories'] for r in steps),
                  env_steps=sum(r['env_steps'] for r in steps),
                  trained_tokens=sum(r['trained_tokens'] for r in steps),
                  wall_hours_sum_of_steps=round(sum(r['step_seconds'] for r in steps) / 3600, 2))
    result = dict(run=str(run), totals=totals, windows=windows)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
