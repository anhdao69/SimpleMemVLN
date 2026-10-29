"""Step-0 drift thresholds for EBF configs (fork.initial_min_drift) from another run's step-0 rollouts on the
same base policy and prompts (the F-B run): exactly the thresholds rl/train_grpo.fork_threshold would compute
at step 1 of that run, pooled (ebf_s8n_r2r1k.yaml) and per position bucket (ebf_pos_s8n_r2r1k.yaml).

Usage: python scripts/rl/calibrate_drift.py RUN_DIR [--quantile 0.5]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from qwen_vl.rl.train_grpo import fork_threshold  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run')
    p.add_argument('--quantile', type=float, default=0.5)
    args = p.parse_args()
    run = Path(args.run)
    if not list((run / 'rollouts').glob('step_0000_rank*.jsonl')):
        raise SystemExit(f'{run} has no step-0 rollouts')
    cfg = dict(drift_quantile=args.quantile, initial_min_drift=None)
    print(json.dumps(dict(run=str(run), quantile=args.quantile,
                          pooled=fork_threshold(run, 1, cfg, signal='drift'),
                          by_position=fork_threshold(run, 1, dict(cfg, drift_by_position=True), signal='drift'))))


if __name__ == '__main__':
    main()
