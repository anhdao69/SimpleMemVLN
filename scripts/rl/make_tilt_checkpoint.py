"""Inference-only S8N checkpoint with a fixed retention tilt (stream/sweep_retention.py).

Every file of SRC is symlinked unchanged (weights, navigation.json, tokenizer); only
``sweep_retention.json`` is added, so the evaluator runs the same policy with a different frame
retention rule. Used for the retention-sensitivity probe of plans/RL/research/s8n_event_rl_design.md §8.

Usage: python scripts/rl/make_tilt_checkpoint.py SRC DST [--birth B] [--consensus A]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from qwen_vl.stream.sweep_retention import SWEEP_RETENTION_FILE, RetentionTilt, save_sweep_retention  # noqa: E402


def attention_layers(navigation):
    episodic = navigation['episodic']
    if episodic.get('version') != 'episodic_window_gdn_sweep_v1' or episodic.get('score') not in ('S8N', 'S8W'):
        raise SystemExit('A retention tilt needs an S8N/S8W GDN-sweep checkpoint')
    return [int(attention) for _, attention in episodic['scoring_pairs']]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('src')
    p.add_argument('dst')
    p.add_argument('--birth', type=float, default=0.0)
    p.add_argument('--consensus', type=float, default=0.0)
    args = p.parse_args()
    src, dst = Path(args.src).resolve(), Path(args.dst)
    if (src / SWEEP_RETENTION_FILE).exists():
        raise SystemExit(f'{src} already carries a retention rule')
    layers = attention_layers(json.loads((src / 'navigation.json').read_text()))
    dst.mkdir(parents=True)
    for item in sorted(src.iterdir()):
        (dst / item.name).symlink_to(item)
    tilt = RetentionTilt(layers, [args.consensus] * len(layers), [args.birth] * len(layers))
    save_sweep_retention(tilt, dst)
    (dst / 'PROBE.md').write_text(f'Inference-only retention probe of {src}: {json.dumps(tilt.to_dict())}\n')
    print(dst, tilt.sha256())


if __name__ == '__main__':
    main()
