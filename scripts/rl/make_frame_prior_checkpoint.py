"""M5 checkpoint + the MemFork prior retention policy (greedy frame-level M5 rule, no learned part).

python scripts/rl/make_frame_prior_checkpoint.py M5_CHECKPOINT OUT_DIR

OUT_DIR gets symlinks to every file of M5_CHECKPOINT and ``memory_retention.pt`` with g = 0, so serving
keeps the running top-B/300 whole frames by mean_l log frame-mean committed write, shared by all
attention layers (val_seen pilot: does frame-level retention keep the M5 policy's skill?).
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from qwen_vl.research.episodic_contract import parse_episodic_spec  # noqa: E402
from qwen_vl.stream.retention import RETENTION_FILE, RetentionPolicy  # noqa: E402


def main():
    source, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=False)
    for path in sorted(source.iterdir()):
        (out / path.name).symlink_to(path)
    spec = parse_episodic_spec(json.loads((source / 'navigation.json').read_text())['config'])
    if spec is None or not spec.needs_write_scores:
        raise SystemExit('Needs an M5 (episodic_window_v1, committed-write) checkpoint')
    layers = len(spec.attention_layers)
    policy = RetentionPolicy(layers=layers).double()
    torch.save(policy.export(), out / RETENTION_FILE)
    print(out, 'retention prior over', layers, 'attention layers')


if __name__ == '__main__':
    main()
