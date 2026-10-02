"""Fail-closed gate for the two-GPU Window8 text-policy startup stress test."""
import json
import math
from pathlib import Path
import sys


def verify(root, manifest):
    episodes = [json.loads(s) for s in manifest.read_text().splitlines()]
    assert len(episodes) == 8
    assert all(len(e['steps']) == 627 for e in episodes)
    assert len({e['episode_uid'] for e in episodes}) == 1
    cfg = json.loads((root / 'resolved_config.json').read_text())
    assert cfg['model']['output_mode'] == 'qwen_text'
    assert cfg['memory']['mode'] == 'window8'
    assert cfg['training']['gradient_accumulation_steps'] == 4
    assert cfg['training']['nominal_episodes_per_update'] == 8
    peaks, timings = [], {}
    for rank in range(2):
        rows = [json.loads(s) for s in (root / f'profile_rank{rank}.jsonl').read_text().splitlines()]
        assert [r['update'] for r in rows] == [1, 2, 3]
        peaks.extend(r['peak_reserved_gib'] for r in rows)
        timings[str(rank)] = [r['seconds'] for r in rows]
        assert all(math.isfinite(t) and t > 0 for t in timings[str(rank)])
        exposures = [json.loads(s) for s in (root / f'exposures_rank{rank}.jsonl').read_text().splitlines()]
        # Exposure records use global_step BEFORE the optimizer update.
        assert [e['update'] for e in exposures] == [0]*4 + [1]*4 + [2]*4
        assert all(e['episode_uid'] == episodes[0]['episode_uid'] for e in exposures)
        assert all(math.isfinite(float(e['loss_sum'])) for e in exposures)
    assert all(math.isfinite(p) and 0 < p <= 78 for p in peaks), peaks
    state = json.loads((root / 'trainer_state.json').read_text())
    assert state['global_step'] == 3
    losses = [r for r in state['log_history'] if 'loss' in r]
    assert [r['step'] for r in losses] == [1, 2, 3]
    assert all(math.isfinite(float(r[k])) for r in losses for k in ('loss', 'grad_norm'))
    return dict(passed=True, updates=3, peak_reserved_gib=max(peaks),
                world_size=2, accumulation=4, global_batch=8,
                output_mode='qwen_text', rank_update_seconds=timings,
                losses=[r['loss'] for r in losses], profile_dir=str(root))


if __name__ == '__main__':
    root, manifest = map(Path, sys.argv[1:])
    result = verify(root, manifest)
    with (root / 'PASS.json').open('x') as f:
        json.dump(result, f, indent=2)
    print('TWO_GPU_MEMORY_PASS', json.dumps(result), flush=True)
