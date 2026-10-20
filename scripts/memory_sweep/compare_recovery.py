"""Fresh-process recovery check for a GDN-sweep arm (same thresholds as prior M5 runs).

Reference: 3 updates with complete recovery saves.  Resumed: a new process
restores the reference's checkpoint-2 (model, optimizer, scheduler, RNG,
navigation and data contracts) and runs update 3.  Final weights must agree to
relative L2 <= 1e-5 and the update-3 losses to 1e-4; bitwise identity is not
claimed.
"""
import argparse
import json
from pathlib import Path
import torch
from qwen_vl.train.recovery import validate_complete
from scripts.vln.check_vggt_recovery import compare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    root = Path(args.root)
    ref, res = root / 'reference/train', root / 'resumed/train'
    markers = {}
    for path, step in ((ref / 'checkpoint-2', 2), (res / 'checkpoint-3', 3)):
        marker = validate_complete(path)
        if marker['world_size'] != 4 or marker['step'] != step:
            raise ValueError(f'{path}: unexpected recovery checkpoint {marker["step"]}/{marker["world_size"]}')
        markers[str(path)] = dict(step=marker['step'], files=len(marker['files']))
    for path in (ref, res):
        saved = json.loads((path / 'final/navigation.json').read_text())
        if saved['episodic']['version'] != 'episodic_window_gdn_sweep_v1':
            raise ValueError('Recovery check must exercise the sweep contract')
    states = [torch.load(path / 'final/pytorch_model.bin', weights_only=True, map_location='cpu', mmap=True)
              for path in (ref, res)]
    if any(k.startswith('research_aux.') for k in states[0]):
        raise ValueError('Unexpected research tensors')
    weights = compare(states)
    losses = [next(x['loss'] for x in reversed(json.loads((path / 'trainer_state.json').read_text())['log_history'])
                   if x.get('step') == 3 and 'loss' in x) for path in (ref, res)]
    passed = weights['relative_l2'] <= 1e-5 and abs(losses[0] - losses[1]) <= 1e-4
    report = dict(passed=passed, weights=weights, losses=losses, checkpoints=markers)
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
    print('RECOVERY_PASS' if passed else 'RECOVERY_FAIL', json.dumps(report), flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
