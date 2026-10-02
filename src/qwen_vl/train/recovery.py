"""Completed distributed checkpoints only; retain epochs and bounded recovery."""
import json
import os
from pathlib import Path
import shutil
import time
from transformers import TrainerCallback

MARKER = 'RECOVERY_COMPLETE.json'


def mark_complete(path, world_size, epoch_boundary):
    path = Path(path)
    required = ['pytorch_model.bin', 'trainer_state.json', 'resume_contract.json', 'navigation.json']
    required += (['rng_state.pth'] if world_size == 1 else
                 [f'rng_state_{r}.pth' for r in range(world_size)])
    if not all((path/name).is_file() and (path/name).stat().st_size for name in required):
        raise ValueError(f'Incomplete model/metadata/RNG checkpoint: {path}')
    standard = all((path/name).is_file() for name in ('optimizer.pt','scheduler.pt'))
    distributed = (len(list(path.rglob('*optim_states.pt'))) == world_size
                   and bool(list(path.rglob('*model_states.pt'))))
    if not (standard or distributed):
        raise ValueError(f'Incomplete optimizer/scheduler checkpoint: {path}')
    files = {str(p.relative_to(path)): p.stat().st_size for p in path.rglob('*')
             if p.is_file() and p.name not in (MARKER, MARKER+'.tmp')}
    if any(p.is_symlink() for p in path.rglob('*')) or not all(files.values()):
        raise ValueError(f'Empty file or symlink in checkpoint: {path}')
    state = json.loads((path/'trainer_state.json').read_text())
    record = dict(step=state['global_step'], epoch_boundary=bool(epoch_boundary),
                  world_size=world_size, files=files)
    temporary = path/(MARKER+'.tmp')
    with temporary.open('w') as f:
        json.dump(record, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path/MARKER)


def validate_complete(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Checkpoint cannot be a symlink')
    record = json.loads((path/MARKER).read_text())
    if path.name != f"checkpoint-{record['step']}":
        raise ValueError('Checkpoint step/path mismatch')
    for name, size in record['files'].items():
        item = path/name
        if Path(name).is_absolute() or '..' in Path(name).parts or item.is_symlink():
            raise ValueError('Unsafe checkpoint inventory')
        if not item.is_file() or item.stat().st_size != size or size <= 0:
            raise ValueError(f'Incomplete/truncated checkpoint file: {item}')
    return record


def completed_checkpoints(root):
    found = []
    for path in Path(root).glob('checkpoint-*'):
        try:
            record = validate_complete(path)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        found.append((record['step'], path, record))
    return sorted(found)


def latest_checkpoint(root):
    checkpoints = completed_checkpoints(root)
    return checkpoints[-1][1] if checkpoints else None


def prune_recovery(root, keep=2):
    if keep < 1:
        raise ValueError('Keep at least one recovery checkpoint')
    rolling = [p for _, p, r in completed_checkpoints(root) if not r['epoch_boundary']]
    for path in rolling[:-keep]:
        # Only direct, completed checkpoints in this run, never legacy/partial saves.
        if path.parent.resolve() != Path(root).resolve() or path.is_symlink():
            raise ValueError('Unsafe checkpoint pruning target')
        shutil.rmtree(path)
        print(f'RECOVERY_PRUNED {path}', flush=True)


def quarantine_incomplete(root):
    complete = {p for _, p, _ in completed_checkpoints(root)}
    for path in Path(root).glob('checkpoint-*'):
        if path not in complete:
            target = Path(root)/'incomplete-checkpoints'/f'{path.name}-{time.time_ns()}'
            target.parent.mkdir(exist_ok=True)
            path.rename(target)
            print(f'RECOVERY_QUARANTINED {path} -> {target}', flush=True)


def rewind_reports(root, step):
    """Archive the failed attempt and exclude replayed updates from reports."""
    root = Path(root)
    archive = root/'recovery-history'/str(time.time_ns())
    for pattern in ('profile_rank*.jsonl','exposures_rank*.jsonl','microtimes_rank*.jsonl','action_metrics.jsonl'):
        for path in root.glob(pattern):
            retained = []
            for line in path.read_text().splitlines():
                try:
                    update = json.loads(line)['update']
                except (ValueError, KeyError):
                    continue  # A kill may interrupt the final JSONL write.
                if update < step if pattern.startswith('exposures') else update <= step:
                    retained.append(line+'\n')
            archive.mkdir(parents=True,exist_ok=True)
            path.rename(archive/path.name)
            path.write_text(''.join(retained))


class RecoverySaves(TrainerCallback):
    def __init__(self, interval):
        if interval < 1:
            raise ValueError('Recovery interval must be positive')
        self.interval = interval

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.interval == 0:
            control.should_save = True
        return control
