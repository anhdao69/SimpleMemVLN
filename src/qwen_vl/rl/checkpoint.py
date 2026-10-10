"""Per-rank resumable ZeRO-2 state that never materializes other ranks' partitions.

DeepSpeed's own ZeRO-2 resume reads every rank's optimizer file and rebuilds a full fp32
copy on the host for each rank (≈17 GB each for 4.2 B parameters), which this 62 GB shared
host cannot hold. Here each rank saves and memory-maps only its own fp32 master partition
and AdamW moments; the bf16 model weights are rebuilt from the restored fp32 masters by
DeepSpeed's ``update_lp_params`` (copy + all-gather), so a resume is exact.
"""
from pathlib import Path

import torch


def _partitions(engine):
    opt = engine.optimizer
    fp32 = opt.single_partition_of_fp32_groups
    for group, tensor in zip(opt.optimizer.param_groups, fp32):
        if len(group['params']) != 1 or group['params'][0] is not tensor:
            raise RuntimeError('Unexpected ZeRO-2 optimizer layout')
    return opt, fp32


def save_rank_state(engine, directory, rank, world):
    opt, fp32 = _partitions(engine)
    state = dict(
        world=world, rank=rank, global_steps=engine.global_steps,
        global_samples=engine.global_samples, micro_steps=engine.micro_steps,
        fp32=[t.detach() for t in fp32],
        adam=[dict(opt.optimizer.state[t]) for t in fp32],
        param_groups=[{k: v for k, v in g.items() if k != 'params'} for g in opt.optimizer.param_groups])
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / f'rank{rank}.pt.tmp'
    torch.save(state, tmp)
    tmp.replace(directory / f'rank{rank}.pt')


@torch.no_grad()
def load_rank_state(engine, directory, rank, world):
    opt, fp32 = _partitions(engine)
    state = torch.load(Path(directory) / f'rank{rank}.pt', map_location='cpu', mmap=True, weights_only=False)
    if (state['world'], state['rank']) != (world, rank):
        raise ValueError('Resumable state was written for another rank layout')
    for tensor, saved, adam, group, saved_group in zip(fp32, state['fp32'], state['adam'],
                                                       opt.optimizer.param_groups, state['param_groups']):
        if tensor.shape != saved.shape:
            raise ValueError('Partition shape changed')
        tensor.copy_(saved)
        opt.optimizer.state[tensor] = {k: (v.to(tensor.device) if torch.is_tensor(v) and v.dim() else
                                           v.clone() if torch.is_tensor(v) else v) for k, v in adam.items()}
        group.update(saved_group)
    opt.update_lp_params()
    engine.global_steps = state['global_steps']
    engine.global_samples = state['global_samples']
    engine.micro_steps = state['micro_steps']
    del state
