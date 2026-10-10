"""Resume gate: per-rank ZeRO-2 save/load reproduces the uninterrupted run exactly.

torchrun --nproc_per_node=2 scripts/rl/check_resume.py OUT_DIR
Phase A trains 3 steps and records parameters after each; it saves after step 2. Phase B
builds a fresh engine, loads the step-2 state and runs step 3. Parameters after step 3 must
be bitwise identical to phase A.
"""
import datetime
import os
import sys

import deepspeed
import torch
import torch.distributed as dist

from qwen_vl.rl.checkpoint import load_rank_state, save_rank_state


def engine_for():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.Tanh(), torch.nn.Linear(64, 4)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
    engine, *_ = deepspeed.initialize(model=model, optimizer=opt, config={
        'train_micro_batch_size_per_gpu': 1, 'gradient_accumulation_steps': 1, 'train_batch_size': 2,
        'gradient_clipping': 1.0, 'bf16': {'enabled': True}, 'zero_allow_untested_optimizer': True,
        'zero_optimization': {'stage': 2, 'overlap_comm': False, 'contiguous_gradients': True,
                              'reduce_bucket_size': 100, 'allgather_bucket_size': 100}})
    return engine


def train_step(engine, step, rank):
    gen = torch.Generator().manual_seed(100 * step + rank)
    x = torch.randn(5, 16, generator=gen).cuda().bfloat16()
    engine.set_gradient_accumulation_boundary(True)
    engine.backward(engine.module(x).float().pow(2).mean(), scale_wrt_gas=False)
    engine.step()


def snapshot(engine):
    return [p.detach().float().cpu().clone() for p in engine.module.parameters()]


def main():
    out = sys.argv[1]
    deepspeed.init_distributed(dist_backend='nccl', timeout=datetime.timedelta(minutes=5))
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    a = engine_for()
    for step in range(3):
        train_step(a, step, rank)
        if step == 1:
            save_rank_state(a, out, rank, world)
            dist.barrier()
    reference = snapshot(a)
    fp32_ref = [t.detach().cpu().clone() for t in a.optimizer.single_partition_of_fp32_groups]
    b = engine_for()
    load_rank_state(b, out, rank, world)
    assert b.global_steps == 2
    train_step(b, 2, rank)
    got = snapshot(b)
    fp32_got = [t.detach().cpu() for t in b.optimizer.single_partition_of_fp32_groups]
    exact = all(torch.equal(x, y) for x, y in zip(reference, got)) and \
        all(torch.equal(x, y) for x, y in zip(fp32_ref, fp32_got))
    moved = any(not torch.equal(x, y) for x, y in zip(snapshot(engine_for()), got))
    print(f'rank{rank} resume exact={exact} params_moved={moved} global_steps={b.global_steps}', flush=True)
    assert exact and moved and b.global_steps == 3
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
