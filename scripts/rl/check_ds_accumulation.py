"""G6b: manual-boundary ZeRO-2 accumulation with dummy padding equals the reference gradient.

torchrun --nproc_per_node=2 scripts/rl/check_ds_accumulation.py
Each rank processes a different number of weighted items (padded with zero-loss
dummy micro-steps), as train_grpo does. The full gradient DeepSpeed holds at the
boundary must equal (1/world) * sum over ranks of the locally weighted losses, and
exactly one optimizer step must be taken.
"""
import datetime
import os

import deepspeed
import torch
import torch.distributed as dist
from deepspeed.utils import safe_get_full_grad


def main():
    deepspeed.init_distributed(dist_backend='nccl', timeout=datetime.timedelta(minutes=5))
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)).cuda()
    reference = [p.detach().clone().float() for p in model.parameters()]
    gen = torch.Generator().manual_seed(1)
    data = [[(torch.randn(3, 8, generator=gen), float(w)) for w in ws] for ws in ([0.5, 1.5, 2.0], [0.25])]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    engine, *_ = deepspeed.initialize(model=model, optimizer=optimizer, config={
        'train_micro_batch_size_per_gpu': 1, 'gradient_accumulation_steps': 1, 'train_batch_size': world,
        'bf16': {'enabled': False}, 'zero_allow_untested_optimizer': True,
        'zero_optimization': {'stage': 2, 'overlap_comm': False, 'contiguous_gradients': True,
                              'reduce_bucket_size': 50, 'allgather_bucket_size': 50}})
    items = data[rank]
    n_max = 3
    for i in range(n_max):
        engine.set_gradient_accumulation_boundary(i == n_max - 1)
        if i < len(items):
            x, w = items[i]
            loss = engine.module(x.cuda()).sum() * w
        else:
            loss = engine.module(torch.zeros(1, 8).cuda()).sum() * 0.0
        engine.backward(loss, scale_wrt_gas=False)
    grads = [safe_get_full_grad(p).float().cpu() for p in engine.module.parameters()]
    engine.step()
    assert engine.global_steps == 1, engine.global_steps
    # Single-process reference: average over ranks of local weighted sums.
    ref = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
    with torch.no_grad():
        for p, r in zip(ref.parameters(), reference):
            p.copy_(r.cpu())
    total = sum(ref(x).sum() * w for ws in data for x, w in ws) / world
    total.backward()
    error = max(float((g - p.grad).abs().max() / p.grad.abs().max()) for g, p in zip(grads, ref.parameters()))
    if rank == 0:
        print('max relative gradient error', error, 'global_steps', engine.global_steps)
    assert error < 1e-5, error
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
