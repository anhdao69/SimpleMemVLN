"""Gate: rl.placement.redistribute moves episodes between ranks without changing them.
torchrun --nproc_per_node=2 scripts/rl/check_placement.py"""
import datetime
import os

import numpy as np
import torch
import torch.distributed as dist

import qwen_vl.rl.placement as placement


def main():
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=5))
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    rng = np.random.default_rng(rank)
    items = []
    for i, n in enumerate([5, 3] if rank == 0 else [7, 2, 4]):
        frames = [rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(n)]
        items.append((dict(_frames=frames, _responses=[dict(tokens=[1, 2], complete=True)] * n,
                           advantage=float(rank * 10 + i), instruction=f'r{rank}i{i}',
                           checksum=int(sum(int(f.sum()) for f in frames))), 0.1 * (rank + 1) + i))
    # Force moves: pretend rank 1 has almost no free memory.
    placement.torch.cuda.mem_get_info = lambda d: ((80 if rank == 0 else 1) * 2 ** 30, 0)
    kept, stats = placement.redistribute(items, device)
    ok = all(int(sum(int(f.sum()) for f in e['_frames'])) == e['checksum'] and len(e['_frames']) == len(e['_responses'])
             for e, _ in kept)
    everything = [None, None]
    dist.all_gather_object(everything, sorted((e['instruction'], e['advantage'], round(w, 6)) for e, w in kept))
    total = sorted(everything[0] + everything[1])
    expected = sorted([('r0i0', 0.0, 0.1), ('r0i1', 1.0, 1.1), ('r1i0', 10.0, 0.2), ('r1i1', 11.0, 1.2), ('r1i2', 12.0, 2.2)])
    print(f'rank{rank} kept={len(kept)} moved={stats["moved"]} frames_intact={ok} union_ok={total == expected}', flush=True)
    assert ok and total == expected and stats['moved'] > 0
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
