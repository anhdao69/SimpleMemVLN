"""Smoke test for the GPU reservation in rl/placement.py (runtime.hold_gpu_gib).

Reserves 2 GiB on GPU 1 with the trainer's allocator setting, checks that release_cache() keeps it,
that device_free() counts it as usable by this process, that later allocations reuse it, and that
the memory is returned once the reservation is cleared.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from qwen_vl.rl import placement as P  # noqa: E402

dist.init_process_group('gloo', init_method='tcp://127.0.0.1:29533', rank=0, world_size=1)  # start_hold polls in step
device = torch.device('cuda', 1)
torch.cuda.set_device(device)
torch.cuda.memory._set_allocator_settings('expandable_segments:True')  # as train_grpo after helpers start
target = 2 * P.GIB
lines = []

reserved = P.start_hold(device, target, 3 * P.GIB, lines.append, wait_seconds=0)
print(lines[-1])
assert reserved >= target, reserved

P.release_cache()
assert torch.cuda.memory_reserved(device) >= target, 'release_cache() returned the reservation'

driver_free = torch.cuda.mem_get_info(device)[0]
assert P.device_free(device) >= driver_free + target - P.GIB // 4, 'device_free() ignores the reservation'

x = torch.empty(int(1.5 * P.GIB), dtype=torch.uint8, device=device)
assert torch.cuda.memory_reserved(device) <= reserved + 64 * 2 ** 20, 'allocation did not reuse the reservation'
del x
assert P.hold_memory(device) >= target, 'top-up failed'
assert P.wait_for_free(device, P.GIB, print) == 0, 'update gate waited although memory is free'

P.HOLD['target'] = 0
P.release_cache()
assert torch.cuda.memory_reserved(device) < P.GIB // 4, 'memory not returned after clearing the hold'

# A target that cannot be reached must end after wait_seconds instead of blocking the trainer.
began = time.time()
P.start_hold(device, 4 * P.GIB, 10 ** 4 * P.GIB, lines.append, wait_seconds=3, poll_seconds=1)  # reserve > free: no growth
assert time.time() - began < 30, 'start_hold did not stop after wait_seconds'
P.HOLD['target'] = 0
P.release_cache()
dist.destroy_process_group()
print('OK: reservation held, reused and returned; unreachable target gave up in time')
