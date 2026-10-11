"""Pinned activation arena: exact round trip, rewind, fork safety (H100)."""
import multiprocessing as mp
import time
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Pinned host memory requires CUDA')


def _child(q):
    from qwen_vl.train.host_memory import process_rss_gib
    q.put(process_rss_gib())


def test_arena_round_trip_rewind_and_fork_does_not_copy():
    from qwen_vl.models.activation_offload import PinnedActivationArena
    arena = PinnedActivationArena(3 * 2**30)
    assert arena.buffer.is_pinned()
    xs = [torch.randn(1, 4096, 2560, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    packed = [arena.pack(x) for x in xs]
    assert arena.live == 3 and arena.cursor > 0
    for x, p in zip(xs, packed):
        assert torch.equal(p.unpack(), x)
    assert arena.live == 0 and arena.cursor == 0           # rewound only after all consumed
    with pytest.raises(RuntimeError, match='exhausted'):
        arena.pack(torch.empty(3 * 2**30 + 1, dtype=torch.uint8, device='cuda'))
    arena.buffer.fill_(1)                                  # touch every pinned page
    q = mp.get_context('fork').Queue()
    child = mp.get_context('fork').Process(target=_child, args=(q,))
    child.start(); rss = q.get(); child.join()
    del packed, xs
    arena.close()
    assert rss < 2.0, f'forked worker inherited the arena ({rss:.2f} GiB RSS)'
