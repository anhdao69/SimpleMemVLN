"""Offload checkpoint inputs, preserving the complete non-reentrant graph.

Backends (engineering only; the saved values and every gradient are identical):

* ``save_on_cpu_pinned`` -- original behavior, ``torch.autograd.graph.save_on_cpu``
  with pinned buffers.  Pinned buffers come from PyTorch's caching host
  allocator, which rounds each request up to a power of two and never returns
  cached blocks; with varying episode lengths the cache grows well beyond one
  episode's need.
* ``pinned_arena_v1`` -- one fixed, page-locked arena per process, sized from the
  audited corpus maximum, ``mmap``-allocated with ``MADV_DONTFORK`` and registered
  with ``cudaHostRegister`` (no power-of-two rounding, no allocator cache, never
  duplicated into forked loader workers).  A bump allocator serves all
  saved checkpoint inputs of the current episode and is rewound only after
  every packed input has been consumed by backward; exhaustion fails loudly.
* ``pageable_v1`` -- ``save_on_cpu(pin_memory=False)``; slower synchronous copies,
  memory returned to the OS after backward.
"""
from contextlib import nullcontext
import threading
import torch
from torch.utils.checkpoint import checkpoint

_ALIGN = 4096


class PinnedActivationArena:
    def __init__(self, capacity_bytes):
        if capacity_bytes <= 0:
            raise ValueError('Arena capacity must be positive')
        import mmap
        self.capacity = (int(capacity_bytes) + _ALIGN - 1) // _ALIGN * _ALIGN
        # Page-aligned anonymous mapping excluded from fork(): on this kernel a fork
        # after cudaHostRegister eagerly copies pinned pages into the child (measured
        # +4 GiB per 4 GiB arena), so every DataLoader worker would duplicate it.
        self._mapping = mmap.mmap(-1, self.capacity, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        self._mapping.madvise(mmap.MADV_DONTFORK)
        self.buffer = torch.frombuffer(self._mapping, dtype=torch.uint8)
        status = torch.cuda.cudart().cudaHostRegister(self.buffer.data_ptr(), self.capacity, 0)
        if int(status) != 0:
            raise RuntimeError(f'cudaHostRegister failed with status {status}')
        self.cursor = self.live = self.peak = 0
        self.lock = threading.Lock()

    def pack(self, tensor):
        nbytes = tensor.numel() * tensor.element_size()
        with self.lock:
            offset = self.cursor
            end = offset + (nbytes + _ALIGN - 1) // _ALIGN * _ALIGN
            if end > self.capacity:
                raise RuntimeError(f'Pinned activation arena exhausted: need {end} > {self.capacity} bytes')
            self.cursor, self.live = end, self.live + 1
            self.peak = max(self.peak, end)
        view = self.buffer[offset:offset + nbytes].view(tensor.dtype).view(tensor.shape)
        view.copy_(tensor.detach(), non_blocking=True)
        return _Packed(self, view, tensor.device)

    def release(self):
        with self.lock:
            self.live -= 1
            if self.live < 0:
                raise RuntimeError('Arena packed tensor released twice')
            if self.live == 0:
                self.cursor = 0     # every input of this episode has been consumed

    def close(self):
        """Unregister before unmapping; a registered range must never be unmapped."""
        if self.buffer is not None:
            if self.live:
                raise RuntimeError('Cannot close an arena with live packed inputs')
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaHostUnregister(self.buffer.data_ptr())
            self.buffer = None
            self._mapping.close()

    def stats(self):
        return dict(arena_capacity_gib=self.capacity / 2**30, arena_peak_gib=self.peak / 2**30,
                    arena_live=self.live)


class _Packed:
    __slots__ = ('arena', 'view', 'device', 'consumed')

    def __init__(self, arena, view, device):
        self.arena, self.view, self.device, self.consumed = arena, view, device, False

    def unpack(self):
        # Stream-ordered H2D read precedes any later reuse of this region.
        result = self.view.to(self.device, non_blocking=True)
        if not self.consumed:
            self.consumed = True
            self.arena.release()
        return result

    def __del__(self):
        if not self.consumed:
            self.consumed = True
            try:
                self.arena.release()
            except Exception:
                pass


_ARENA = None


def install_activation_arena(capacity_bytes):
    global _ARENA
    if _ARENA is not None:
        if _ARENA.capacity < capacity_bytes:
            raise RuntimeError('Activation arena already installed with smaller capacity')
        return _ARENA
    _ARENA = PinnedActivationArena(capacity_bytes)
    return _ARENA


def activation_arena():
    return _ARENA


def offload_context(hidden, min_tokens, backend='save_on_cpu_pinned'):
    if hidden.shape[-2] < min_tokens:
        return nullcontext()
    if backend == 'save_on_cpu_pinned':
        return torch.autograd.graph.save_on_cpu(pin_memory=hidden.is_cuda)
    if backend == 'pageable_v1':
        return torch.autograd.graph.save_on_cpu(pin_memory=False)
    if backend == 'pinned_arena_v1':
        if _ARENA is None:
            raise RuntimeError('pinned_arena_v1 requires install_activation_arena before training')
        return torch.autograd.graph.saved_tensors_hooks(_ARENA.pack, _Packed.unpack)
    raise ValueError(f'Unknown activation offload backend {backend!r}')


def checkpoint_with_cpu_offload(function, *args, min_tokens, backend='save_on_cpu_pinned', **kwargs):
    if kwargs.pop('use_reentrant', False):
        raise ValueError('Activation offload requires non-reentrant checkpointing')
    hidden = args[0]
    context = offload_context(hidden, min_tokens, backend)
    # The outer hooks capture checkpoint inputs; checkpoint's inner hooks still
    # discard intermediate activations and recompute them normally on backward.
    with context:
        return checkpoint(function, *args, use_reentrant=False, **kwargs)


def arena_bytes_for(max_tokens, hidden_size, num_layers, dtype_bytes=2):
    """Exact bound for one episode: one [T, width] checkpoint input per decoder layer."""
    per_layer = (max_tokens * hidden_size * dtype_bytes + _ALIGN - 1) // _ALIGN * _ALIGN
    return per_layer * num_layers
