"""Offload checkpoint inputs, preserving the complete non-reentrant graph."""
from contextlib import nullcontext
import torch
from torch.utils.checkpoint import checkpoint


def checkpoint_with_cpu_offload(function, *args, min_tokens, **kwargs):
    if kwargs.pop('use_reentrant', False):
        raise ValueError('Activation offload requires non-reentrant checkpointing')
    hidden = args[0]
    context = (torch.autograd.graph.save_on_cpu(pin_memory=hidden.is_cuda)
               if hidden.shape[-2] >= min_tokens else nullcontext())
    # The outer hooks capture checkpoint inputs; checkpoint's inner hooks still
    # discard intermediate activations and recompute them normally on backward.
    with context:
        return checkpoint(function, *args, use_reentrant=False, **kwargs)
