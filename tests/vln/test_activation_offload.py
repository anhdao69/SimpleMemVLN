import torch
import pytest
from torch.utils.checkpoint import checkpoint


def test_offloaded_checkpoint_preserves_gradients():
    from qwen_vl.models.activation_offload import checkpoint_with_cpu_offload
    x = torch.linspace(-0.5, 0.5, 64).reshape(8, 8).requires_grad_()
    weight = torch.randn(8, 8, requires_grad=True)
    ref = (x @ weight).sin().square().sum()
    expected = torch.autograd.grad(ref, (x, weight))
    actual = checkpoint_with_cpu_offload(lambda a: (a @ weight).sin(), x, min_tokens=1)
    grads = torch.autograd.grad(actual.square().sum(), (x, weight))
    for a, b in zip(grads, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU memory/offload gate')
def test_offloaded_checkpoint_reduces_gpu_memory_without_changing_gradient():
    from qwen_vl.models.activation_offload import checkpoint_with_cpu_offload
    def run(offload):
        x = torch.linspace(-0.5, 0.5, 4096 * 1024, device='cuda').reshape(4096, 1024).requires_grad_()
        torch.cuda.reset_peak_memory_stats()
        y = x
        for _ in range(8):
            y = (checkpoint_with_cpu_offload(torch.sin, y, min_tokens=1) if offload
                 else checkpoint(torch.sin, y, use_reentrant=False))
        y.square().mean().backward()
        torch.cuda.synchronize()
        return x.grad.cpu(), torch.cuda.max_memory_allocated()
    expected, normal_peak = run(False)
    actual, offload_peak = run(True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert offload_peak < normal_peak - 32 * 2**20
