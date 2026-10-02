"""Window attention must not retain one overlapping KV copy per step."""
import sys
from types import SimpleNamespace

import torch
import pytest
from qwen_vl.stream.window_attention import step_flash_attention


def _cpu_flash(q, k, v, dropout_p, softmax_scale, causal):
    # CPU oracle for FA's bottom-right causal alignment; production stays FA2.
    assert dropout_p == 0 and causal
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    mask = torch.arange(k.shape[-2])[None, :] <= (
        torch.arange(q.shape[-2])[:, None] + k.shape[-2] - q.shape[-2]
    )
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, scale=softmax_scale
    ).transpose(1, 2)


def test_window_saved_storage_and_gradients(monkeypatch):
    monkeypatch.setitem(sys.modules, 'flash_attn', SimpleNamespace(flash_attn_func=_cpu_flash))
    torch.manual_seed(19)
    prefix, count, size = 4, 24, 8
    length = prefix + count * size
    spans = [(prefix + i * size, prefix + (i + 1) * size) for i in range(count)]
    inputs = [torch.randn(1, h, length, 16, dtype=torch.float64, requires_grad=True)
              for h in (4, 2, 2)]
    saved = {}
    def pack(t):
        storage = t.untyped_storage()
        saved[storage.data_ptr()] = storage.nbytes()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        output, _ = step_flash_attention(None, *inputs, None, step_plan=(prefix, spans))
    # Independent token-level dense oracle, including eviction after step eight.
    positions = torch.arange(length)
    step = (positions - prefix).div(size, rounding_mode='floor')
    allowed = (positions[None, :] <= positions[:, None]) & (
        (positions[None, :] < prefix) | (step[None, :] >= step[:, None] - 7)
    )
    q, k, v = inputs
    reference = torch.nn.functional.scaled_dot_product_attention(
        q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), attn_mask=allowed
    ).transpose(1, 2)
    torch.testing.assert_close(output, reference, rtol=1e-10, atol=1e-10)
    weight = torch.randn_like(output)
    actual_grads = torch.autograd.grad((output * weight).sum(), inputs)
    expected_grads = torch.autograd.grad((reference * weight).sum(), inputs)
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)
    input_bytes = sum(t.numel() * t.element_size() for t in inputs)
    assert sum(saved.values()) <= 2 * input_bytes, (sum(saved.values()), input_bytes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Real FA2 memory regression')
def test_window_backward_gpu_memory_and_gradients():
    import qwen_vl.stream.window_attention as window
    from torch.utils.checkpoint import checkpoint

    def run(bounded):
        torch.cuda.empty_cache()
        torch.manual_seed(29)
        prefix, count, size = 64, 64, 256
        length = prefix + count * size
        inputs = [torch.randn(1, h, length, 128, device='cuda', dtype=torch.bfloat16,
                              requires_grad=True) for h in (16, 4, 4)]
        spans = [(prefix + i * size, prefix + (i + 1) * size) for i in range(count)]
        def forward(q, k, v):
            if not bounded:
                # Original implementation: independent overlapping KV copies
                # and full-sequence SliceBackward temporaries for every step.
                from flash_attn import flash_attn_func
                def attend(q, k, v):
                    return flash_attn_func(q.transpose(1, 2).contiguous(),
                                           k.transpose(1, 2).contiguous(),
                                           v.transpose(1, 2).contiguous(), causal=True)
                parts = [attend(q[..., :prefix, :], k[..., :prefix, :], v[..., :prefix, :])]
                for t, (start, end) in enumerate(spans):
                    first = spans[max(0, t - 7)][0]
                    wk = torch.cat((k[..., :prefix, :], k[..., first:end, :]), -2)
                    wv = torch.cat((v[..., :prefix, :], v[..., first:end, :]), -2)
                    parts.append(attend(q[..., start:end, :], wk, wv))
                return torch.cat(parts, 1)
            return window.step_flash_attention(None, q, k, v, None,
                                               step_plan=(prefix, spans))[0]
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        out = checkpoint(forward, *inputs, use_reentrant=False)
        forward_peak = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        # Measure the attention VJP, not FP32 square-loss temporaries shared
        # by both implementations (which can hide the attention peak).
        out.backward(torch.full_like(out, 1.0 / out.numel()))
        torch.cuda.synchronize()
        print(f'WINDOW_PHASE_MEMORY bounded={bounded} baseline={baseline} '
              f'forward_peak={forward_peak} backward_peak={torch.cuda.max_memory_allocated()}')
        return out.detach().cpu(), [x.grad.cpu() for x in inputs], torch.cuda.max_memory_allocated()

    expected, expected_grads, old_peak = run(False)
    actual, actual_grads, new_peak = run(True)
    print(f'WINDOW_MEMORY_BYTES old={old_peak} new={new_peak}')
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for a, b in zip(actual_grads, expected_grads):
        # FA2 BF16 backward is not bitwise deterministic (atomic reductions).
        # The FP64 CPU oracle above separately checks derivatives at 1e-9.
        torch.testing.assert_close(a, b, rtol=0.02, atol=1e-10)
    assert new_peak < 0.85 * old_peak, (old_peak, new_peak)
