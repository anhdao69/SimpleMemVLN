"""Episodic gathered-window attention versus an independent FP64 dense-mask oracle."""
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from qwen_vl.data.episodic_metadata import EpisodicLayout
from qwen_vl.research.episodic_contract import EpisodicSpec, QWEN35_4B_PAIRS
from qwen_vl.stream.episodic import build_layer_plan
from qwen_vl.stream.window_attention import step_flash_attention
from test_window_memory import _cpu_flash
from test_window_batched import _cpu_varlen


def _cpu_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, 'flash_attn', SimpleNamespace(
        flash_attn_varlen_func=_cpu_varlen, flash_attn_func=_cpu_flash))


def _layout(sizes, prefix=5, visual=3):
    """Tiny geometry: each group has `visual` image rows starting one token in."""
    spans, obs, starts, cursor = [], [], [], prefix
    for size in sizes:
        spans.append((cursor, cursor + size)); starts.append(cursor + 1)
        obs.append(cursor + 1 + visual); cursor += size
    return spans, starts, cursor


def _plan(layer, recent, budget, spans, starts, visual, seed):
    """Random fixed priorities over the tiny visual rows, via the real plan builder."""
    layout = EpisodicLayout(5, tuple(spans), tuple(s + visual for s in starts), tuple(starts), visual)
    rng = np.random.default_rng(seed)
    scores = torch.tensor(rng.random((len(spans), visual)), dtype=torch.float32)
    spec = EpisodicSpec('episodic_window_v1', recent, budget, 'token', 'committed_mean_l2_v1',
                        QWEN35_4B_PAIRS, 0, None, 'b', 's', 'a')
    return build_layer_plan(spec, 'ep', layer, layout, scores)


def _dense_mask(prefix, spans, recent, plan, length):
    mask = torch.zeros(length, length, dtype=torch.bool)
    for i in range(prefix):
        mask[i, :i + 1] = True
    for t, (start, end) in enumerate(spans):
        first = spans[max(0, t - recent + 1)][0]
        for q in range(start, end):
            mask[q, :prefix] = True
            mask[q, first:q + 1] = True
            if plan is not None:
                mask[q, torch.as_tensor(plan.group_ids(t))] = True
    return mask


SIZES = [5, 7, 6, 9, 5, 8, 8, 5, 6, 5, 7, 6, 9, 5, 6, 5, 8]


@pytest.mark.parametrize('recent,budget,batch', [(8, 0, 1), (8, 0, 5), (9, 0, 16), (8, 6, 4),
                                                 (4, 12, 3), (3, 5, 64), (2, 30, 7)])
def test_episodic_matches_dense_oracle_values_and_gradients(monkeypatch, recent, budget, batch):
    _cpu_backend(monkeypatch)
    torch.manual_seed(budget + recent)
    spans, starts, length = _layout(SIZES)
    plan = _plan(7, recent, budget, spans, starts, 3, recent) if budget else None
    xs = [torch.randn(1, h, length, 8, dtype=torch.float64, requires_grad=True) for h in (4, 2, 2)]
    stored = {}

    def save(t):
        stored[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
        return t
    with torch.autograd.graph.saved_tensors_hooks(save, lambda t: t):
        actual = step_flash_attention(SimpleNamespace(layer_idx=7), *xs, None, scaling=8 ** -.5,
                                      step_plan=(5, spans), window_attention_batch_steps=batch,
                                      episodic_recent_groups=recent, episodic_plan=plan)[0]
    q, k, v = xs
    mask = _dense_mask(5, spans, recent, plan, length)
    reference = torch.nn.functional.scaled_dot_product_attention(
        q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), attn_mask=mask).transpose(1, 2)
    torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-10)
    weight = torch.randn_like(actual)
    got = torch.autograd.grad((actual * weight).sum(), xs, retain_graph=True)
    want = torch.autograd.grad((reference * weight).sum(), xs)
    for a, e in zip(got, want):
        torch.testing.assert_close(a, e, rtol=1e-9, atol=1e-9)
    assert sum(stored.values()) <= sum(t.numel() * t.element_size() for t in xs)
    if plan is not None:
        # A late loss reaches an old episodic row through the episodic path only.
        old = int(plan.group_ids(len(spans) - 1)[0])
        late = torch.zeros_like(actual); late[0, spans[-1][0]:] = 1
        gk = torch.autograd.grad((actual * late).sum(), xs[1])[0]
        assert float(gk[0, :, old].abs().sum()) > 0
        assert old < spans[len(spans) - recent][0]


def test_b0_r8_equals_legacy_window8_bitwise(monkeypatch):
    _cpu_backend(monkeypatch)
    torch.manual_seed(1)
    spans, _, length = _layout(SIZES)
    xs = [torch.randn(1, h, length, 8, dtype=torch.float64, requires_grad=True) for h in (4, 2, 2)]
    legacy = step_flash_attention(None, *xs, None, scaling=.3, step_plan=(5, spans),
                                  window_attention_batch_steps=4)[0]
    new = step_flash_attention(SimpleNamespace(layer_idx=3), *xs, None, scaling=.3,
                               step_plan=(5, spans), window_attention_batch_steps=4,
                               episodic_recent_groups=8)[0]
    assert torch.equal(legacy, new)
    w = torch.randn_like(new)
    for a, b in zip(torch.autograd.grad((legacy * w).sum(), xs), torch.autograd.grad((new * w).sum(), xs)):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


def test_wrong_layer_and_tampered_plans_fail(monkeypatch):
    _cpu_backend(monkeypatch)
    spans, starts, length = _layout(SIZES)
    plan = _plan(7, 4, 6, spans, starts, 3, 0)
    xs = [torch.randn(1, h, length, 8, dtype=torch.float64, requires_grad=True) for h in (4, 2, 2)]
    with pytest.raises(ValueError, match='wrong attention layer'):
        step_flash_attention(SimpleNamespace(layer_idx=3), *xs, None, scaling=.3, step_plan=(5, spans),
                             episodic_recent_groups=4, episodic_plan=plan)
    with pytest.raises(ValueError, match='without the episodic window'):
        step_flash_attention(SimpleNamespace(layer_idx=7), *xs, None, scaling=.3, step_plan=(5, spans),
                             episodic_plan=plan)
    y = step_flash_attention(SimpleNamespace(layer_idx=7), *xs, None, scaling=.3, step_plan=(5, spans),
                             episodic_recent_groups=4, episodic_plan=plan)[0]
    object.__setattr__(plan, 'source_digest', '0' * 64)
    with pytest.raises(RuntimeError, match='changed'):
        y.sum().backward()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Pinned H100 FA2 required')
@pytest.mark.parametrize('batch', [4, 8, 16])
def test_real_fa2_episodic_b0_matches_serial_and_b300_dense(batch):
    torch.manual_seed(153)
    prefix, spans, cursor = 71, [], 71
    for i in range(33):
        size = 97 + i % 7 * 13; spans.append((cursor, cursor + size)); cursor += size
    xs = [torch.randn(1, h, cursor, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
          for h in (16, 4, 4)]
    weight = torch.randn(1, cursor, 16, 128, device='cuda', dtype=torch.bfloat16) / 100

    def run(**kw):
        y = step_flash_attention(SimpleNamespace(layer_idx=7), *xs, None, scaling=128 ** -.5,
                                 step_plan=(prefix, spans), **kw)[0]
        return y, torch.autograd.grad((y * weight).sum(), xs)
    serial, gs = run(window_attention_batch_steps=1)
    new, gn = run(window_attention_batch_steps=batch, episodic_recent_groups=8)
    torch.testing.assert_close(new, serial, rtol=.02, atol=.015)
    for a, e in zip(gn, gs):
        assert float((a - e).float().norm() / e.float().norm()) < .02
