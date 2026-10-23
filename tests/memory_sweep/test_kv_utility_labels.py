"""Exact local LOO labels versus an independent brute-force masked attention (FP64)."""
import math
import numpy as np
import pytest
import torch
from qwen_vl.research.kv_utility_labels import FixedQueryContext, visible_rows

F64 = torch.float64
HQ, HKV, D, DM = 8, 2, 16, 24


def _case(n=40, seed=0, sharp=None):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(HQ, D, generator=g, dtype=F64)
    gate = torch.sigmoid(torch.randn(HQ, D, generator=g, dtype=F64))
    keys = torch.randn(n, HKV, D, generator=g, dtype=F64)
    values = torch.randn(n, HKV, D, generator=g, dtype=F64)
    w_o = torch.randn(DM, HQ * D, generator=g, dtype=F64) / math.sqrt(HQ * D)
    if sharp is not None:            # make row `sharp` dominate query head 0 (KV head 0)
        keys[sharp, 0] = 200 * q[0] / q[0].norm()
    return q, gate, keys, values, w_o


def _brute_output(q, gate, keys, values, w_o, rows):
    """Independent per-head loop: softmax over the explicit row list, native repeat_kv mapping."""
    heads = []
    rep = HQ // HKV
    for h in range(HQ):
        kv = h // rep
        k = keys[rows, kv]
        v = values[rows, kv]
        weights = torch.softmax(k @ q[h] / math.sqrt(D), 0)
        heads.append(gate[h] * (weights @ v))
    return w_o @ torch.cat(heads)


def _brute_utility(q, gate, keys, values, w_o, context, candidate):
    with_i = _brute_output(q, gate, keys, values, w_o, context)
    without = _brute_output(q, gate, keys, values, w_o, [r for r in context if r != candidate])
    return float((with_i - without).norm() / math.sqrt(DM))


def test_resident_matches_brute_force():
    q, gate, keys, values, w_o = _case()
    ctx = FixedQueryContext(q, gate, keys, values, 1 / math.sqrt(D), w_o)
    rows = [0, 3, 17, 39]
    got = ctx.resident(rows).numpy()
    want = [_brute_utility(q, gate, keys, values, w_o, list(range(40)), r) for r in rows]
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(ctx.output().numpy(),
                               _brute_output(q, gate, keys, values, w_o, list(range(40))).numpy(), rtol=1e-12)


def test_shadow_matches_brute_force_with_displacement():
    q, gate, keys, values, w_o = _case(n=50, seed=1)
    reference = list(range(40))                       # rows 40.. are archived eligible candidates
    ctx = FixedQueryContext(q, gate, keys[:40], values[:40], 1 / math.sqrt(D), w_o)
    displaced = 11
    candidates = [40, 45, 49]
    got = ctx.shadow(keys[candidates], values[candidates], displaced).numpy()
    for c, value in zip(candidates, got):
        context = [r for r in reference if r != displaced] + [c]
        assert len(context) == len(reference)          # equal occupancy
        assert value == pytest.approx(_brute_utility(q, gate, keys, values, w_o, context, c), rel=1e-10)


def test_probability_near_one_uses_masked_recompute():
    q, gate, keys, values, w_o = _case(n=30, seed=2, sharp=5)
    ctx = FixedQueryContext(q, gate, keys, values, 1 / math.sqrt(D), w_o)
    assert float(ctx.prob[:, 5].max()) > 1 - 1e-12              # a ~ 1: closed form would divide by ~0
    got = ctx.resident([5]).numpy()[0]
    assert ctx.fallbacks == 1
    assert got == pytest.approx(_brute_utility(q, gate, keys, values, w_o, list(range(30)), 5), rel=1e-9)
    ctx2 = FixedQueryContext(q, gate, keys, values, 1 / math.sqrt(D), w_o)
    extra = _case(n=31, seed=3)[2][30:]
    v_extra = _case(n=31, seed=3)[3][30:]
    allk, allv = torch.cat((keys, extra)), torch.cat((values, v_extra))
    got = ctx2.shadow(extra, v_extra, 5).numpy()[0]                # displaced row dominates: fallback
    assert ctx2.fallbacks == 1
    context = [r for r in range(30) if r != 5] + [30]
    assert got == pytest.approx(_brute_utility(q, gate, allk, allv, w_o, context, 30), rel=1e-9)


def test_gqa_row_removed_from_whole_group_and_cross_head_cancellation():
    q, gate, keys, values, w_o = _case(n=12, seed=4)
    # Two heads of one KV group with opposite projections cancel exactly after W_O.
    q[1] = q[0]
    w_o[:, D:2 * D] = -w_o[:, :D]
    ctx = FixedQueryContext(q, gate.clone().fill_(0.5), keys, values, 1 / math.sqrt(D), w_o)
    zeroed = w_o.clone()
    zeroed[:, :2 * D] = 0
    ctx_zero = FixedQueryContext(q, gate.clone().fill_(0.5), keys, values, 1 / math.sqrt(D), zeroed)
    np.testing.assert_allclose(ctx.resident([3]).numpy(), ctx_zero.resident([3]).numpy(), rtol=1e-10)
    # Removing only one of the two cancelling heads changes the label: contributions are
    # summed before the norm, not replaced by per-head magnitudes.
    half = w_o.clone()
    half[:, D:2 * D] = 0
    ctx_half = FixedQueryContext(q, gate.clone().fill_(0.5), keys, values, 1 / math.sqrt(D), half)
    assert abs(float(ctx_half.resident([3])[0] - ctx.resident([3])[0])) > 1e-6
    # The same logical row leaves every head of its KV group (heads 0..3 read KV head 0).
    brute = _brute_utility(q, gate.clone().fill_(0.5), keys, values, w_o, list(range(12)), 3)
    assert float(ctx.resident([3])[0]) == pytest.approx(brute, rel=1e-10)


def test_bias_cancels_and_inputs_are_not_mutated():
    q, gate, keys, values, w_o = _case(seed=5)
    copies = [t.clone() for t in (q, gate, keys, values, w_o)]
    ctx = FixedQueryContext(q, gate, keys, values, 1 / math.sqrt(D), w_o)
    ctx.resident([1, 2])
    ctx.shadow(keys[:3].clone(), values[:3].clone(), 7)
    for a, b in zip(copies, (q, gate, keys, values, w_o)):
        assert torch.equal(a, b)


def test_requires_fp64_and_finite():
    q, gate, keys, values, w_o = _case()
    with pytest.raises(TypeError):
        FixedQueryContext(q.float(), gate, keys, values, 0.25, w_o)
    keys[0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        FixedQueryContext(q, gate, keys, values, 0.25, w_o)


def test_visible_rows_contract():
    ids = visible_rows(5, [7, 9, 20], 30, 33)
    assert ids.tolist() == [0, 1, 2, 3, 4, 7, 9, 20, 30, 31, 32, 33]
    with pytest.raises(ValueError):
        visible_rows(5, [9, 7], 30, 33)
    with pytest.raises(ValueError):
        visible_rows(5, [31], 30, 33)
