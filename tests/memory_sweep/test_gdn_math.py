"""GDN sweep features versus an independent FP64 sequential oracle (Task 1)."""
import itertools
import numpy as np
import pytest
import torch
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as native
from qwen_vl.models.committed_write import (
    CommittedWriteScorer, _scan_blocks, internal_blocks, prepare_gdn_inputs)
from qwen_vl.models.gdn_features import (
    F0_SCHEMA, F1_LEGACY_SCHEMA, F1_SCHEMA, GdnSweepScorer, cosine_error, drift_priority,
    encode_features, frame_mean_priority, headwise_committed_scan, raw_features,
    require_schema, residual_block_scan)
from qwen_vl.research.gdn_sweep_contract import ARM_REGISTRY, FIXED, GdnSweepSpec

TOL = dict(rtol=1e-10, atol=1e-10)


def oracle(key, value, g, beta, state, value_per_key, segments=()):
    """Token loop straight from the recurrence (key-by-value state per value head)."""
    key = key.repeat_interleave(value_per_key, dim=1)
    out = {name: [] for name in ('error', 'prediction', 'delta', 'self_error')}
    boundaries = []
    starts = {a: b for a, b in segments}
    for i in range(key.shape[0]):
        if i in starts:
            boundaries.append(dict(start=i, end=starts[i], before=state.clone(),
                                   log_decay=torch.zeros(state.shape[0], dtype=state.dtype)))
        state = g[i].exp()[:, None, None] * state
        prediction = torch.einsum('hkv,hk->hv', state, key[i])
        error = value[i] - prediction
        delta = beta[i][:, None] * error
        state = state + torch.einsum('hk,hv->hkv', key[i], delta)
        out['error'].append(error); out['prediction'].append(prediction); out['delta'].append(delta)
        out['self_error'].append(value[i] - torch.einsum('hkv,hk->hv', state, key[i]))
        for b in boundaries:
            if b['start'] <= i < b['end']:
                b['log_decay'] = b['log_decay'] + g[i]
            if i == b['end'] - 1:
                b['after'] = state.clone()
                # Exact transported write: each committed delta decayed by every later alpha.
                write = torch.zeros_like(state)
                for j in range(b['start'], b['end']):
                    later = g[j + 1:b['end']].sum(0).exp()
                    write = write + later[:, None, None] * torch.einsum(
                        'hk,hv->hkv', key[j], out['delta'][j])
                b['write'] = write
    return {k: torch.stack(v) for k, v in out.items()}, state, boundaries


def _inputs(tokens, hk=2, hv=4, dk=5, dv=3, seed=0, dtype=torch.float64, zero_state=False):
    gen = torch.Generator().manual_seed(seed)
    key = torch.randn(tokens, hk, dk, generator=gen, dtype=dtype)
    key = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + 1e-6)    # native additive eps
    value = torch.randn(tokens, hv, dv, generator=gen, dtype=dtype)
    g = -torch.rand(tokens, hv, generator=gen, dtype=dtype) * 0.7
    beta = torch.rand(tokens, hv, generator=gen, dtype=dtype)
    state = torch.zeros(hv, dk, dv, dtype=dtype) if zero_state else torch.randn(
        hv, dk, dv, generator=gen, dtype=dtype)
    return key, value, g, beta, state


SEGMENTS = [[(0, 200)], [(0, 1), (1, 64), (64, 129), (129, 130), (130, 200)],
            [(0, 63), (63, 127), (127, 192), (192, 200)], [(0, 17), (17, 150), (150, 153), (153, 200)]]


@pytest.mark.parametrize('segments', SEGMENTS)
@pytest.mark.parametrize('zero_state', [False, True])
def test_residual_scan_matches_sequential_oracle_fp64(segments, zero_state):
    key, value, g, beta, state = _inputs(200, zero_state=zero_state)
    expected, final, _ = oracle(key, value, g, beta, state.clone(), 2)
    err, pred, dot, actual = residual_block_scan(key, value, g, beta, state.clone(),
                                                 internal_blocks(segments), 2)
    torch.testing.assert_close(err, expected['error'].norm(dim=-1), **TOL)
    torch.testing.assert_close(pred, expected['prediction'].norm(dim=-1), **TOL)
    torch.testing.assert_close(dot, (expected['prediction'] * value).sum(-1), **TOL)
    torch.testing.assert_close(actual, final, **TOL)
    if zero_state:
        assert float(pred[0].abs().max()) == 0.0          # exact zero prior: no fake prediction


@pytest.mark.parametrize('fill', [0.0, 1e-30, 1.0])
def test_beta_extremes_without_division_by_beta(fill):
    key, value, g, _, state = _inputs(70, seed=4)
    beta = torch.full((70, 4), fill, dtype=torch.float64)
    expected, final, _ = oracle(key, value, g, beta, state.clone(), 2)
    err, pred, _, actual = residual_block_scan(key, value, g, beta, state.clone(),
                                               internal_blocks([(0, 70)]), 2)
    assert torch.isfinite(err).all() and torch.isfinite(pred).all()
    torch.testing.assert_close(err, expected['error'].norm(dim=-1), **TOL)
    torch.testing.assert_close(actual, final, **TOL)
    if fill == 0.0:      # pure decay: the residual is v minus the decayed initial read
        decayed = g.cumsum(0).exp()[:, :, None, None] * state
        read = torch.einsum('thkv,thk->thv', decayed, key.repeat_interleave(2, dim=1))
        torch.testing.assert_close(err, (value - read).norm(dim=-1), **TOL)


@pytest.mark.parametrize('decay', [0.0, -80.0])
def test_no_decay_and_strong_decay_are_finite_and_exact(decay):
    key, value, _, beta, state = _inputs(130, seed=7)
    g = torch.full((130, 4), decay, dtype=torch.float64)
    expected, final, _ = oracle(key, value, g, beta, state.clone(), 2)
    err, pred, _, actual = residual_block_scan(key, value, g, beta, state.clone(),
                                               internal_blocks([(0, 60), (60, 130)]), 2)
    torch.testing.assert_close(err, expected['error'].norm(dim=-1), **TOL)
    torch.testing.assert_close(pred, expected['prediction'].norm(dim=-1), **TOL)
    torch.testing.assert_close(actual, final, **TOL)


def test_immediate_self_read_identity_and_s1_equals_s0_plus_s9():
    key, value, g, beta, state = _inputs(90, seed=2)
    expected, _, _ = oracle(key, value, g, beta, state.clone(), 2)
    knorm = (key * key).sum(-1).repeat_interleave(2, dim=1)                    # slightly below one
    torch.testing.assert_close(expected['self_error'],
                               (1 - beta * knorm)[..., None] * expected['error'], **TOL)
    assert bool(((1 - beta * knorm) >= 0).all())
    err, _, _, _ = residual_block_scan(key, value, g, beta, state.clone(), internal_blocks([(0, 90)]), 2)
    s1 = err.mean(-1)
    s0 = (beta * err).mean(-1)
    s9 = ((1 - beta) * err).mean(-1)
    torch.testing.assert_close(s1, s0 + s9, rtol=1e-12, atol=1e-12)


def test_prediction_is_the_decayed_prior_not_the_post_write_read():
    key, value, g, _, state = _inputs(40, seed=9)
    beta = torch.ones(40, 4, dtype=torch.float64)
    expected, _, _ = oracle(key, value, g, beta, state.clone(), 2)
    _, pred, _, _ = residual_block_scan(key, value, g, beta, state.clone(), internal_blocks([(0, 40)]), 2)
    post = (value - expected['self_error']).norm(dim=-1)                        # S_i^T k_i
    torch.testing.assert_close(pred, expected['prediction'].norm(dim=-1), **TOL)
    assert float((pred - post).abs().max()) > 0.1


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('segments', SEGMENTS)
def test_headwise_all_head_scan_is_bitwise_the_legacy_s0_scan(dtype, segments):
    key, value, g, beta, state = _inputs(200, seed=3, dtype=dtype)
    blocks = internal_blocks(segments)
    legacy, legacy_state = _scan_blocks(key, value, g, beta, state.clone(), blocks, 2)
    scores, final, _ = headwise_committed_scan(key, value, g, beta, state.clone(), blocks, 2)
    assert torch.equal(scores, legacy) and torch.equal(final, legacy_state)
    all_heads = torch.arange(4)
    masked, _, _ = headwise_committed_scan(key, value, g, beta, state.clone(), blocks, 2, heads=all_heads)
    assert torch.equal(masked, legacy)
    # Boundary tracking never changes scores or the carried state.
    rows, first = [], 0
    for start, end in segments:
        count = len(internal_blocks([(start, end)]))
        rows.append((first, first + count - 1)); first += count
    tracked, final2, stats = headwise_committed_scan(key, value, g, beta, state.clone(), blocks, 2,
                                                     tracked=rows[1::2])
    assert torch.equal(tracked, legacy) and torch.equal(final2, legacy_state)
    assert stats.shape == (len(rows[1::2]), 6)


def test_headwise_slow_head_mean_matches_oracle():
    key, value, g, beta, state = _inputs(150, seed=5)
    expected, _, _ = oracle(key, value, g, beta, state.clone(), 2)
    heads = torch.tensor([1, 3])
    scores, _, _ = headwise_committed_scan(key, value, g, beta, state.clone(),
                                           internal_blocks([(0, 70), (70, 150)]), 2, heads=heads)
    torch.testing.assert_close(scores, expected['delta'].norm(dim=-1)[:, heads].mean(-1), **TOL)


@pytest.mark.parametrize('zero_beta', [False, True])
def test_boundary_drift_and_exact_transported_write(zero_beta):
    key, value, g, beta, state = _inputs(260, seed=11)
    if zero_beta:
        beta = torch.zeros_like(beta)
    segments = [(0, 30), (30, 131), (131, 135), (135, 260)]                     # prefix|obs|resp|obs
    _, final, boundaries = oracle(key, value, g, beta, state.clone(), 2, [(30, 131), (135, 260)])
    rows, first = [], 0
    for start, end in segments:
        count = len(internal_blocks([(start, end)]))
        rows.append((first, first + count - 1)); first += count
    _, actual, stats = headwise_committed_scan(key, value, g, beta, state.clone(),
                                               internal_blocks(segments), 2, tracked=[rows[1], rows[3]])
    torch.testing.assert_close(actual, final, **TOL)
    for row, b in zip(stats, boundaries):
        alpha = b['log_decay'].exp()[:, None, None]
        torch.testing.assert_close(b['after'], alpha * b['before'] + b['write'], **TOL)  # S_b = A S + W
        decay_part = (alpha - 1) * b['before']
        expected = torch.stack(((b['after'] - b['before']).norm(), b['write'].norm(), b['before'].norm(),
                                decay_part.norm(), (decay_part * b['write']).sum(), b['after'].norm()))
        torch.testing.assert_close(row, expected, **TOL)
        # ||dS||^2 = ||D||^2 + ||W||^2 + 2<D, W>; logged so cancellation is visible.
        torch.testing.assert_close(row[0] ** 2, row[3] ** 2 + row[1] ** 2 + 2 * row[4], **TOL)
        if zero_beta:
            assert float(row[1]) == 0.0                                          # pure decay: W = 0


def test_inputs_are_not_mutated_and_future_tokens_never_change_earlier_scores():
    key, value, g, beta, state = _inputs(150, seed=13)
    copies = [x.clone() for x in (key, value, g, beta, state)]
    blocks = internal_blocks([(0, 70), (70, 150)])
    base = residual_block_scan(key, value, g, beta, state, blocks, 2)
    for before, after in zip(copies, (key, value, g, beta, state)):
        assert torch.equal(before, after)
    changed = value.clone(); changed[100:] += 5.0
    moved = residual_block_scan(key, changed, g, beta, state.clone(), blocks, 2)
    for a, b in zip(base[:3], moved[:3]):
        assert torch.equal(a[:100], b[:100])
        assert float((a[101:] - b[101:]).abs().max()) > 1e-3
    head_base, _, _ = headwise_committed_scan(key, value, g, beta, state.clone(), blocks, 2)
    head_moved, _, _ = headwise_committed_scan(key, changed, g, beta, state.clone(), blocks, 2)
    assert torch.equal(head_base[:100], head_moved[:100])


def test_cosine_error_conventions():
    dot = torch.tensor([0.0, 2.0, -2.0, 1.0, 0.5])
    pred = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0])
    vnorm = torch.tensor([3.0, 2.0, 2.0, 1e-7, 1.0])
    error, zero = cosine_error(dot, pred, vnorm)
    assert float(error[0]) == 1.0                         # zero prediction, nonzero value
    assert abs(float(error[1])) < 1e-6 and abs(float(error[2]) - 2.0) < 1e-6
    assert float(error[3]) == 0.0 and zero.tolist() == [False, False, False, True, False]
    assert abs(float(error[4]) - 0.5) < 1e-6
    with pytest.raises(ValueError, match='Nonfinite'):
        cosine_error(torch.tensor([float('nan')]), pred[:1], vnorm[:1])


def test_drift_floor_and_order_independent_frame_mean():
    stats = torch.tensor([[2.0, 1.0, 4.0, 0, 0, 0], [2.0, 1.0, 1e-9, 0, 0, 0]])
    torch.testing.assert_close(drift_priority(stats, 'net_state_drift', 1e-3),
                               torch.tensor([0.5, 2.0 / 1e-3]))
    torch.testing.assert_close(drift_priority(stats, 'transported_write', 1e-3),
                               torch.tensor([0.25, 1.0 / 1e-3]))
    rng = np.random.default_rng(0)
    scores = rng.random(300).astype(np.float32) * np.float32(1e3)
    assert frame_mean_priority(scores) == frame_mean_priority(scores[rng.permutation(300)])
    assert abs(frame_mean_priority(scores) - float(scores.astype(np.float64).mean())) < 1e-9
    with pytest.raises(TypeError):
        frame_mean_priority(scores.astype(np.float64))
    with pytest.raises(TypeError):
        frame_mean_priority(scores[:299])


def test_raw_v2_records_and_feature_schemas():
    key, value, g, beta, state = _inputs(80, seed=17, zero_state=True)
    value[3] = 0.0                                                             # value-near-zero row
    raw, _ = raw_features(key, value, g, beta, state.clone(), internal_blocks([(0, 80)]), 2)
    f = raw['fields']
    assert raw['schema'] == 'gdn_raw_v2' and f['beta'].shape == (80, 4)
    torch.testing.assert_close(f['committed_norm'], beta * f['error_norm'], rtol=0, atol=0)
    assert bool(f['value_near_zero'][3].all()) and float(f['cosine_error'][3].abs().max()) == 0.0
    assert bool(f['prediction_near_zero'][0].all())                             # zero prior state
    f0 = require_schema(encode_features(raw, F0_SCHEMA), F0_SCHEMA)
    f1 = require_schema(encode_features(raw, F1_SCHEMA), F1_SCHEMA)
    legacy = encode_features(raw, F1_LEGACY_SCHEMA)
    assert f0.shape == (80, 4, 3) and f1.shape == (80, 4, 6)
    assert torch.equal(f1[..., :3], f0)                                          # F0_v1 unchanged
    torch.testing.assert_close(f1[..., 4], torch.log(torch.clamp_min(f['prediction_norm'], 1e-6)),
                               rtol=0, atol=0)
    assert float(f1[0, :, 4].max()) == pytest.approx(np.log(1e-6))              # floored zero
    torch.testing.assert_close(legacy['values'][..., 4],
                               torch.log(torch.clamp_min(f['committed_norm'], 1e-6)), rtol=0, atol=0)
    with pytest.raises(ValueError, match='not'):
        require_schema(legacy, F1_SCHEMA)                                       # never reinterpret
    with pytest.raises(ValueError):
        encode_features(dict(schema='gdn_raw_v1', fields=f), F1_SCHEMA)


# ---- real native module: prepared inputs, serving == offline -------------
def _tiny_layer():
    config = Qwen3_5TextConfig(
        hidden_size=48, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=12, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=6, linear_conv_kernel_dim=4,
        layer_types=['linear_attention'] * 3 + ['full_attention'], vocab_size=32)
    torch.manual_seed(5)
    layer = native.Qwen3_5DecoderLayer(config, 2).float().eval()
    for parameter in layer.parameters():
        torch.nn.init.normal_(parameter, std=0.3)
    with torch.no_grad():
        layer.linear_attn.A_log.uniform_(-1, 1)
    mixer = layer.linear_attn
    mixer.chunk_gated_delta_rule = native.torch_chunk_gated_delta_rule
    mixer.norm = native.Qwen3_5RMSNormGated(mixer.head_v_dim)
    return layer


def tiny_spec(score, heads=(0, 2), floor=1e-3):
    unit, reduction, backend, statistic = ARM_REGISTRY[score]
    return GdnSweepSpec(
        version='episodic_window_gdn_sweep_v1', recent_groups=2, budget_tokens=600, granularity=unit,
        score=score, scoring_pairs=((2, 3),), selector_seed=429, per_source_cap=None,
        score_backend=backend, score_segmentation='prefix_obs_response_block64_v1',
        head_reduction=reduction, statistic=statistic,
        slow_head_mask=((2, tuple(heads)),) if score == 'S6' else None,
        slow_head_mask_sha256='x' if score == 'S6' else None,
        drift_floor=((2, floor),) if score in ('S8N', 'S8W') else None,
        drift_floor_sha256='x' if score in ('S8N', 'S8W') else None, **FIXED)


GDN_ARMS = [s for s in ARM_REGISTRY if not s.startswith('C_')]


@pytest.mark.parametrize('score', GDN_ARMS)
def test_serving_appends_equal_offline_scores_for_every_arm(score):
    layer = _tiny_layer()
    hidden = torch.randn(150, 48, generator=torch.Generator().manual_seed(1))
    segments = [(0, 20), (20, 90), (90, 93), (93, 130), (130, 150)]
    spec = tiny_spec(score)
    tokens, stats = GdnSweepScorer(spec, 2, layer, 'cpu', max_super_block_tokens=50).score_sequence(
        hidden, segments, (1, 3))
    online = GdnSweepScorer(spec, 2, layer, 'cpu')
    parts = [online.append(hidden[0:20], 'prefix'), online.append(hidden[20:90], 'observation')]
    for i in range(90, 92):
        assert online.append(hidden[i:i + 1], 'response_token') is None
    parts.append(online.append(hidden[92:93], 'response_close'))
    parts.append(online.append(hidden[93:130], 'observation'))
    parts.append(online.append(hidden[130:150], 'response_close'))
    if score in ('S8N', 'S8W'):
        assert tokens is None and stats.shape == (2, 6)
        online_stats = torch.cat([p[1] for p in parts if p[1] is not None])
        torch.testing.assert_close(online_stats, stats, rtol=1e-5, atol=1e-6)
    else:
        torch.testing.assert_close(torch.cat([p[0] for p in parts]), tokens, rtol=1e-5, atol=1e-6)


def test_s0_sweep_scorer_is_the_unchanged_legacy_scorer_and_tracks_native_state():
    layer = _tiny_layer()
    hidden = torch.randn(140, 48, generator=torch.Generator().manual_seed(2))
    segments = [(0, 20), (20, 90), (90, 93), (93, 140)]
    legacy = CommittedWriteScorer(layer, 'cpu').score_sequence(hidden, segments)
    tokens, _ = GdnSweepScorer(tiny_spec('S0'), 2, layer, 'cpu').score_sequence(hidden, segments)
    assert torch.equal(tokens, legacy)
    # Residual backend's carried state equals the independent native recurrence.
    tail = torch.zeros(3, layer.linear_attn.conv_dim)
    key, value, g, beta, _ = prepare_gdn_inputs(layer, hidden, tail)
    _, expected_state, _ = oracle(key.double(), value.double(), g.double(), beta.double(),
                                  torch.zeros(4, 8, 6, dtype=torch.float64), 2)
    scorer = GdnSweepScorer(tiny_spec('S1'), 2, layer, 'cpu')
    scorer.score_sequence(hidden, segments)
    torch.testing.assert_close(scorer.state.double(), expected_state, rtol=2e-4, atol=2e-5)


def test_token_statistics_follow_the_registry_formulas():
    layer = _tiny_layer()
    hidden = torch.randn(70, 48, generator=torch.Generator().manual_seed(3))
    segments = [(0, 70)]
    key, value, g, beta, _ = prepare_gdn_inputs(layer, hidden, torch.zeros(3, layer.linear_attn.conv_dim))
    err, pred, dot, _ = residual_block_scan(key, value, g, beta, torch.zeros(4, 8, 6),
                                            internal_blocks(segments), 2)
    vnorm = value.norm(dim=-1)
    cos, _ = cosine_error(dot, pred, vnorm)
    expected = dict(S1=err.mean(-1), S2=(beta * vnorm).mean(-1), S3=vnorm.mean(-1),
                    S4=cos.min(-1).values, S5=cos.mean(-1), S9=((1 - beta) * err).mean(-1))
    for score, value_ in expected.items():
        tokens, _ = GdnSweepScorer(tiny_spec(score), 2, layer, 'cpu').score_sequence(hidden, segments)
        torch.testing.assert_close(tokens, value_, rtol=0, atol=0, msg=score)
