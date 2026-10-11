"""Committed-write delta: FP64 block form versus an independent sequential oracle."""
import pytest
import torch
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as native
from qwen_vl.models.committed_write import (
    CommittedWriteScorer, _scan_blocks, internal_blocks, prepare_gdn_inputs)


def sequential_oracle(key, value, g, beta, state, value_per_key):
    """Token loop straight from the recurrence; key-by-value state per value head."""
    key = key.repeat_interleave(value_per_key, dim=1)
    deltas = []
    for i in range(key.shape[0]):
        state = g[i].exp()[:, None, None] * state
        residual = value[i] - torch.einsum('hkv,hk->hv', state, key[i])
        delta = beta[i][:, None] * residual
        state = state + torch.einsum('hk,hv->hkv', key[i], delta)
        deltas.append(delta)
    deltas = torch.stack(deltas)
    return deltas.norm(dim=-1).mean(-1), state, deltas


def _inputs(tokens, hk=2, hv=4, dk=5, dv=3, seed=0, dtype=torch.float64):
    gen = torch.Generator().manual_seed(seed)
    key = torch.randn(tokens, hk, dk, generator=gen, dtype=dtype)
    key = key / key.norm(dim=-1, keepdim=True)
    value = torch.randn(tokens, hv, dv, generator=gen, dtype=dtype)
    g = -torch.rand(tokens, hv, generator=gen, dtype=dtype) * 0.7          # variable decay
    beta = torch.rand(tokens, hv, generator=gen, dtype=dtype)
    state = torch.randn(hv, dk, dv, generator=gen, dtype=dtype)              # nonzero initial state
    return key, value, g, beta, state


@pytest.mark.parametrize('segments', [[(0, 200)], [(0, 17), (17, 150), (150, 153), (153, 200)],
                                      [(0, 1)] * 1 + [(1, 64), (64, 129), (129, 130), (130, 200)]])
def test_block_scan_matches_sequential_oracle_fp64_unequal_widths(segments):
    key, value, g, beta, state = _inputs(200)
    expected, final, _ = sequential_oracle(key, value, g, beta, state.clone(), 2)
    scores, actual_final = _scan_blocks(key, value, g, beta, state.clone(), internal_blocks(segments), 2)
    torch.testing.assert_close(scores, expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(actual_final, final, rtol=1e-10, atol=1e-10)


def test_beta_extremes_near_zero_keys_and_no_double_beta():
    key, value, g, beta, state = _inputs(70, seed=4)
    scores, _ = _scan_blocks(key, value, g, torch.zeros_like(beta), state.clone(), internal_blocks([(0, 70)]), 2)
    assert float(scores.abs().max()) == 0.0
    ones = torch.ones_like(beta)
    expected, _, deltas = sequential_oracle(key, value, g, ones, state.clone(), 2)
    scores, _ = _scan_blocks(key, value, g, ones, state.clone(), internal_blocks([(0, 70)]), 2)
    torch.testing.assert_close(scores, expected, rtol=1e-10, atol=1e-10)
    # Half beta must halve the first committed delta (no beta^2 anywhere).
    half = torch.full_like(beta, 0.5)
    first, _ = _scan_blocks(key[:1], value[:1], g[:1], half, state.clone(), [(0, 1)], 2)
    once, _ = _scan_blocks(key[:1], value[:1], g[:1], ones[:1], state.clone(), [(0, 1)], 2)
    torch.testing.assert_close(first, once / 2, rtol=1e-12, atol=1e-12)
    tiny = key.clone(); tiny[5] = 1e-30
    expected, _, _ = sequential_oracle(tiny, value, g, beta, state.clone(), 2)
    scores, _ = _scan_blocks(tiny, value, g, beta, state.clone(), internal_blocks([(0, 70)]), 2)
    torch.testing.assert_close(scores, expected, rtol=1e-10, atol=1e-10)


def test_prefix_and_nonvisual_writes_change_later_scores_strong_decay_is_finite():
    key, value, g, beta, state = _inputs(130, seed=7)
    base, _ = _scan_blocks(key, value, g, beta, state.clone(), internal_blocks([(0, 60), (60, 130)]), 2)
    edited = value.clone(); edited[10] += 3.0                       # an earlier (e.g. prefix/text) write
    moved, _ = _scan_blocks(key, edited, g, beta, state.clone(), internal_blocks([(0, 60), (60, 130)]), 2)
    torch.testing.assert_close(moved[:10], base[:10], rtol=0, atol=0)
    assert float((moved[11:] - base[11:]).abs().max()) > 1e-3
    strong = torch.full_like(g, -80.0)                              # exp underflow inside blocks
    scores, final = _scan_blocks(key, value, strong, beta, state.clone(), internal_blocks([(0, 130)]), 2)
    expected, efinal, _ = sequential_oracle(key, value, strong, beta, state.clone(), 2)
    assert torch.isfinite(scores).all() and torch.isfinite(final).all()
    torch.testing.assert_close(scores, expected, rtol=1e-10, atol=1e-10)


def _tiny_layer(dtype=torch.float32):
    config = Qwen3_5TextConfig(
        hidden_size=48, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=12, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=6, linear_conv_kernel_dim=4,
        layer_types=['linear_attention'] * 3 + ['full_attention'], vocab_size=32)
    torch.manual_seed(5)
    layer = native.Qwen3_5DecoderLayer(config, 2).to(dtype).eval()
    for parameter in layer.parameters():
        torch.nn.init.normal_(parameter, std=0.3)
    with torch.no_grad():
        layer.linear_attn.A_log.uniform_(-1, 1)
    mixer = layer.linear_attn
    mixer.chunk_gated_delta_rule = native.torch_chunk_gated_delta_rule
    mixer.norm = native.Qwen3_5RMSNormGated(mixer.head_v_dim).to(dtype)   # CPU fixture
    return layer


def test_prepared_inputs_match_native_mixer_and_scores_track_native_state():
    layer = _tiny_layer()
    hidden = torch.randn(1, 150, 48)
    captured = {}

    def capture(query, key, value, g, beta, **kwargs):
        captured.update(key=key, value=value, g=g, beta=beta)
        captured['result'] = native.torch_chunk_gated_delta_rule(
            query.double(), key.double(), value.double(), g=g.double(), beta=beta.double(),
            output_final_state=True, use_qk_l2norm_in_kernel=True)
        return captured['result'][0].to(query.dtype), None
    layer.linear_attn.chunk_gated_delta_rule = capture
    with torch.no_grad():
        layer(hidden, position_embeddings=None)
    tail = torch.zeros(3, layer.linear_attn.conv_dim)
    key, value, g, beta, _ = prepare_gdn_inputs(layer, hidden[0], tail)
    native_key = captured['key'][0, :, ::2]                          # undo repeat_interleave
    native_key = native_key / torch.sqrt((native_key * native_key).sum(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(key, native_key, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(value, captured['value'][0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(g, captured['g'][0], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(beta, captured['beta'][0], rtol=1e-6, atol=1e-6)
    # The native torch chunk rule's final state equals the helper's carried state.
    _, native_state = captured['result']
    scorer = CommittedWriteScorer(layer, 'cpu')
    scorer.conv_tail = tail.clone()
    scores = scorer.score_sequence(hidden[0], [(0, 40), (40, 147), (147, 150)])
    expected, final, _ = sequential_oracle(key.double(), value.double(), g.double(), beta.double(),
                                           torch.zeros(4, 8, 6, dtype=torch.float64), 2)
    torch.testing.assert_close(scores.double(), expected, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(scorer.state.double(), final, rtol=2e-4, atol=2e-5)
    # HF state layout is [B, H, Dk, Dv]; independent native recurrence agrees.
    torch.testing.assert_close(scorer.state.double(), native_state[0].double(), rtol=2e-4, atol=2e-5)


def test_serving_appends_equal_offline_scores_with_identical_segmentation():
    layer = _tiny_layer()
    hidden = torch.randn(140, 48)
    segments = [(0, 20), (20, 90), (90, 93), (93, 120), (120, 140)]
    offline = CommittedWriteScorer(layer, 'cpu', max_super_block_tokens=50).score_sequence(hidden, segments)
    online = CommittedWriteScorer(layer, 'cpu')
    parts = [online.append(hidden[0:20], 'prefix'), online.append(hidden[20:90], 'observation')]
    for i in range(90, 92):
        assert online.append(hidden[i:i + 1], 'response_token') is None
    parts.append(online.append(hidden[92:93], 'response_close'))
    parts.append(online.append(hidden[93:120], 'observation'))
    parts.append(online.append(hidden[120:140], 'response_close'))
    torch.testing.assert_close(torch.cat(parts), offline, rtol=1e-5, atol=1e-6)
