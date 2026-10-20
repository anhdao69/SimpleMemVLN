"""Tiny real Qwen3.5 text model (FP32 CPU): sweep offline training == incremental serving."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from qwen_vl.models.episodic_forward import episodic_text_forward
from qwen_vl.research.episodic_contract import EpisodicSpec, scoring_pairs_from_layer_types
from qwen_vl.research.gdn_sweep_contract import ARM_REGISTRY, FIXED, GdnSweepSpec
from qwen_vl.stream.cache import make_stream_cache_fp32
from qwen_vl.stream.gdn_sweep_state import GdnSweepSessionState
from test_episodic_forward import _episode, _offline, tiny  # noqa: F401  (fixture)
from test_episodic_forward import _online as _legacy_online


def sweep_spec(model, score, recent=3, budget=600):
    unit, reduction, backend, statistic = ARM_REGISTRY[score]
    pairs = scoring_pairs_from_layer_types(model.config.layer_types)
    return GdnSweepSpec(
        version='episodic_window_gdn_sweep_v1', recent_groups=recent, budget_tokens=budget,
        granularity=unit, score=score, scoring_pairs=pairs, selector_seed=429, per_source_cap=None,
        score_backend=backend,
        score_segmentation='none' if score.startswith('C_') else 'prefix_obs_response_block64_v1',
        head_reduction=reduction, statistic=statistic,
        slow_head_mask=tuple((gdn, (1, 2)) for gdn, _ in pairs) if score == 'S6' else None,
        slow_head_mask_sha256='test' if score == 'S6' else None,
        drift_floor=tuple((gdn, 1e-3) for gdn, _ in pairs) if score in ('S8N', 'S8W') else None,
        drift_floor_sha256='test' if score in ('S8N', 'S8W') else None, **FIXED)


def _online(model, spec, layout, embeds, positions):
    wrapper = SimpleNamespace(backbone=SimpleNamespace(model=SimpleNamespace(language_model=model)))
    state = GdnSweepSessionState(spec, wrapper, 'ep', 'cpu')
    cache = make_stream_cache_fp32(model.config)
    outputs = []

    def append(start, end, kind, offset=None):
        state.begin_append(kind, start, end - start, offset)
        out = episodic_text_forward(model, embeds[:, start:end], positions[..., start:end], spec,
                                    past_key_values=cache, use_cache=True, stream_append=True,
                                    serving=state)
        state.end_append(cache)
        outputs.append(out)
    append(0, layout.prefix, 'prefix')
    for t, ((start, end), observed) in enumerate(zip(layout.group_spans, layout.observation_ends)):
        state.promote(cache, t)
        state.open_group(t, start, layout.visual_starts[t])
        append(start, observed, 'observation', layout.visual_starts[t] - start)
        for token in range(observed, end - 2):
            append(token, token + 1, 'response_token')
        append(end - 2, end, 'response_close')
        state.close_group(end)
        for layer in spec.attention_layers:
            assert cache.layers[layer].keys.shape[-2] <= layout.prefix + spec.budget_tokens + spec.recent_groups * 320
    return torch.cat(outputs, 1), state


@pytest.mark.parametrize('score', list(ARM_REGISTRY))
def test_offline_training_equals_incremental_serving(tiny, score):
    spec = sweep_spec(tiny, score)
    layout, embeds, positions = _episode(9)
    diagnostics = {}
    with torch.no_grad():
        offline = _offline(tiny, spec, layout, embeds, positions, diagnostics=diagnostics)
        online, state = _online(tiny, spec, layout, embeds, positions)
    torch.testing.assert_close(online, offline, rtol=2e-4, atol=2e-5)
    last = layout.groups - 1
    for layer in spec.attention_layers:
        final = np.sort(state.running[layer].ids)
        np.testing.assert_array_equal(final, diagnostics['plans'][layer].group_ids(last))
        assert final.size == min(spec.budget_tokens, max(0, last - spec.recent_groups + 1) * 300)
        if spec.is_frame:
            assert np.unique(state.running[layer].sources).size == final.size // 300
    # Cosine error is bounded in [0, 2]; its natural FP32 path-difference metric is absolute.
    atol = 1e-4 if spec.statistic == 'cosine_error' else 1e-6
    for group in state.groups:            # resident groups still hold their fixed priorities
        for layer in spec.attention_layers:
            if 'visual_scores' in diagnostics and not spec.is_frame:
                np.testing.assert_allclose(group.scores[layer],
                                           diagnostics['visual_scores'][layer][group.step_id].numpy(),
                                           rtol=1e-4, atol=atol)
            if 'frame_scores' in diagnostics:
                np.testing.assert_allclose(group.scores[layer],
                                           diagnostics['frame_scores'][layer][group.step_id], rtol=1e-4)


def test_s0_sweep_forward_is_bitwise_legacy_m5(tiny):
    pairs = scoring_pairs_from_layer_types(tiny.config.layer_types)
    legacy = EpisodicSpec('episodic_window_v1', 3, 600, 'token', 'committed_mean_l2_v1', pairs, 429,
                          None, 'b', 's', 'a')
    layout, embeds, positions = _episode(9, seed=5)
    d_legacy, d_sweep = {}, {}
    with torch.no_grad():
        a = _offline(tiny, legacy, layout, embeds, positions, diagnostics=d_legacy)
        b = _offline(tiny, sweep_spec(tiny, 'S0'), layout, embeds, positions, diagnostics=d_sweep)
        online_legacy, _ = _legacy_online(tiny, legacy, layout, embeds, positions)
        online_sweep, _ = _online(tiny, sweep_spec(tiny, 'S0'), layout, embeds, positions)
    assert torch.equal(a, b) and torch.equal(online_legacy, online_sweep)
    for layer in legacy.attention_layers:
        assert d_legacy['plans'][layer].source_digest == d_sweep['plans'][layer].source_digest
        assert torch.equal(d_legacy['visual_scores'][layer], d_sweep['visual_scores'][layer])


@pytest.mark.parametrize('score', ['S1', 'S8W', 'S7', 'C_RECENCY'])
def test_passive_diagnostics_do_not_change_outputs(tiny, score):
    spec = sweep_spec(tiny, score)
    layout, embeds, positions = _episode(8, seed=2)
    with torch.no_grad():
        plain = _offline(tiny, spec, layout, embeds, positions)
        traced = _offline(tiny, spec, layout, embeds, positions, diagnostics={})
    assert torch.equal(plain, traced)


@pytest.mark.parametrize('score', ['S1', 'S4', 'S8N', 'S6'])
def test_checkpointing_gradients_late_loss_and_future_independence(tiny, score):
    spec = sweep_spec(tiny, score, recent=2, budget=300)
    layout, embeds, positions = _episode(7, seed=3)
    leaf = embeds.clone().requires_grad_()

    def loss_and_grad(checkpoint):
        tiny.train()
        if checkpoint:
            tiny.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        else:
            tiny.gradient_checkpointing_disable()
        out = _offline(tiny, spec, layout, leaf, positions)
        loss = out[0, layout.group_spans[-1][0]:].pow(2).sum()
        grad, = torch.autograd.grad(loss, leaf)
        tiny.eval()
        return out.detach(), grad
    plain_out, plain = loss_and_grad(False)
    ckpt_out, ckpt = loss_and_grad(True)
    tiny.gradient_checkpointing_disable()
    torch.testing.assert_close(ckpt_out, plain_out, rtol=0, atol=0)
    torch.testing.assert_close(ckpt, plain, rtol=1e-6, atol=1e-7)
    changed = embeds.clone(); changed[:, layout.group_spans[4][0]:] += 3.0
    with torch.no_grad():
        moved = _offline(tiny, spec, layout, changed, positions)
    cut = layout.group_spans[4][0]
    torch.testing.assert_close(moved[:, :cut], plain_out[:, :cut], rtol=0, atol=0)
