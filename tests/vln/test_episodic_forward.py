"""Tiny real Qwen3.5 text model (FP32 CPU, torch fallbacks): offline == serving."""
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as native
from qwen_vl.data.episodic_metadata import EpisodicLayout
from qwen_vl.models.episodic_forward import episodic_text_forward
from qwen_vl.research.episodic_contract import EpisodicSpec, scoring_pairs_from_layer_types
from qwen_vl.stream.cache import make_stream_cache_fp32
from qwen_vl.stream.episodic_cache import EpisodicSessionState
from qwen_vl.stream.window_attention import register_window_attention
from test_window_memory import _cpu_flash
from test_window_batched import _cpu_varlen

PREFIX, LEAD, RESPONSE = 9, 2, (3, 2, 4, 1, 3)


@pytest.fixture
def tiny(monkeypatch):
    for name in ['FusedRMSNormGated', 'chunk_gated_delta_rule', 'fused_recurrent_gated_delta_rule',
                 'causal_conv1d_fn', 'causal_conv1d_update']:
        monkeypatch.setattr(native, name, None)
    monkeypatch.setitem(sys.modules, 'flash_attn', SimpleNamespace(
        flash_attn_varlen_func=_cpu_varlen, flash_attn_func=_cpu_flash))
    register_window_attention()
    torch.manual_seed(7)
    types = ['linear_attention', 'full_attention'] * 2
    cfg = Qwen3_5TextConfig(
        hidden_size=48, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=12, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=6, vocab_size=64, layer_types=types,
        rope_parameters={'rope_type': 'default', 'rope_theta': 1e6, 'partial_rotary_factor': 1.0,
                         'mrope_section': [2, 2, 2]},
        attn_implementation='simplememvln_step_attention')
    model = native.Qwen3_5TextModel(cfg).float().eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.2)
    return model


def _spec(model, recent, budget, score):
    return EpisodicSpec('episodic_window_v1', recent, budget, 'token', score,
                        scoring_pairs_from_layer_types(model.config.layer_types), 3, None,
                        'b', 's', 'a')


def _episode(groups, seed=0):
    spans, obs, starts, kinds, cursor = [], [], [], [], PREFIX
    for t in range(groups):
        starts.append(cursor + LEAD)
        observed = cursor + LEAD + 300 + 3
        end = observed + RESPONSE[t % len(RESPONSE)] + 2       # response tokens + EOS + separator
        spans.append((cursor, end)); obs.append(observed); cursor = end
    layout = EpisodicLayout(PREFIX, tuple(spans), tuple(obs), tuple(starts))
    gen = torch.Generator().manual_seed(seed)
    embeds = torch.randn(1, cursor, 48, generator=gen)
    positions = torch.arange(cursor).view(1, 1, -1).expand(4, 1, -1)
    return layout, embeds, positions


def _offline(model, spec, layout, embeds, positions, batch=3, diagnostics=None):
    return episodic_text_forward(model, embeds, positions, spec, step_plan=(layout.prefix, layout.group_spans),
                                 layout=layout, episode_uid='ep', window_attention_batch_steps=batch,
                                 diagnostics=diagnostics)


def _online(model, spec, layout, embeds, positions):
    wrapper = SimpleNamespace(backbone=SimpleNamespace(model=SimpleNamespace(language_model=model)))
    state = EpisodicSessionState(spec, wrapper, 'ep', 'cpu')
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
        kv = cache.layers[1].keys.shape[-2]
        assert kv <= layout.prefix + spec.budget_tokens + spec.recent_groups * 320
    return torch.cat(outputs, 1), state


@pytest.mark.parametrize('recent,budget,score', [(8, 0, 'none'), (3, 0, 'none'), (3, 300, 'committed_mean_l2_v1'),
                                                 (2, 450, 'committed_mean_l2_v1'), (3, 300, 'random_hash_v1')])
def test_offline_training_equals_incremental_serving(tiny, recent, budget, score):
    spec = _spec(tiny, recent, budget, score)
    layout, embeds, positions = _episode(9)
    diagnostics = {}
    with torch.no_grad():
        offline = _offline(tiny, spec, layout, embeds, positions, diagnostics=diagnostics)
        online, state = _online(tiny, spec, layout, embeds, positions)
    torch.testing.assert_close(online, offline, rtol=2e-4, atol=2e-5)
    plans = diagnostics['plans']
    for layer in spec.attention_layers:
        final = np.sort(state.running[layer].ids)
        if budget:
            # Serving E before the last group equals the offline plan of that group.
            last = layout.groups - 1
            expected = plans[layer].group_ids(last)
            # promote() at step `last` produced exactly that set; later groups never promoted.
            np.testing.assert_array_equal(final, expected)
            assert final.size == min(budget, max(0, last - recent + 1) * 300)
        else:
            assert final.size == 0 and layer not in plans
    if score == 'committed_mean_l2_v1':
        for group in state.groups:              # resident groups still hold their fixed scores
            for layer in spec.attention_layers:
                np.testing.assert_allclose(group.scores[layer],
                                           diagnostics['visual_scores'][layer][group.step_id].numpy(),
                                           rtol=1e-4, atol=1e-6)


def test_project_forward_b0_matches_native_window8_model_bitwise(tiny):
    layout, embeds, positions = _episode(10)
    step = (layout.prefix, layout.group_spans)
    with torch.no_grad():
        native_out = tiny(inputs_embeds=embeds, position_ids=positions, use_cache=False,
                          step_plan=step, window_attention_batch_steps=4).last_hidden_state
        project = _offline(tiny, _spec(tiny, 8, 0, 'none'), layout, embeds, positions, batch=4)
    assert torch.equal(native_out, project)


def test_checkpointing_gradients_late_loss_and_future_independence(tiny):
    spec = _spec(tiny, 2, 300, 'committed_mean_l2_v1')
    layout, embeds, positions = _episode(7, seed=3)
    leaf = embeds.clone().requires_grad_()

    def loss_and_grad(checkpoint):
        tiny.train()
        if checkpoint:
            tiny.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        else:
            tiny.gradient_checkpointing_disable()
        out = _offline(tiny, spec, layout, leaf, positions)
        loss = out[0, layout.group_spans[-1][0]:].pow(2).sum()     # loss only in the final group
        grad, = torch.autograd.grad(loss, leaf)
        tiny.eval()
        return out.detach(), grad
    plain_out, plain = loss_and_grad(False)
    ckpt_out, ckpt = loss_and_grad(True)
    torch.testing.assert_close(ckpt_out, plain_out, rtol=0, atol=0)
    torch.testing.assert_close(ckpt, plain, rtol=1e-6, atol=1e-7)
    # Old episodic rows (group 0 visual tokens) get gradient although outside the window.
    assert float(plain[0, layout.visual_starts[0]:layout.visual_starts[0] + 300].abs().sum()) > 0
    # Changing only future inputs never changes earlier outputs.
    changed = embeds.clone(); changed[:, layout.group_spans[4][0]:] += 3.0
    with torch.no_grad():
        moved = _offline(tiny, spec, layout, changed, positions)
    torch.testing.assert_close(moved[:, :layout.group_spans[4][0]], plain_out[:, :layout.group_spans[4][0]],
                               rtol=0, atol=0)


def test_interleaved_episodes_keep_their_own_plans(tiny):
    spec = _spec(tiny, 2, 300, 'committed_mean_l2_v1')
    a = _episode(6, seed=1); b = _episode(6, seed=2)
    tiny.train()
    tiny.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    da, db = {}, {}
    out_a = _offline(tiny, spec, *a, diagnostics=da)
    out_b = _offline(tiny, spec, *b, diagnostics=db)      # forward B before A's backward
    ga = torch.autograd.grad(out_a.sum(), [p for p in tiny.parameters()], allow_unused=True)
    gb = torch.autograd.grad(out_b.sum(), [p for p in tiny.parameters()], allow_unused=True)
    tiny.gradient_checkpointing_disable(); tiny.eval()
    for plan_a, plan_b in zip(da['plans'].values(), db['plans'].values()):
        plan_a.verify(); plan_b.verify()
    # Reference: A alone, no interleaving.
    tiny.train()
    ref = torch.autograd.grad(_offline(tiny, spec, *a).sum(), [p for p in tiny.parameters()], allow_unused=True)
    tiny.eval()
    for x, y in zip(ga, ref):
        if x is not None:
            torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6)
