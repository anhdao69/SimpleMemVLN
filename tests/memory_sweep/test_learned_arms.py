"""Phase-B learned arms: embedded artifacts fail closed; offline training == incremental serving."""
from copy import deepcopy
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as native
from qwen_vl.research.episodic_contract import parse_episodic_spec, scoring_pairs_from_layer_types
from qwen_vl.research.gdn_ranker import (
    ARMS, RankerHead, build_artifact, canonical_sha256, flatten_window, input_width, parameter_count)
from qwen_vl.research.gdn_sweep_contract import (
    FIXED, LEARNED_REGISTRY, GdnSweepSpec, arm_memory_config)
from qwen_vl.stream.window_attention import register_window_attention
from test_episodic_forward import _episode
from test_gdn_sweep_contract import _with
from test_gdn_sweep_forward import _online
from test_window_memory import _cpu_flash
from test_window_batched import _cpu_varlen
from test_episodic_forward import _offline

LEARNED = list(LEARNED_REGISTRY)


def random_artifact(arm, attention_layers, value_heads, seed=0, scale=1.0):
    torch.manual_seed(seed)
    width = input_width(arm, value_heads)
    heads, norms = {}, {}
    rng = np.random.default_rng(seed)
    for a in attention_layers:
        heads[a] = RankerHead(width, ARMS[arm]['hidden'])
        for p in heads[a].parameters():
            torch.nn.init.normal_(p, std=scale / np.sqrt(width))
        norms[a] = (rng.normal(size=width).astype(np.float32), (0.5 + rng.random(width)).astype(np.float32))
    return build_artifact(arm, heads, norms, {a: 1.0 for a in attention_layers}, dict(test=True),
                          value_heads=value_heads)


def test_parameter_counts_match_the_plan():
    assert {arm: parameter_count(arm) for arm in ARMS} == dict(L0=776, L1=50184, L2=148488, L3=295944)


@pytest.mark.parametrize('arm', LEARNED)
def test_learned_arm_parses_with_embedded_artifact_and_fails_closed(arm):
    layers = tuple(range(3, 32, 4))
    artifact = random_artifact(arm, layers, 32)
    memory = arm_memory_config(arm, artifacts=dict(ranker=artifact, ranker_sha256=canonical_sha256(artifact)))
    spec = parse_episodic_spec(_with(memory))
    assert isinstance(spec, GdnSweepSpec) and spec.is_learned and spec.score == arm
    assert spec.feature_layers_for(7) == ((6,) if ARMS[arm]['window'] == 1 else (4, 5, 6))
    assert set(spec.feature_pairs) == ({a - 1 for a in layers} if ARMS[arm]['window'] == 1 else
                                       {g for a in layers for g in range(a - 3, a)})
    assert 'ranker_json' not in spec.to_dict() and spec.to_dict()['ranker_sha256'] == canonical_sha256(artifact)
    with pytest.raises(ValueError, match='requires an embedded frozen ranker'):
        parse_episodic_spec(_with(arm_memory_config(arm)))
    tampered = deepcopy(artifact)
    tampered['layers']['7']['mean']['b64'] = artifact['layers']['11']['mean']['b64']
    bad = arm_memory_config(arm, artifacts=dict(ranker=tampered, ranker_sha256=canonical_sha256(artifact)))
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        parse_episodic_spec(_with(bad))
    other = 'L3' if arm != 'L3' else 'L2'                        # wrong schema/window for this arm
    foreign = random_artifact(other, layers, 32)
    bad = arm_memory_config(arm, artifacts=dict(ranker=foreign, ranker_sha256=canonical_sha256(foreign)))
    with pytest.raises(ValueError):
        parse_episodic_spec(_with(bad))
    stripped = deepcopy(artifact)
    del stripped['layers']['31']
    bad = arm_memory_config(arm, artifacts=dict(ranker=stripped, ranker_sha256=canonical_sha256(stripped)))
    with pytest.raises(ValueError, match='exactly the episodic attention layers'):
        parse_episodic_spec(_with(bad))


def test_flatten_order_is_layer_then_head_then_channel():
    a = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    b = 100 + a
    flat = flatten_window({4: a, 5: b}, (4, 5))
    assert flat[0].tolist() == [0, 1, 2, 3, 4, 5, 100, 101, 102, 103, 104, 105]


@pytest.fixture
def tiny8(monkeypatch):
    for name in ['FusedRMSNormGated', 'chunk_gated_delta_rule', 'fused_recurrent_gated_delta_rule',
                 'causal_conv1d_fn', 'causal_conv1d_update']:
        monkeypatch.setattr(native, name, None)
    monkeypatch.setitem(sys.modules, 'flash_attn', SimpleNamespace(
        flash_attn_varlen_func=_cpu_varlen, flash_attn_func=_cpu_flash))
    register_window_attention()
    torch.manual_seed(11)
    types = (['linear_attention'] * 3 + ['full_attention']) * 2
    cfg = Qwen3_5TextConfig(
        hidden_size=48, intermediate_size=64, num_hidden_layers=8, num_attention_heads=4,
        num_key_value_heads=2, head_dim=12, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=6, vocab_size=64, layer_types=types,
        rope_parameters={'rope_type': 'default', 'rope_theta': 1e6, 'partial_rotary_factor': 1.0,
                         'mrope_section': [2, 2, 2]},
        attn_implementation='simplememvln_step_attention')
    model = native.Qwen3_5TextModel(cfg).float().eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.2)
    return model


def learned_spec(model, arm, recent=3, budget=600, seed=0):
    from qwen_vl.research.gdn_ranker import validate_artifact
    pairs = scoring_pairs_from_layer_types(model.config.layer_types)
    attention = tuple(a for _, a in pairs)
    artifact = random_artifact(arm, attention, model.config.linear_num_value_heads, seed=seed)
    validate_artifact(artifact, arm, attention, list(model.config.layer_types), model.config.linear_num_value_heads)
    unit, reduction, backend, statistic = LEARNED_REGISTRY[arm]
    import json
    return GdnSweepSpec(
        version='episodic_window_gdn_sweep_v1', recent_groups=recent, budget_tokens=budget, granularity=unit,
        score=arm, scoring_pairs=pairs, selector_seed=429, per_source_cap=None, score_backend=backend,
        score_segmentation='prefix_obs_response_block64_v1', head_reduction=reduction, statistic=statistic,
        ranker_sha256=canonical_sha256(artifact),
        ranker_json=json.dumps(artifact, sort_keys=True, separators=(',', ':')), **FIXED)


@pytest.mark.parametrize('arm', LEARNED)
def test_learned_offline_training_equals_incremental_serving(tiny8, arm):
    spec = learned_spec(tiny8, arm)
    layout, embeds, positions = _episode(9, seed=1)
    diagnostics = {}
    with torch.no_grad():
        offline = _offline(tiny8, spec, layout, embeds, positions, diagnostics=diagnostics)
        online, state = _online(tiny8, spec, layout, embeds, positions)
    torch.testing.assert_close(online, offline, rtol=2e-4, atol=2e-5)
    last = layout.groups - 1
    for layer in spec.attention_layers:
        final = np.sort(state.running[layer].ids)
        np.testing.assert_array_equal(final, diagnostics['plans'][layer].group_ids(last))
        assert final.size == min(spec.budget_tokens, max(0, last - spec.recent_groups + 1) * 300)
    for group in state.groups:            # resident groups still hold their fixed learned priorities
        assert group.features == {}       # window features are released once scored
        for layer in spec.attention_layers:
            np.testing.assert_allclose(group.scores[layer],
                                       diagnostics['visual_scores'][layer][group.step_id].numpy(),
                                       rtol=1e-4, atol=1e-5)


def test_learned_scores_are_frame_batched_and_deterministic(tiny8):
    spec = learned_spec(tiny8, 'L3')
    ranker = spec.ranker()
    x = torch.randn(900, ranker.layers[3]['mean'].numel())
    full = ranker.score(3, x)
    parts = torch.cat([ranker.score(3, x[i:i + 300]) for i in (0, 300, 600)])
    assert torch.equal(full, parts) and full.dtype == torch.float32
    with pytest.raises(ValueError):
        ranker.score(3, x[:299])


def test_learned_plan_ignores_future_and_keeps_gradients_out_of_the_ranker(tiny8):
    spec = learned_spec(tiny8, 'L3', recent=2, budget=300)
    layout, embeds, positions = _episode(7, seed=3)
    params = sum(p.numel() for p in tiny8.parameters())
    leaf = embeds.clone().requires_grad_()
    tiny8.train()
    out = _offline(tiny8, spec, layout, leaf, positions)
    out[0, layout.group_spans[-1][0]:].pow(2).sum().backward()
    tiny8.eval()
    assert leaf.grad is not None and sum(p.numel() for p in tiny8.parameters()) == params
    changed = embeds.clone()
    changed[:, layout.group_spans[4][0]:] += 3.0
    with torch.no_grad():
        base = _offline(tiny8, spec, layout, embeds, positions)
        moved = _offline(tiny8, spec, layout, changed, positions)
    cut = layout.group_spans[4][0]
    torch.testing.assert_close(moved[:, :cut], base[:, :cut], rtol=0, atol=0)


def test_learned_checkpointing_is_exact(tiny8):
    spec = learned_spec(tiny8, 'L2', recent=2, budget=300)
    layout, embeds, positions = _episode(7, seed=5)

    def run(checkpoint):
        leaf = embeds.clone().requires_grad_()
        tiny8.train()
        if checkpoint:
            tiny8.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        else:
            tiny8.gradient_checkpointing_disable()
        out = _offline(tiny8, spec, layout, leaf, positions)
        grad, = torch.autograd.grad(out[0, layout.group_spans[-1][0]:].pow(2).sum(), leaf)
        tiny8.eval()
        return out.detach(), grad
    a, ga = run(False)
    b, gb = run(True)
    tiny8.gradient_checkpointing_disable()
    torch.testing.assert_close(b, a, rtol=0, atol=0)
    torch.testing.assert_close(gb, ga, rtol=1e-6, atol=1e-7)
