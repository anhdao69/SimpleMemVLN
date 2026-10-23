"""Tiny real Qwen3.5: label hooks reconstruct the native module output; feature extractor parity."""
import numpy as np
import pytest
import torch
from qwen_vl.models.gdn_features import GdnFeatureExtractor, GdnSweepScorer
from qwen_vl.research.kv_utility_labels import FixedQueryContext, native_qkv, visible_rows
from test_episodic_forward import _episode, _offline, tiny  # noqa: F401  (fixture)
from test_gdn_sweep_forward import sweep_spec


@pytest.mark.parametrize('score', ['C_RANDOM', 'S0'])
def test_reconstructed_reference_context_matches_module_output(tiny, score):
    spec = sweep_spec(tiny, score, recent=3, budget=600)
    layout, embeds, positions = _episode(9, seed=4)
    captured = {}

    def hook(module, args, kwargs, output):
        captured[module.layer_idx] = (kwargs['hidden_states'].clone(), kwargs['position_embeddings'],
                                      kwargs['episodic_plan'], output[0].clone())
    handles = [tiny.layers[a].self_attn.register_forward_hook(hook, with_kwargs=True)
               for a in spec.attention_layers]
    with torch.no_grad():
        plain = _offline(tiny, spec, layout, embeds, positions)
        for h in handles:
            h.remove()
        traced = _offline(tiny, spec, layout, embeds, positions)
    assert torch.equal(plain, traced)                       # hooks are passive
    checked = 0
    for layer, (hidden, rope, plan, output) in captured.items():
        module = tiny.layers[layer].self_attn
        with torch.no_grad():
            query, gate, keys, values = native_qkv(module, hidden, rope)
        w_o = module.o_proj.weight.detach().double()
        for u in range(layout.groups):
            bank = plan.group_ids(u)
            first = layout.group_spans[max(0, u - spec.recent_groups + 1)][0]
            p = layout.observation_ends[u] - 1
            ids = torch.from_numpy(visible_rows(layout.prefix, bank, first, p))
            ctx = FixedQueryContext(query[:, p].double(), torch.sigmoid(gate[p].double()),
                                    keys[ids].double(), values[ids].double(), module.scaling, w_o)
            torch.testing.assert_close(ctx.output().float(), output[0, p], rtol=2e-4, atol=2e-5)
            if len(bank):                                   # a wrong context is detected
                wrong = torch.from_numpy(visible_rows(layout.prefix, bank[1:], first, p))
                bad = FixedQueryContext(query[:, p].double(), torch.sigmoid(gate[p].double()),
                                        keys[wrong].double(), values[wrong].double(), module.scaling, w_o)
                assert (bad.output().float() - output[0, p]).abs().max() > 1e-5
                checked += 1
    assert checked > 0


def test_resident_loo_equals_module_without_the_row(tiny):
    """Removing a row from the visible set and recomputing the module equals the LOO utility."""
    spec = sweep_spec(tiny, 'C_RANDOM', recent=3, budget=600)
    layout, embeds, positions = _episode(8, seed=6)
    captured = {}

    def hook(module, args, kwargs, output):
        captured[module.layer_idx] = (kwargs['hidden_states'].clone(), kwargs['position_embeddings'],
                                      kwargs['episodic_plan'])
    layer = spec.attention_layers[0]
    handle = tiny.layers[layer].self_attn.register_forward_hook(hook, with_kwargs=True)
    with torch.no_grad():
        _offline(tiny, spec, layout, embeds, positions)
    handle.remove()
    hidden, rope, plan = captured[layer]
    module = tiny.layers[layer].self_attn
    with torch.no_grad():
        query, gate, keys, values = native_qkv(module, hidden, rope)
    u = layout.groups - 1
    bank = plan.group_ids(u)
    first = layout.group_spans[u - spec.recent_groups + 1][0]
    p = layout.observation_ends[u] - 1
    ids = visible_rows(layout.prefix, bank, first, p)
    ctx = FixedQueryContext(query[:, p].double(), torch.sigmoid(gate[p].double()),
                            keys[torch.from_numpy(ids)].double(), values[torch.from_numpy(ids)].double(),
                            module.scaling, module.o_proj.weight.detach().double())
    candidate = int(bank[7])
    got = float(ctx.resident([layout.prefix + 7])[0])
    keep = torch.from_numpy(ids[ids != candidate])
    reduced = FixedQueryContext(query[:, p].double(), torch.sigmoid(gate[p].double()),
                                keys[keep].double(), values[keep].double(), module.scaling,
                                module.o_proj.weight.detach().double())
    want = float((ctx.output() - reduced.output()).norm() / np.sqrt(module.o_proj.weight.shape[0]))
    assert got == pytest.approx(want, rel=1e-9)
    assert got > 0


def test_feature_extractor_offline_equals_serving_and_matches_scorer(tiny):
    spec = sweep_spec(tiny, 'S1')
    layout, embeds, positions = _episode(6, seed=8)
    gdn = spec.scoring_pairs[0][0]
    layer = tiny.layers[gdn]
    hidden = embeds[0]
    segments = layout.score_segments()
    with torch.no_grad():
        offline = GdnFeatureExtractor(layer, 'cpu').sequence(hidden, segments)
        stateless = GdnFeatureExtractor(layer, 'cpu', with_state=False).sequence(hidden, segments)
        scores, _ = GdnSweepScorer(spec, gdn, layer, 'cpu').score_sequence(hidden, segments)
        serving = GdnFeatureExtractor(layer, 'cpu')
        parts = {k: [] for k in offline}
        kinds = ['prefix'] + [k for t in range(layout.groups) for k in ('observation', 'response')]
        for (start, end), kind in zip(segments, kinds):
            if kind == 'response':
                for token in range(start, end - 1):
                    assert serving.append(hidden[token:token + 1], 'response_token') is None
                out = serving.append(hidden[end - 1:end], 'response_close')
            else:
                out = serving.append(hidden[start:end], kind)
            for k in parts:
                parts[k].append(out[k])
    for name in ('beta', 'log_decay_g', 'value_norm'):
        assert torch.equal(stateless[name], offline[name])          # F0 never reads state
    for name, value in offline.items():
        online = torch.cat(parts[name])
        if value.dtype == torch.bool:
            continue
        torch.testing.assert_close(online, value, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(offline['error_norm'].mean(-1), scores, rtol=0, atol=0)   # S1 = mean_h ||e||
