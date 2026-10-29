"""S8N (GDN-sweep frame memory) RL rollouts on a tiny real Qwen3.5 text model (FP32 CPU, torch fallbacks).

* the realised per-layer frame record of an incremental rollout, injected into the full-episode
  training forward, reproduces the offline S8N plans and outputs exactly;
* a branch restored from a snapshot of the stream cache plus ``clone_episodic(state)`` continues
  exactly as the original session would, independently of it (frames, scorer state, record).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vln'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'memory_sweep'))

from test_episodic_forward import _episode, _offline, tiny  # noqa: E402,F401  (fixture)
from test_gdn_sweep_forward import sweep_spec  # noqa: E402

from qwen_vl.models.episodic_forward import episodic_text_forward  # noqa: E402
from qwen_vl.rl.sampling import clone_cache, clone_episodic  # noqa: E402
from qwen_vl.rl.sweep_memory import SweepRolloutState, injected_layer_plans  # noqa: E402
from qwen_vl.stream.cache import make_stream_cache_fp32  # noqa: E402


class Stream:
    """The serving append sequence of ``SamplingSession`` for one synthetic S8N episode."""

    def __init__(self, model, spec, layout, embeds, positions):
        self.model, self.spec, self.layout, self.embeds, self.positions = model, spec, layout, embeds, positions
        wrapper = SimpleNamespace(backbone=SimpleNamespace(model=SimpleNamespace(language_model=model)))
        self.state = SweepRolloutState(spec, wrapper, 'ep', 'cpu')
        self.cache = make_stream_cache_fp32(model.config)
        self.append(0, layout.prefix, 'prefix')
        self.next_group = 0
        self.drift = []

    def append(self, start, end, kind, offset=None):
        self.state.begin_append(kind, start, end - start, offset)
        out = episodic_text_forward(self.model, self.embeds[:, start:end], self.positions[..., start:end],
                                    self.spec, past_key_values=self.cache, use_cache=True,
                                    stream_append=True, serving=self.state)
        self.state.end_append(self.cache)
        return out

    def group(self):
        t, layout = self.next_group, self.layout
        (start, end), observed = layout.group_spans[t], layout.observation_ends[t]
        self.state.promote(self.cache, t, generator=None)
        self.state.promote(self.cache, t)                   # sample_step repeats it as a no-op
        self.state.open_group(t, start, layout.visual_starts[t])
        outs = [self.append(start, observed, 'observation', layout.visual_starts[t] - start)]
        self.drift.append(self.state.observation_drift())
        outs += [self.append(token, token + 1, 'response_token') for token in range(observed, end - 2)]
        outs.append(self.append(end - 2, end, 'response_close'))
        self.state.close_group(end)
        self.next_group += 1
        return torch.cat(outs, 1)

    def fork(self):
        twin = object.__new__(Stream)
        twin.__dict__.update(self.__dict__)
        twin.cache, twin.state, twin.drift = clone_cache(self.cache), clone_episodic(self.state), list(self.drift)
        return twin


def _frames(stream):
    return {layer: stream.state.running[layer].frame_sources.copy() for layer in stream.spec.attention_layers}


def _injected_forward(model, spec, layout, embeds, positions, plans):
    return episodic_text_forward(model, embeds, positions, spec, step_plan=(layout.prefix, layout.group_spans),
                                 layout=layout, episode_uid='ep', window_attention_batch_steps=3, plans=plans)


@pytest.mark.parametrize('recent,budget', [(3, 600), (2, 300), (4, 1200)])
def test_realised_layer_banks_reproduce_offline_s8n_plans(tiny, recent, budget):
    spec = sweep_spec(tiny, 'S8N', recent=recent, budget=budget)
    layout, embeds, positions = _episode(11, seed=4)
    diagnostics = {}
    with torch.no_grad():
        offline = _offline(tiny, spec, layout, embeds, positions, diagnostics=diagnostics)
        stream = Stream(tiny, spec, layout, embeds, positions)
        for _ in range(layout.groups):
            stream.group()
        record = stream.state.export()
        plans = injected_layer_plans(spec, layout, record['layer_banks'])
        replayed = _injected_forward(tiny, spec, layout, embeds, positions, plans)
    for layer in spec.attention_layers:
        reference = diagnostics['plans'][layer]
        np.testing.assert_array_equal(plans[layer].token_ids, reference.token_ids)
        np.testing.assert_array_equal(plans[layer].group_offsets, reference.group_offsets)
        np.testing.assert_array_equal(plans[layer].token_sources, reference.token_sources)
        assert plans[layer].source_digest == reference.source_digest
        # Per-step drift seen by the rollout == the offline frame priorities (fixed at birth).
        np.testing.assert_allclose([d[spec.attention_layers.index(layer)] for d in stream.drift],
                                   diagnostics['frame_scores'][layer], rtol=1e-4)
    assert torch.equal(replayed, offline)
    # Only promotions that changed some layer's frames are recorded; the first is at group R.
    steps = [int(s) for s in record['layer_banks']]
    assert steps == sorted(steps) and steps[0] == recent


def test_injected_plans_follow_the_record_not_the_scores(tiny):
    """A record that differs from the drift top-F (e.g. a BF16 near-tie flip) is what the trainer sees."""
    spec = sweep_spec(tiny, 'S8N', recent=2, budget=300)
    layout, embeds, positions = _episode(6, seed=1)
    layers = spec.attention_layers
    banks = {2: {layers[0]: [0], layers[1]: [0]}, 4: {layers[0]: [2], layers[1]: [1]}}
    plans = injected_layer_plans(spec, layout, banks)
    first = plans[layers[1]]
    visible = [first.token_ids[first.group_offsets[g]:first.group_offsets[g + 1]] for g in range(layout.groups)]
    assert [v.size for v in visible] == [0, 0, 300, 300, 300, 300]
    np.testing.assert_array_equal(visible[3], np.arange(layout.visual_starts[0], layout.visual_starts[0] + 300))
    np.testing.assert_array_equal(visible[5], np.arange(layout.visual_starts[1], layout.visual_starts[1] + 300))
    with torch.no_grad():
        out = _injected_forward(tiny, spec, layout, embeds, positions, plans)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize('banks,message', [
    ({2: {1: [1], 3: [0]}}, 'recent window'),
    ({4: {1: [0, 1], 3: [0]}}, 'Invalid retained'),
    ({4: {1: [1, 0], 3: [0]}}, 'Invalid retained'),
    ({4: {1: [0]}}, 'every attention layer'),
    ({9: {1: [0], 3: [0]}}, 'outside the layout'),
])
def test_invalid_layer_banks_are_rejected(tiny, banks, message):
    spec = sweep_spec(tiny, 'S8N', recent=2, budget=300)
    layout, _, _ = _episode(6, seed=1)
    with pytest.raises(ValueError, match=message):
        injected_layer_plans(spec, layout, banks)


def test_restored_branch_continues_exactly_and_independently(tiny):
    spec = sweep_spec(tiny, 'S8N', recent=2, budget=600)
    layout, embeds, positions = _episode(10, seed=5)
    with torch.no_grad():
        main = Stream(tiny, spec, layout, embeds, positions)
        for _ in range(5):                       # frame memory already full at the fork
            main.group()
        branch = main.fork()
        scorer_state = {k: s.state.clone() for k, s in branch.state.scorers.items()}
        frames_at_fork = _frames(branch)
        main_out = torch.cat([main.group() for _ in range(5)], 1)
        for k, s in branch.state.scorers.items():
            assert torch.equal(s.state, scorer_state[k])
        for layer, sources in frames_at_fork.items():
            np.testing.assert_array_equal(branch.state.running[layer].frame_sources, sources)
        assert max(int(s) for s in branch.state.export()['layer_banks']) < 5
        branch_out = torch.cat([branch.group() for _ in range(5)], 1)
    assert torch.equal(main_out, branch_out)
    assert main.state.export() == branch.state.export() and main.drift == branch.drift
    for layer, sources in _frames(main).items():
        np.testing.assert_array_equal(sources, branch.state.running[layer].frame_sources)
        assert sources.size == 2
    for a, b in zip(main.cache.layers, branch.cache.layers):
        for name in ('keys', 'values', 'recurrent_states', 'conv_states'):
            if getattr(a, name, None) is not None:
                assert torch.equal(getattr(a, name), getattr(b, name))


def test_branch_with_different_inputs_leaves_main_unchanged(tiny):
    spec = sweep_spec(tiny, 'S8N', recent=2, budget=600)
    layout, embeds, positions = _episode(9, seed=9)
    other = embeds.clone()
    other[:, layout.group_spans[4][0]:] += 0.5   # the branch sees different future observations
    with torch.no_grad():
        reference = Stream(tiny, spec, layout, embeds, positions)
        reference_out = torch.cat([reference.group() for _ in range(9)], 1)
        main = Stream(tiny, spec, layout, embeds, positions)
        prefix_out = torch.cat([main.group() for _ in range(4)], 1)
        branch = main.fork()
        branch.embeds = other
        branch_out = torch.cat([branch.group() for _ in range(5)], 1)
        main_out = torch.cat([prefix_out] + [main.group() for _ in range(5)], 1)
    assert torch.equal(main_out, reference_out)
    assert main.state.export() == reference.state.export()
    assert not torch.equal(branch_out, reference_out[:, prefix_out.shape[1]:])
    assert branch.drift[4:] != reference.drift[4:]


def test_snapshot_inside_an_append_is_rejected(tiny):
    spec = sweep_spec(tiny, 'S8N', recent=2, budget=300)
    layout, embeds, positions = _episode(3, seed=2)
    with torch.no_grad():
        stream = Stream(tiny, spec, layout, embeds, positions)
        stream.state.begin_append('observation', 0, 1, 0)
        with pytest.raises(ValueError, match='inside an append'):
            clone_episodic(stream.state)
