"""M5 fork snapshots on a tiny real Qwen3.5 text model (FP32 CPU, torch fallbacks).

A branch restored from a snapshot of the stream cache plus ``clone_episodic(state)`` must
continue exactly as the original session would, and continuing either copy must leave the
other untouched (independent bank, recent groups and committed-write scorer state).
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vln'))

from test_episodic_forward import _episode, _spec, tiny  # noqa: E402,F401  (fixture)

from qwen_vl.models.episodic_forward import episodic_text_forward  # noqa: E402
from qwen_vl.rl.sampling import clone_cache, clone_episodic  # noqa: E402
from qwen_vl.stream.cache import make_stream_cache_fp32  # noqa: E402
from qwen_vl.stream.episodic_cache import EpisodicSessionState  # noqa: E402


class Stream:
    """The serving append sequence of ``StreamSession`` for one synthetic episode."""

    def __init__(self, model, spec, layout, embeds, positions):
        self.model, self.spec, self.layout, self.embeds, self.positions = model, spec, layout, embeds, positions
        self.wrapper = type('W', (), {})()
        self.wrapper.backbone = type('B', (), {})()
        self.wrapper.backbone.model = type('M', (), {})()
        self.wrapper.backbone.model.language_model = model
        self.state = EpisodicSessionState(spec, self.wrapper, 'ep', 'cpu')
        self.cache = make_stream_cache_fp32(model.config)
        self.append(0, layout.prefix, 'prefix')
        self.next_group = 0

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
        self.state.promote(self.cache, t)
        self.state.open_group(t, start, layout.visual_starts[t])
        outs = [self.append(start, observed, 'observation', layout.visual_starts[t] - start)]
        outs += [self.append(token, token + 1, 'response_token') for token in range(observed, end - 2)]
        outs.append(self.append(end - 2, end, 'response_close'))
        self.state.close_group(end)
        self.next_group += 1
        return torch.cat(outs, 1)

    def fork(self):
        twin = object.__new__(Stream)
        twin.__dict__.update(self.__dict__)
        twin.cache, twin.state = clone_cache(self.cache), clone_episodic(self.state)
        return twin


def _bank(stream):
    return {layer: stream.state.running[layer].ids.copy() for layer in stream.spec.attention_layers}


def test_restored_branch_continues_exactly_and_independently(tiny):
    spec = _spec(tiny, 2, 450, 'committed_mean_l2_v1')
    layout, embeds, positions = _episode(9, seed=5)
    with torch.no_grad():
        main = Stream(tiny, spec, layout, embeds, positions)
        for _ in range(4):                       # bank already holds expired groups at the fork
            main.group()
        branch = main.fork()
        scorer_state = {k: s.state.clone() for k, s in branch.state.scorers.items()}
        main_out = torch.cat([main.group() for _ in range(5)], 1)
        # Continuing the main session must not have touched the snapshot copy.
        for k, s in branch.state.scorers.items():
            assert torch.equal(s.state, scorer_state[k])
        branch_out = torch.cat([branch.group() for _ in range(5)], 1)
    assert torch.equal(main_out, branch_out)
    for layer, ids in _bank(main).items():
        np.testing.assert_array_equal(ids, branch.state.running[layer].ids)
        assert ids.size == 450
    for a, b in zip(main.cache.layers, branch.cache.layers):
        for name in ('keys', 'values'):
            if getattr(a, name, None) is not None:
                assert torch.equal(getattr(a, name), getattr(b, name))


def test_branch_with_different_inputs_leaves_main_unchanged(tiny):
    spec = _spec(tiny, 2, 300, 'committed_mean_l2_v1')
    layout, embeds, positions = _episode(8, seed=9)
    other = embeds.clone()
    other[:, layout.group_spans[4][0]:] += 0.5   # the branch sees different future observations
    with torch.no_grad():
        reference = Stream(tiny, spec, layout, embeds, positions)
        reference_out = torch.cat([reference.group() for _ in range(8)], 1)
        main = Stream(tiny, spec, layout, embeds, positions)
        prefix_out = torch.cat([main.group() for _ in range(4)], 1)
        branch = main.fork()
        branch.embeds = other
        branch_out = torch.cat([branch.group() for _ in range(4)], 1)
        main_out = torch.cat([prefix_out] + [main.group() for _ in range(4)], 1)
    assert torch.equal(main_out, reference_out)
    assert not torch.equal(branch_out, reference_out[:, prefix_out.shape[1]:])
    for layer, ids in _bank(reference).items():
        np.testing.assert_array_equal(ids, main.state.running[layer].ids)
