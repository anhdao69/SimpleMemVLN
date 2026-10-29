"""Learned S8N retention rule (stream/sweep_retention.RetentionTilt) on a tiny real Qwen3.5 (FP32 CPU).

* zero tilt keeps exactly S8N's frames; extreme birth tilts give recency / episode-start memory;
  full consensus gives every layer the same ranking;
* the streaming tilted selection equals the offline batch reference on the tilted priorities, and the
  realised record replays the episode exactly (what the RL trainer relies on).
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vln'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'memory_sweep'))

from test_episodic_forward import _episode, _offline, tiny  # noqa: E402,F401  (fixture)
from test_gdn_sweep_forward import sweep_spec  # noqa: E402
from test_s8n_rollout import Stream, _injected_forward  # noqa: E402

from qwen_vl.rl.sweep_memory import injected_layer_plans  # noqa: E402
from qwen_vl.stream.gdn_selectors import build_sweep_layer_plan  # noqa: E402
from qwen_vl.stream.gdn_sweep_state import GdnSweepSessionState, _Group  # noqa: E402
from qwen_vl.stream.sweep_retention import (RetentionTilt, load_sweep_retention,  # noqa: E402
                                            save_sweep_retention)


def run(model, spec, layout, embeds, positions, tilt):
    stream = Stream(model, spec, layout, embeds, positions)
    stream.state.tilt = tilt
    out = torch.cat([stream.group() for _ in range(layout.groups)], 1)
    return stream, out


def test_zero_tilt_keeps_the_s8n_frames(tiny):
    spec = sweep_spec(tiny, 'S8N', recent=2, budget=600)
    layout, embeds, positions = _episode(10, seed=3)
    with torch.no_grad():
        plain, plain_out = run(tiny, spec, layout, embeds, positions, None)
        zero, zero_out = run(tiny, spec, layout, embeds, positions, RetentionTilt.zeros(spec.attention_layers))
    assert plain.state.export() == zero.state.export()
    assert torch.equal(plain_out, zero_out)


@pytest.mark.parametrize('birth,expected', [(100.0, 'latest'), (-100.0, 'first')])
def test_extreme_birth_tilts_give_recency_or_start_memory(tiny, birth, expected):
    spec = sweep_spec(tiny, 'S8N', recent=2, budget=600)
    layout, embeds, positions = _episode(10, seed=3)
    layers = spec.attention_layers
    with torch.no_grad():
        stream, _ = run(tiny, spec, layout, embeds, positions,
                        RetentionTilt(layers, [0.0] * len(layers), [birth] * len(layers)))
    expired = layout.groups - 1 - spec.recent_groups        # sources admitted by the last promotion
    want = [expired - 1, expired] if expected == 'latest' else [0, 1]
    for layer in layers:
        assert stream.state.running[layer].frame_sources.tolist() == want


def test_full_consensus_gives_every_layer_one_ranking(tiny):
    spec = sweep_spec(tiny, 'S8N', recent=2, budget=600)
    layout, embeds, positions = _episode(10, seed=6)
    layers = spec.attention_layers
    with torch.no_grad():
        plain, _ = run(tiny, spec, layout, embeds, positions, None)
        shared, _ = run(tiny, spec, layout, embeds, positions,
                        RetentionTilt(layers, [1.0] * len(layers), [0.0] * len(layers)))
    for banks in shared.state.export()['layer_banks'].values():
        assert len({tuple(v) for v in banks.values()}) == 1
    # The tiny model's layers disagree under plain S8N, so the test is not vacuous.
    assert any(len({tuple(v) for v in banks.values()}) > 1 for banks in plain.state.export()['layer_banks'].values())


def test_tilted_stream_equals_the_offline_reference_and_replays_exactly(tiny):
    spec = sweep_spec(tiny, 'S8N', recent=2, budget=600)
    layout, embeds, positions = _episode(11, seed=4)
    layers = spec.attention_layers
    tilt = RetentionTilt(layers, [0.5, -0.3][:len(layers)] + [0.2] * (len(layers) - 2),
                         [0.7, 1.4][:len(layers)] + [0.3] * (len(layers) - 2))
    diagnostics = {}
    with torch.no_grad():
        _offline(tiny, spec, layout, embeds, positions, diagnostics=diagnostics)
        stream, stream_out = run(tiny, spec, layout, embeds, positions, tilt)
        plans = injected_layer_plans(spec, layout, stream.state.export()['layer_banks'])
        replayed = _injected_forward(tiny, spec, layout, embeds, positions, plans)
        reference = {}
        for layer in layers:
            drift = {l: diagnostics['frame_scores'][l] for l in layers}
            tilted = np.asarray([tilt.priority({l: drift[l][g] for l in layers}, layer, g)
                                 for g in range(layout.groups)], dtype=np.float64)
            reference[layer] = build_sweep_layer_plan(spec, 'ep', layer, layout, frame_scores=tilted)
        plain_plans = injected_layer_plans(spec, layout, run(tiny, spec, layout, embeds, positions, None)[0]
                                           .state.export()['layer_banks'])
    for layer in layers:
        np.testing.assert_array_equal(plans[layer].token_ids, reference[layer].token_ids)
        np.testing.assert_array_equal(plans[layer].group_offsets, reference[layer].group_offsets)
    assert any(not np.array_equal(plans[l].token_ids, plain_plans[l].token_ids) for l in layers)
    stream_rows = torch.cat([stream_out], 1)
    assert torch.allclose(replayed, _injected_forward(tiny, spec, layout, embeds, positions, reference))
    assert torch.isfinite(stream_rows).all()


def test_tilt_validation_and_round_trip(tmp_path):
    tilt = RetentionTilt([3, 7], [0.1, -0.2], [0.3, 0.0])
    assert RetentionTilt.from_dict(tilt.to_dict()).to_dict() == tilt.to_dict()
    np.testing.assert_array_equal(tilt.vector(), [0.1, -0.2, 0.3, 0.0])
    assert RetentionTilt.from_vector([3, 7], tilt.vector()).to_dict() == tilt.to_dict()
    assert load_sweep_retention(tmp_path) is None
    save_sweep_retention(tilt, tmp_path)
    assert load_sweep_retention(tmp_path).sha256() == tilt.sha256()
    scores = {3: math.e, 7: math.e ** 3}
    # (1 - a) log p_l + a mean + b log(1 + s): layer 3, a = 0.1, b = 0.3, s = 4
    assert tilt.priority(scores, 3, 4) == pytest.approx(0.9 * 1 + 0.1 * 2 + 0.3 * math.log(5))
    for bad in ({3: 0.0, 7: 1.0}, {3: float('nan'), 7: 1.0}, {3: 1.0}):
        with pytest.raises(ValueError):
            tilt.priority(bad, 3, 0)
    with pytest.raises(ValueError):
        RetentionTilt([3, 7], [0.1], [0.0, 0.0])
    with pytest.raises(ValueError):
        RetentionTilt.from_dict(dict(tilt.to_dict(), schema='other'))


def test_tilt_is_rejected_for_non_drift_arms(tiny):
    spec = sweep_spec(tiny, 'C_RECENCY', recent=2, budget=600)
    wrapper = SimpleNamespace(backbone=SimpleNamespace(model=SimpleNamespace(language_model=tiny)))
    state = GdnSweepSessionState(spec, wrapper, 'ep', 'cpu')
    state.tilt = RetentionTilt.zeros(spec.attention_layers)
    group = _Group(0, 0, visual_start=0)
    with pytest.raises(ValueError, match='S8 frame drift'):
        state._priorities(group, spec.attention_layers[0], np.arange(300))
