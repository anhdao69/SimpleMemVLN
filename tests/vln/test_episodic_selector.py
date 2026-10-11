"""Deterministic running top-B, causality and random-key reproducibility."""
import numpy as np
import pytest
import torch
from qwen_vl.data.episodic_metadata import EpisodicLayout, build_episodic_layout
from qwen_vl.research.episodic_contract import EpisodicSpec, QWEN35_4B_PAIRS
from qwen_vl.stream.episodic import (
    RunningTopB, as_priorities, batch_reference_selection, build_layer_plan, random_priorities)


def _spec(recent=8, budget=300, score='committed_mean_l2_v1'):
    return EpisodicSpec('episodic_window_v1', recent, budget, 'token', score, QWEN35_4B_PAIRS,
                        429, None, 'b', 's', 'a')


def _layout(groups, prefix=11, extra=(3, 1, 5)):
    spans, obs, starts, cursor = [], [], [], prefix
    for t in range(groups):
        lead = 4 + t % 3
        starts.append(cursor + lead)
        observed = cursor + lead + 300 + 6
        end = observed + extra[t % len(extra)]
        spans.append((cursor, end)); obs.append(observed); cursor = end
    return EpisodicLayout(prefix, tuple(spans), tuple(obs), tuple(starts))


@pytest.mark.parametrize('groups,recent,budget', [(1, 8, 300), (8, 8, 300), (9, 8, 300), (10, 8, 300),
                                                  (23, 8, 300), (17, 4, 1200), (14, 8, 1000), (12, 3, 50)])
def test_running_plan_equals_batch_reference_for_every_prefix(groups, recent, budget):
    rng = np.random.default_rng(groups * 7 + recent)
    layout = _layout(groups)
    # Quantized scores force many exact ties.
    scores = torch.tensor(rng.integers(0, 40, (groups, 300)) / 8, dtype=torch.float32)
    plan = build_layer_plan(_spec(recent, budget), 'ep', 7, layout, scores)
    visual = layout.visual_matrix().numpy()
    prios = [as_priorities(scores[g]) for g in range(groups)]
    for t in range(groups):
        expected = batch_reference_selection(budget, visual, prios, t - recent)
        np.testing.assert_array_equal(plan.group_ids(t), expected)
        assert plan.group_ids(t).size == min(budget, max(0, t - recent + 1) * 300)
        if plan.group_ids(t).size:
            assert plan.group_ids(t).max() < layout.group_spans[t - recent + 1][0]
            assert np.all(np.diff(plan.group_ids(t)) > 0)
    plan.verify()


def test_tie_break_is_priority_desc_then_original_id_asc():
    running = RunningTopB(3, random_keys=False)
    running.admit(np.array([10, 11, 12, 13]), np.array([1., 2., 2., 2.]), 0)
    np.testing.assert_array_equal(running.ids, [11, 12, 13])
    running.admit(np.array([20, 21]), np.array([2., 3.]), 1)
    np.testing.assert_array_equal(running.ids, [11, 12, 21])


def test_future_scores_never_change_earlier_plans_and_b0_needs_nothing():
    layout = _layout(20)
    rng = np.random.default_rng(3)
    base = torch.tensor(rng.random((20, 300)), dtype=torch.float32)
    changed = base.clone(); changed[12:] = torch.tensor(rng.random((8, 300)), dtype=torch.float32)
    a = build_layer_plan(_spec(), 'ep', 3, layout, base)
    b = build_layer_plan(_spec(), 'ep', 3, layout, changed)
    for t in range(20):   # group t sees sources <= t-8; scores of groups >=12 matter from t>=20
        np.testing.assert_array_equal(a.group_ids(t), b.group_ids(t))
    empty = build_layer_plan(_spec(8, 0, 'none'), 'ep', 3, layout)
    assert empty.token_ids.size == 0 and empty.groups == 20
    with pytest.raises(ValueError):
        build_layer_plan(_spec(8, 0, 'none'), 'ep', 3, layout, base)


def test_random_keys_reproducible_across_chunking_and_differ_by_layer():
    ids = np.arange(1000, 1600)
    whole = random_priorities(429, 'rxrce:x:1', 7, ids)
    chunked = np.concatenate([random_priorities(429, 'rxrce:x:1', 7, ids[i:i + 37]) for i in range(0, 600, 37)])
    np.testing.assert_array_equal(whole, chunked)
    assert whole.dtype == np.uint64 and len(np.unique(whole)) == 600
    other = random_priorities(429, 'rxrce:x:1', 11, ids)
    assert np.mean(whole == other) < 0.01
    layout = _layout(15)
    p3 = build_layer_plan(_spec(score='random_hash_v1'), 'u', 3, layout)
    p7 = build_layer_plan(_spec(score='random_hash_v1'), 'u', 7, layout)
    np.testing.assert_array_equal(np.diff(p3.group_offsets), np.diff(p7.group_offsets))
    assert not np.array_equal(p3.token_ids, p7.token_ids)
    again = build_layer_plan(_spec(score='random_hash_v1'), 'u', 3, layout)
    assert again.source_digest == p3.source_digest
    with pytest.raises(ValueError):
        build_layer_plan(_spec(score='random_hash_v1'), 'u', 3, layout, torch.zeros(15, 300))


def test_validation_rejects_nonfinite_duplicates_and_tamper():
    layout = _layout(10)
    bad = torch.zeros(10, 300); bad[0, 5] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        build_layer_plan(_spec(), 'e', 3, layout, bad)
    with pytest.raises(TypeError):
        build_layer_plan(_spec(), 'e', 3, layout, torch.zeros(10, 300, dtype=torch.float64))
    running = RunningTopB(5, random_keys=False)
    with pytest.raises(ValueError, match='Duplicate'):
        running.admit(np.array([1, 1]), np.array([1., 2.]), 0)
    plan = build_layer_plan(_spec(), 'e', 3, layout, torch.rand(10, 300))
    with pytest.raises(ValueError):
        plan.token_ids[0] = 5            # read-only arrays
    object.__setattr__(plan, 'source_digest', 'x' * 64)
    with pytest.raises(RuntimeError, match='changed'):
        plan.verify()


def test_layout_builder_validates_real_token_geometry():
    image = 99
    ids = [1] * 5
    obs, spans, cursor = [], [], 5
    for t in range(3):
        group = [2, 3] + [image] * 300 + [4, 5] + [6, 7, 8]
        obs.append((cursor, cursor + 304)); spans.append((cursor, cursor + 307))
        ids += group; cursor += 307
    layout = build_episodic_layout(torch.tensor([ids]), image, 5, obs, spans)
    assert layout.visual_starts == (7, 314, 621)
    assert layout.score_segments()[:3] == ((0, 5), (5, 309), (309, 312))
    broken = list(ids); broken[100] = 0
    with pytest.raises(ValueError, match='image tokens'):
        build_episodic_layout(torch.tensor([broken]), image, 5, obs, spans)
