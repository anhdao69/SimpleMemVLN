"""Running token/frame selection, causality, no resurrection and controls (Task 2)."""
import numpy as np
import pytest
import torch
from qwen_vl.research.episodic_contract import EpisodicSpec, QWEN35_4B_PAIRS
from qwen_vl.research.gdn_sweep_contract import ARM_REGISTRY, FIXED, GdnSweepSpec
from qwen_vl.stream.episodic import as_priorities, batch_reference_selection, build_layer_plan
from qwen_vl.stream.gdn_selectors import (
    RunningTopFrames, batch_reference_frames, build_sweep_layer_plan, observation_segment_indices)
from test_episodic_selector import _layout


def spec(score, recent=4, budget=1200, seed=429):
    unit, reduction, backend, statistic = ARM_REGISTRY[score]
    return GdnSweepSpec(
        version='episodic_window_gdn_sweep_v1', recent_groups=recent, budget_tokens=budget,
        granularity=unit, score=score, scoring_pairs=QWEN35_4B_PAIRS, selector_seed=seed,
        per_source_cap=None, score_backend=backend,
        score_segmentation='none' if score.startswith('C_') else 'prefix_obs_response_block64_v1',
        head_reduction=reduction, statistic=statistic, **FIXED)


def _assert_no_resurrection(plan, layout, recent):
    visual = layout.visual_matrix().numpy()
    for t in range(1, plan.groups):
        before, after = plan.group_ids(t - 1), plan.group_ids(t)
        new = visual[t - recent] if t - recent >= 0 else np.empty(0, np.int64)
        assert np.isin(after, np.concatenate((before, new))).all()           # bank ∪ newly eligible
        evicted = np.setdiff1d(before, after)
        for later in range(t, plan.groups):
            assert not np.isin(evicted, plan.group_ids(later)).any()          # never returns


@pytest.mark.parametrize('groups,recent,budget', [(1, 4, 1200), (4, 4, 1200), (5, 4, 1200), (9, 4, 1200),
                                                  (23, 4, 1200), (14, 8, 300), (12, 3, 50), (30, 2, 1800)])
def test_token_plan_equals_batch_reference_for_every_prefix(groups, recent, budget):
    rng = np.random.default_rng(groups * 11 + recent)
    layout = _layout(groups)
    scores = torch.tensor(rng.integers(0, 40, (groups, 300)) / 8, dtype=torch.float32)  # many ties
    plan = build_sweep_layer_plan(spec('S1', recent, budget), 'ep', 7, layout, scores)
    visual = layout.visual_matrix().numpy()
    prios = [as_priorities(scores[g]) for g in range(groups)]
    for t in range(groups):
        np.testing.assert_array_equal(plan.group_ids(t), batch_reference_selection(budget, visual, prios, t - recent))
        assert plan.group_ids(t).size == min(budget, max(0, t - recent + 1) * 300)
        if plan.group_ids(t).size:
            assert plan.group_ids(t).max() < layout.group_spans[t - recent + 1][0]
            assert np.all(np.diff(plan.group_ids(t)) > 0)
    _assert_no_resurrection(plan, layout, recent)
    plan.verify()


@pytest.mark.parametrize('groups,recent,budget', [(1, 4, 1200), (6, 4, 1200), (8, 4, 1200), (9, 4, 1200),
                                                  (25, 4, 1200), (12, 2, 1800), (10, 3, 300)])
@pytest.mark.parametrize('score', ['S7', 'S8N', 'S8W'])
def test_frame_plan_keeps_whole_frames_and_equals_batch_reference(groups, recent, budget, score):
    rng = np.random.default_rng(groups + recent)
    layout = _layout(groups)
    frames = rng.integers(0, 5, groups).astype(np.float64)                      # exact frame ties
    plan = build_sweep_layer_plan(spec(score, recent, budget), 'ep', 11, layout, frame_scores=frames)
    starts = np.asarray(layout.visual_starts)
    for t in range(groups):
        ids = plan.group_ids(t)
        expected = batch_reference_frames(budget // 300, starts, frames, t - recent)
        np.testing.assert_array_equal(ids, expected)
        assert ids.size == min(budget, max(0, t - recent + 1) * 300)
        sources = plan.token_sources[plan.group_offsets[t]:plan.group_offsets[t + 1]]
        for source in np.unique(sources):                                          # all-or-none frames
            np.testing.assert_array_equal(ids[sources == source], np.arange(starts[source], starts[source] + 300))
    _assert_no_resurrection(plan, layout, recent)


def test_frame_ties_prefer_the_earlier_source_and_never_split_rows():
    running = RunningTopFrames(2)
    for source, priority in enumerate([1.0, 2.0, 2.0, 2.0]):
        running.admit(np.arange(1000 * source, 1000 * source + 300), priority, source)
    np.testing.assert_array_equal(np.unique(running.sources), [1, 2])
    assert running.ids.size == 600 and running.last_margin == 0.0
    with pytest.raises(ValueError, match='contiguous'):
        running.admit(np.arange(5000, 5300)[::-1], 9.0, 4)
    with pytest.raises(ValueError, match='follow'):
        running.admit(np.arange(10, 310), 9.0, 5)
    with pytest.raises(ValueError, match='Nonfinite'):
        running.admit(np.arange(9000, 9300), float('nan'), 6)


def test_recency_keeps_latest_visual_rows_only_and_is_not_window8():
    layout = _layout(15)
    plan = build_sweep_layer_plan(spec('C_RECENCY'), 'ep', 3, layout)
    visual = layout.visual_matrix().numpy()
    for t in range(15):
        expired = list(range(max(0, t - 4 + 1 - 4), max(0, t - 4 + 1)))           # last four expired
        expected = np.sort(visual[expired].reshape(-1)) if expired else np.empty(0, np.int64)
        np.testing.assert_array_equal(plan.group_ids(t), expected)
        old_text = np.setdiff1d(np.arange(layout.group_spans[0][0], layout.group_spans[max(0, t - 3)][0]),
                                visual.reshape(-1))
        assert not np.isin(old_text, plan.group_ids(t)).any()                      # Window8 would keep these
    _assert_no_resurrection(plan, layout, 4)


def test_random_control_equals_legacy_m5c_and_differs_by_seed():
    layout = _layout(20)
    legacy_spec = EpisodicSpec('episodic_window_v1', 4, 1200, 'token', 'random_hash_v1', QWEN35_4B_PAIRS,
                               429, None, 'b', 's', 'a')
    for layer in (3, 15, 31):
        legacy = build_layer_plan(legacy_spec, 'rxrce:x:1', layer, layout)
        sweep = build_sweep_layer_plan(spec('C_RANDOM'), 'rxrce:x:1', layer, layout)
        assert sweep.source_digest == legacy.source_digest
        np.testing.assert_array_equal(sweep.token_ids, legacy.token_ids)
    other = build_sweep_layer_plan(spec('C_RANDOM', seed=430), 'rxrce:x:1', 3, layout)
    assert other.source_digest != build_sweep_layer_plan(spec('C_RANDOM'), 'rxrce:x:1', 3, layout).source_digest
    with pytest.raises(ValueError):
        build_sweep_layer_plan(spec('C_RANDOM'), 'u', 3, layout, torch.zeros(20, 300))


def test_s0_sweep_plan_equals_legacy_m5_plan():
    layout = _layout(17)
    scores = torch.rand(17, 300, generator=torch.Generator().manual_seed(4))
    legacy_spec = EpisodicSpec('episodic_window_v1', 4, 1200, 'token', 'committed_mean_l2_v1',
                               QWEN35_4B_PAIRS, 429, None, 'b', 's', 'a')
    legacy = build_layer_plan(legacy_spec, 'ep', 7, layout, scores)
    sweep = build_sweep_layer_plan(spec('S0'), 'ep', 7, layout, scores)
    assert sweep.source_digest == legacy.source_digest
    np.testing.assert_array_equal(sweep.cutoff_margins, legacy.cutoff_margins)


@pytest.mark.parametrize('score', ['S1', 'S7'])
def test_future_scores_never_change_earlier_plans(score):
    layout = _layout(20)
    rng = np.random.default_rng(3)
    if score == 'S7':
        base = rng.random(20)
        changed = base.copy(); changed[12:] = rng.random(8) * 100
        a = build_sweep_layer_plan(spec(score), 'ep', 3, layout, frame_scores=base)
        b = build_sweep_layer_plan(spec(score), 'ep', 3, layout, frame_scores=changed)
    else:
        base = torch.tensor(rng.random((20, 300)), dtype=torch.float32)
        changed = base.clone(); changed[12:] = 100
        a = build_sweep_layer_plan(spec(score), 'ep', 3, layout, base)
        b = build_sweep_layer_plan(spec(score), 'ep', 3, layout, changed)
    for t in range(16):                   # group t sees sources <= t-4; sources >= 12 matter from t >= 16
        np.testing.assert_array_equal(a.group_ids(t), b.group_ids(t))
    assert not np.array_equal(a.group_ids(19), b.group_ids(19))


def test_input_validation_fails_closed():
    layout = _layout(10)
    bad = torch.zeros(10, 300); bad[9, 5] = float('nan')                         # even a non-eligible group
    with pytest.raises(ValueError, match='Nonfinite'):
        build_sweep_layer_plan(spec('S1'), 'e', 3, layout, bad)
    with pytest.raises(TypeError):
        build_sweep_layer_plan(spec('S1'), 'e', 3, layout, torch.zeros(10, 300, dtype=torch.float64))
    with pytest.raises(ValueError):
        build_sweep_layer_plan(spec('S1'), 'e', 3, layout, torch.zeros(9, 300))
    with pytest.raises(ValueError):
        build_sweep_layer_plan(spec('S7'), 'e', 3, layout, torch.zeros(10, 300))
    with pytest.raises(ValueError):
        build_sweep_layer_plan(spec('S7'), 'e', 3, layout, frame_scores=np.zeros(10, np.float32))
    with pytest.raises(ValueError, match='Nonfinite'):
        build_sweep_layer_plan(spec('S8W'), 'e', 3, layout, frame_scores=np.full(10, np.inf))
    with pytest.raises(ValueError):
        build_sweep_layer_plan(spec('S1'), 'e', 2, layout, torch.zeros(10, 300))  # not an attention layer


def test_observation_segments_are_located_from_the_layout():
    layout = _layout(5)
    segments = layout.score_segments()
    indices = observation_segment_indices(layout)
    assert indices == (1, 3, 5, 7, 9)
    for index, ((start, _), observed) in zip(indices, zip(layout.group_spans, layout.observation_ends)):
        assert segments[index] == (start, observed)
