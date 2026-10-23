"""Near/tail horizon plans, estimator weights, missingness and row/context sampling."""
import itertools
import math
import random
import numpy as np
import pytest
from qwen_vl.research.gdn_label_horizons import (
    aggregate_horizon_labels, displaced_row, make_horizon_plan, plan_sha256, remaining_query_weights,
    sample_patch_positions)


@pytest.mark.parametrize('remaining', [0, 1, 2, 3, 4, 5, 16, 17, 18, 20, 21, 40, 137])
def test_supports_offsets_and_no_post_terminal_query(remaining):
    source, recent = 3, 4
    steps = source + recent + remaining
    plan = make_horizon_plan('ep', source, steps, 429)
    assert plan.admission_group == source + recent
    assert plan.near_support == min(16, remaining)
    assert plan.tail_support == remaining - plan.near_support
    assert len(plan.near_offsets) == min(4, plan.near_support)
    assert len(plan.tail_offsets) == min(4, plan.tail_support)
    assert all(0 <= u < plan.near_support for u in plan.near_offsets)
    assert all(plan.near_support <= u < remaining for u in plan.tail_offsets)
    assert list(plan.near_offsets) == sorted(set(plan.near_offsets))
    assert all(step < steps for step in plan.query_steps())
    if plan.near_support <= 4:
        assert plan.near_offsets == tuple(range(plan.near_support))
    if 0 < plan.tail_support <= 4:
        assert plan.tail_offsets == tuple(range(plan.near_support, remaining))


def test_source_shorter_than_window_has_no_support():
    plan = make_horizon_plan('ep', 2, 5, 429)
    assert plan.remaining_support == 0 and plan.offsets == ()
    agg = aggregate_horizon_labels(plan, {})
    assert (agg.near_status, agg.tail_status, agg.remaining_status) == ('not_eligible', 'not_applicable', 'not_eligible')


def test_deterministic_local_rng_without_global_state():
    random.seed(1)
    np.random.seed(1)
    state = (random.getstate(), np.random.get_state()[1].copy())
    a = make_horizon_plan('ep-x', 5, 80, 429)
    random.random()
    np.random.random()
    b = make_horizon_plan('ep-x', 5, 80, 429)
    assert a == b and plan_sha256(a) == plan_sha256(b)
    np.random.seed(1)
    random.seed(1)
    assert random.getstate() == state[0]
    c = make_horizon_plan('ep-x', 5, 80, 430)
    d = make_horizon_plan('ep-y', 5, 80, 429)
    assert (a.near_offsets, a.tail_offsets) != (c.near_offsets, c.tail_offsets) or \
        (a.near_offsets, a.tail_offsets) != (d.near_offsets, d.tail_offsets)


def test_near_and_tail_streams_are_independent():
    a = make_horizon_plan('ep', 1, 60, 429)
    b = make_horizon_plan('ep', 1, 60, 429, existing_near_offsets=a.near_offsets)
    assert b.tail_offsets == a.tail_offsets and b.near_origin == 'imported'
    with pytest.raises(ValueError):
        make_horizon_plan('ep', 1, 60, 429, existing_near_offsets=(0, 0, 1, 2))
    with pytest.raises(ValueError):
        make_horizon_plan('ep', 1, 60, 429, existing_near_offsets=(0, 1, 2, 16))


def test_dense_plan_enumerates_every_query():
    plan = make_horizon_plan('ep', 0, 30, 429, dense=True)
    assert plan.near_offsets == tuple(range(16)) and plan.tail_offsets == tuple(range(16, 26))


def test_missing_is_never_zero():
    plan = make_horizon_plan('ep', 0, 40, 429)
    values = {u: 1.0 for u in plan.offsets}
    values[plan.tail_offsets[0]] = None
    agg = aggregate_horizon_labels(plan, values)
    assert agg.near_mean == 1.0 and agg.near_status == 'complete'
    assert agg.tail_mean is None and agg.tail_status == 'missing'
    assert agg.remaining_mean is None and agg.remaining_status == 'missing'
    values = {u: 1.0 for u in plan.offsets}
    del values[plan.near_offsets[0]]
    agg = aggregate_horizon_labels(plan, values)
    assert agg.near_status == 'missing' and agg.remaining_mean is None
    with pytest.raises(ValueError):
        aggregate_horizon_labels(plan, {999: 1.0})
    with pytest.raises(ValueError):
        aggregate_horizon_labels(plan, {u: -1.0 for u in plan.offsets})


def test_no_tail_remaining_equals_near():
    plan = make_horizon_plan('ep', 0, 14, 429)             # M = 10 <= 16
    agg = aggregate_horizon_labels(plan, {u: float(u) for u in plan.offsets})
    assert agg.tail_status == 'not_applicable' and agg.tail_mean is None
    assert agg.remaining_mean == agg.near_mean


@pytest.mark.parametrize('remaining', [7, 18, 23])
def test_stratified_estimator_is_unbiased_by_exact_enumeration(remaining):
    """Average over every equally likely (near, tail) sample equals the raw remaining-mean."""
    rng = np.random.default_rng(0)
    raw = rng.exponential(size=remaining)                    # raw utilities (not logs)
    plan = make_horizon_plan('ep', 0, 4 + remaining, 429)
    n, f = plan.near_support, plan.tail_support
    estimates = []
    near_sets = list(itertools.combinations(range(n), min(4, n)))
    tail_sets = list(itertools.combinations(range(n, remaining), min(4, f))) if f else [()]
    for near in near_sets:
        for tail in tail_sets:
            fake = plan.__class__(**{**plan.__dict__, 'near_offsets': near, 'tail_offsets': tail})
            agg = aggregate_horizon_labels(fake, {u: raw[u] for u in near + tail})
            estimates.append(agg.remaining_mean)
    assert math.isclose(np.mean(estimates), raw.mean(), rel_tol=1e-12)
    # Equal-weight averaging of the 8 queried values is biased when N != F.
    if f and f != n:
        naive = [np.mean([raw[u] for u in near + tail]) for near in near_sets for tail in tail_sets]
        assert not math.isclose(np.mean(naive), raw.mean(), rel_tol=1e-6)


def test_weights_and_log_after_aggregation():
    plan = make_horizon_plan('ep', 0, 40, 429)
    weights = remaining_query_weights(plan)
    assert math.isclose(sum(w[3] for w in weights.values()), 1.0)
    values = {u: (u + 1.0) ** 2 for u in plan.offsets}
    agg = aggregate_horizon_labels(plan, values)
    assert math.isclose(agg.remaining_mean, sum(weights[u][3] * values[u] for u in plan.offsets))
    kappa = 3.0
    weighted_logs = (plan.near_support * math.log1p(agg.near_mean / kappa)
                     + plan.tail_support * math.log1p(agg.tail_mean / kappa)) / plan.remaining_support
    assert not math.isclose(math.log1p(agg.remaining_mean / kappa), weighted_logs)


def test_patch_sampling_two_per_tile_and_stable():
    pos = sample_patch_positions(429, 'ep', 3)
    assert len(pos) == 24 and len(set(pos)) == 24
    tiles = {}
    for p in pos:
        r, c = divmod(p, 20)
        tiles.setdefault((r // 5, c // 5), []).append(p)
    assert len(tiles) == 12 and all(len(v) == 2 for v in tiles.values())
    assert pos == sample_patch_positions(429, 'ep', 3)
    assert pos != sample_patch_positions(429, 'ep', 4)


def test_displacement_keyed_by_context_not_candidate():
    bank = list(range(100, 1300))
    d = displaced_row(429, 'ep', 2, 9, 7, bank)
    assert d in bank and d == displaced_row(429, 'ep', 2, 9, 7, bank)
    others = {displaced_row(429, 'ep', 2, 9, layer, bank) for layer in (3, 11, 15, 19, 23, 27, 31)}
    assert len(others | {d}) > 1
    with pytest.raises(ValueError):
        displaced_row(429, 'ep', 2, 9, 7, [5, 3])
