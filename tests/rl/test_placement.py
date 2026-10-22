from qwen_vl.rl.placement import GIB, plan


def test_long_episodes_go_where_they_fit_and_every_item_is_assigned_once():
    free = [70 * GIB, 30 * GIB]
    lengths = [[60000, 10000], [50000, 8000, 9000]]
    a = plan(free, lengths)
    assert set(a) == {(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)}
    assert a[(0, 0)] == 0 and a[(1, 0)] == 0      # only rank 0 can hold 50-60k-token episodes
    assert all(t in (0, 1) for t in a.values())


def test_balanced_when_memory_is_ample_and_owner_preferred_on_ties():
    a = plan([90 * GIB, 90 * GIB], [[10000, 10000], [10000, 10000]])
    assert [a[(0, 0)], a[(0, 1)], a[(1, 0)], a[(1, 1)]].count(0) == 2
    assert a == plan([90 * GIB, 90 * GIB], [[10000, 10000], [10000, 10000]])  # deterministic
