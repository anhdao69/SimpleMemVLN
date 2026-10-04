import pytest
import torch
from qwen_vl.stream.lane_cache import StepLaneCache
from step_lane_test_utils import tiny_spec


def cache(uid='episode'):
    return StepLaneCache(tiny_spec(), uid, device='cpu')


def test_disjoint_layers_caches_and_clone_protection():
    a, b = cache(), cache('other')
    with torch.no_grad():
        initial = a.initial_for(0, 0)
        initial.add_(3)
        assert torch.equal(a.initial_for(0, 0), torch.zeros_like(initial))
        a.commit(0, 0, 3, initial)
        initial.add_(7)
        assert a.initial_for(0, 3).eq(3).all()
        assert a.initial_for(1, 0).eq(0).all()
        assert b.initial_for(0, 0).eq(0).all()
        with pytest.raises(ValueError):
            a.assert_complete_append(3)
        a.commit(1, 0, 3, torch.zeros_like(initial))
        a.assert_complete_append(3)


def test_zero_write_advances_counts_and_reset_clears_everything():
    c = cache()
    with torch.no_grad():
        for index in (0, 1):
            c.commit(index, 0, 8, c.initial_for(index, 0))
            assert c.initial_for(index, 8).eq(0).all()
        c.assert_complete_append(8)
        c.commit(0, 8, 1, torch.ones(1, 2, 3, 5))
        c.reset('new-episode')
        assert c.episode_uid == 'new-episode'
        for index in (0, 1):
            assert c.initial_for(index, 0).eq(0).all()
        c.assert_complete_append(0)


@pytest.mark.parametrize('case', ['shape', 'dtype', 'nan', 'grad', 'layer', 'negative', 'skipped', 'empty'])
def test_invalid_commit_is_atomic(case):
    c = cache()
    state = torch.ones(1, 2, 3, 5)
    index, start, count = 0, 0, 2
    if case == 'shape': state = state.transpose(-1, -2)
    if case == 'dtype': state = state.bfloat16()
    if case == 'nan': state[0, 0, 0, 0] = float('nan')
    if case == 'grad': state.requires_grad_()
    if case == 'layer': index = 4
    if case == 'negative': start = -1
    if case == 'skipped': start = 1
    if case == 'empty': count = 0
    with torch.no_grad(), pytest.raises(ValueError):
        c.commit(index, start, count, state)
    with torch.no_grad():
        assert c.initial_for(0, 0).eq(0).all()


def test_repeated_commit_offsets_and_training_access_rejected():
    c = cache()
    with pytest.raises(RuntimeError, match='grad'):
        c.initial_for(0, 0)
    with pytest.raises(RuntimeError, match='grad'):
        c.commit(0, 0, 1, torch.zeros(1, 2, 3, 5))
    with torch.no_grad():
        for index, start in ((3, 0), (0, -1), (0, 1)):
            with pytest.raises(ValueError):
                c.initial_for(index, start)
        c.commit(0, 0, 3, c.initial_for(0, 0))
        with pytest.raises(ValueError):
            c.commit(0, 0, 3, torch.zeros(1, 2, 3, 5))
        with pytest.raises(ValueError):
            c.assert_complete_append(-1)


def test_native_eviction_has_no_sidecar_dependency():
    from qwen_vl.stream.cache import evict_kv
    from types import SimpleNamespace
    c = cache()
    layer = SimpleNamespace(is_updated=True, keys=torch.arange(5.).reshape(1, 1, 5, 1),
                            values=torch.arange(5.).reshape(1, 1, 5, 1))
    native = SimpleNamespace(layers=[layer])
    with torch.no_grad():
        c.commit(0, 0, 5, torch.ones(1, 2, 3, 5))
        before = c.initial_for(0, 5)
        evict_kv(native, 1, 2)
        assert torch.equal(c.initial_for(0, 5), before)
