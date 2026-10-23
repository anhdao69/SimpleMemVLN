"""Ranker dataset and fitting on synthetic label files in the exact collector format."""
import json
import numpy as np
import pytest
import torch
from qwen_vl.research.gdn_label_horizons import aggregate_horizon_labels, make_horizon_plan
from qwen_vl.research.gdn_ranker import FrozenRanker, canonical_sha256, validate_artifact
from qwen_vl.research.gdn_ranker_dataset import STATUS, RankerData, build, check_partitions
from scripts.memory_sweep import train_ranker
from scripts.memory_sweep.build_labels import episode_file

ATTENTION = list(range(3, 32, 4))
GDN = [g for g in range(32) if g not in ATTENTION]
FIELDS = ['beta', 'log_decay_g', 'value_norm', 'error_norm', 'committed_norm', 'prediction_norm',
          'prediction_dot_value', 'cosine_error', 'key_norm_sq']
R, HV, D = 24, 32, 2560


def _episodes():
    out = []
    for i in range(18):
        partition = ('ranker_fit', 'ranker_dev', 'ranker_holdout')[i % 3]
        dataset = ('r2rce', 'rxrce')[(i // 3) % 2]
        out.append(dict(cell_rank=i, dataset=dataset, dense=False, episode_uid=f'{dataset}:train:sc{i % 6}:{i}:{i}',
                        logger=('C_RANDOM', 'C_RANDOM', 'S0')[i % 3], logger_seed=(7001, 7002, 429)[i % 3],
                        partition=partition, pilot=True, scene_id=f'sc{i % 6}', steps=12 + i, stratum=0))
    return out


def _write(tmp_path, episodes, seed=0):
    rng = np.random.default_rng(seed)
    (tmp_path / 'episodes').mkdir(parents=True, exist_ok=True)
    reference = {}
    for e in episodes:
        T = e['steps']
        sources = list(range(T - 4))
        S = len(sources)
        plans = [make_horizon_plan(e['episode_uid'], s, T, 429) for s in sources]
        features = rng.random((S, R, len(GDN), HV, len(FIELDS)), dtype=np.float32) + 0.05
        features[..., 1] = -features[..., 1]                       # log_decay_g < 0
        features[..., 7] = rng.uniform(-1, 1, features.shape[:-1])  # cosine_error
        utility = np.full((S, len(ATTENTION), R, 8), np.nan)
        for slot, plan in enumerate(plans):
            for k, a in enumerate(ATTENTION):
                # learnable: utility tracks the value norm of the layer before attention, head 3
                signal = features[slot, :, GDN.index(a - 1), 3, 2] ** 4
                for q in range(len(plan.offsets)):
                    utility[slot, k, :, q] = signal * (1 + 0.1 * rng.random(R)) + 1e-9
        near = np.full((S, 4), -1, np.int16)
        tail = np.full((S, 4), -1, np.int16)
        for slot, plan in enumerate(plans):
            near[slot, :len(plan.near_offsets)] = plan.near_offsets
            tail[slot, :len(plan.tail_offsets)] = plan.tail_offsets
        positions = np.tile(np.arange(R, dtype=np.int16), (S, 1))
        arrays = dict(
            sources=np.asarray(sources, np.int32), positions=positions,
            token_ids=np.arange(S * R, dtype=np.int64).reshape(S, R) + 1000,
            near_offsets=near, tail_offsets=tail,
            near_support=np.asarray([p.near_support for p in plans], np.int16),
            tail_support=np.asarray([p.tail_support for p in plans], np.int16),
            utility=utility, resident=np.zeros(utility.shape, np.int8), features=features,
            s0=rng.random((S, R, len(ATTENTION)), dtype=np.float32),
            hidden_bf16=rng.integers(0x3c00, 0x3f80, (S, R, len(ATTENTION), D), dtype=np.uint16))
        meta = dict(episode=e, attention_layers=ATTENTION, gdn_layers=GDN, store_fields=FIELDS,
                    rows_per_source=R, coverage=dict(missing={}))
        np.savez(episode_file(tmp_path, e['episode_uid']), meta=np.asarray(json.dumps(meta)), **arrays)
        reference[e['episode_uid']] = (plans, utility, features, arrays)
    (tmp_path / 'selection.json').write_text(json.dumps(dict(episodes=episodes)))
    return reference


@pytest.fixture
def dataset(tmp_path):
    episodes = _episodes()
    reference = _write(tmp_path / 'labels', episodes)
    build(tmp_path / 'labels', tmp_path / 'ds', subset='pilot')
    return RankerData(tmp_path / 'ds'), episodes, reference


def test_aggregates_match_reference_and_rows_align(dataset):
    data, episodes, reference = dataset
    rows = data.rows
    cursor = 0
    for index, e in enumerate(episodes):
        plans, utility, features, arrays = reference[e['episode_uid']]
        for slot, plan in enumerate(plans):
            for r in range(R):
                assert rows['episode'][cursor] == index and rows['source'][cursor] == plan.source_group
                assert rows['token_id'][cursor] == arrays['token_ids'][slot, r]
                for k, a in enumerate(ATTENTION):
                    want = aggregate_horizon_labels(plan, {o: float(utility[slot, k, r, q])
                                                           for q, o in enumerate(plan.offsets)})
                    for name in ('near', 'tail', 'remaining'):
                        value, status = getattr(want, f'{name}_mean'), getattr(want, f'{name}_status')
                        assert rows[f'{name}_status'][cursor, k] == STATUS[status]
                        if value is None:
                            assert np.isnan(rows[name][cursor, k])
                        else:
                            assert rows[name][cursor, k] == pytest.approx(value, rel=1e-12)
                    window = [GDN.index(g) for g in range(a - 3, a)]
                    np.testing.assert_array_equal(data.raw(a)[cursor], features[slot, r, window])
                    np.testing.assert_array_equal(data.hidden(a)[cursor], arrays['hidden_bf16'][slot, r, k])
                cursor += 1
    assert cursor == rows['episode'].size


def test_leakage_and_incomplete_files_are_refused(tmp_path):
    episodes = _episodes()
    leaked = [dict(e) for e in episodes]
    leaked[1]['scene_id'] = leaked[0]['scene_id']
    with pytest.raises(ValueError, match='appears in'):
        check_partitions(leaked)
    _write(tmp_path / 'labels', episodes[:3])
    (tmp_path / 'labels' / 'selection.json').write_text(json.dumps(dict(episodes=episodes[:4])))
    with pytest.raises(FileNotFoundError):
        build(tmp_path / 'labels', tmp_path / 'ds')


def test_weights_balance_domains_and_episodes(dataset):
    data, _, _ = dataset
    index = np.flatnonzero(data.mask('ranker_fit', 3))
    w = data.weights(index)
    episode, domain = data.rows['episode'][index], data.rows['domain'][index]
    assert w.sum() == pytest.approx(1.0)
    for d in (0, 1):
        assert w[domain == d].sum() == pytest.approx(0.5)
        per_episode = [w[episode == x].sum() for x in np.unique(episode[domain == d])]
        np.testing.assert_allclose(per_episode, per_episode[0])


def test_fit_constants_never_see_dev_rows(dataset, monkeypatch):
    data, _, _ = dataset
    a = train_ranker.LayerData('L3', data, 7, 'cpu')
    dev = data.mask('ranker_dev', 7)
    data.rows['near'][dev, data.layer_column(7)] *= 1000.0
    b = train_ranker.LayerData('L3', data, 7, 'cpu')
    assert a.kappa == b.kappa
    assert torch.equal(a.mean, b.mean) and torch.equal(a.std, b.std)


def test_feature_widths_and_projection_are_fixed(dataset):
    data, _, _ = dataset
    index = np.flatnonzero(data.mask('ranker_fit', 11))[:50]
    widths = {arm: train_ranker.learned_features(data, arm, 11, index).shape[1]
              for arm in ('L0', 'L1', 'L2', 'L3', 'L0_raw', 'L0_S0res', 'H_linear', 'H_mlp_matched')}
    assert widths == dict(L0=96, L1=96, L2=288, L3=576, L0_raw=96, L0_S0res=96, H_linear=D, H_mlp_matched=576)
    assert torch.equal(train_ranker.projection(D), train_ranker.projection(D))
    hidden = train_ranker.learned_features(data, 'H_linear', 11, index)
    bits = np.asarray(data.hidden(11)[index])
    want = torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16).float()    # torch's own BF16 decode
    assert hidden.dtype == torch.float32 and torch.equal(hidden, want)


def test_zero_initialized_residual_reproduces_s0_order(dataset, monkeypatch):
    data, _, _ = dataset
    monkeypatch.setattr(train_ranker, 'EPOCHS', 0)
    d = train_ranker.LayerData('L0_S0res', data, 15, 'cpu')
    result = train_ranker.train_layer('L0_S0res', data, 15, 'cpu', 1e-3, d)
    _, scores = train_ranker.evaluate('L0_S0res', data, 15, result, 'ranker_holdout')
    _, s0 = train_ranker.evaluate('S0', data, 15, None, 'ranker_holdout')
    np.testing.assert_array_equal(np.argsort(-scores, kind='stable'), np.argsort(-s0, kind='stable'))


def test_learned_arm_fits_signal_and_artifact_round_trips(dataset, monkeypatch):
    data, _, _ = dataset
    monkeypatch.setattr(train_ranker, 'EPOCHS', 6)
    monkeypatch.setattr(train_ranker, 'BATCH', 64)         # the synthetic fit set is a single 4096 batch
    layer = 19
    d = train_ranker.LayerData('L0', data, layer, 'cpu')
    result = train_ranker.train_layer('L0', data, layer, 'cpu', 1e-3, d)
    learned, scores = train_ranker.evaluate('L0', data, layer, result)
    random, _ = train_ranker.evaluate('random', data, layer, None)
    oracle = train_ranker.pool_metrics(data, np.flatnonzero(data.mask('ranker_holdout', layer)),
                                       data.rows['near'][np.flatnonzero(data.mask('ranker_holdout', layer)),
                                                         data.layer_column(layer)],
                                       data.rows['near'][np.flatnonzero(data.mask('ranker_holdout', layer)),
                                                         data.layer_column(layer)])
    assert oracle['capture'] == pytest.approx(1.0) and oracle['spearman'] == pytest.approx(1.0)
    assert learned['capture'] > random['capture'] + 0.3 and learned['spearman'] > 0.6
    from qwen_vl.research.gdn_ranker import build_artifact
    heads = {a: result['head'] for a in ATTENTION}
    norms = {a: (result['mean'].numpy(), result['std'].numpy()) for a in ATTENTION}
    artifact = build_artifact('L0', heads, norms, {a: result['kappa'] for a in ATTENTION}, {})
    validate_artifact(artifact, 'L0', ATTENTION)
    ranker = FrozenRanker.load(json.dumps(artifact, sort_keys=True), canonical_sha256(artifact))
    index = np.flatnonzero(data.mask('ranker_holdout', layer))
    x = train_ranker.learned_features(data, 'L0', layer, index)
    keep = (x.shape[0] // 300) * 300
    frozen = ranker.score(layer, x[:keep]).double().numpy()
    np.testing.assert_allclose(frozen, scores[:keep], rtol=1e-5, atol=1e-5)


def test_spearman_matches_the_definition_with_ties():
    a = np.array([3.0, 1.0, 2.0, 2.0, 5.0, 1.0])
    b = np.array([1.0, 0.0, 4.0, 4.0, 9.0, 2.0])
    np.testing.assert_array_equal(train_ranker.average_ranks(a), [4, 0.5, 2.5, 2.5, 5, 0.5])
    ra, rb = train_ranker.average_ranks(a), train_ranker.average_ranks(b)
    assert train_ranker.spearman(a, b) == pytest.approx(np.corrcoef(ra, rb)[0, 1], rel=1e-12)
    assert train_ranker.spearman(a, a) == pytest.approx(1.0)
