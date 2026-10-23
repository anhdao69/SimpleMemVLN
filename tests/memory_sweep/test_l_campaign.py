"""Queue/recipe invariants of the L campaign controller (no GPU, no Slurm)."""
import json
from pathlib import Path
import pytest
import yaml
from qwen_vl.contracts import validate_config
from qwen_vl.research.episodic_contract import parse_episodic_spec
from qwen_vl.research.gdn_ranker import canonical_sha256
from qwen_vl.research.gdn_sweep_contract import arm_memory_config
from qwen_vl.train.vln_runtime import resolve_config
from scripts.memory_sweep import l_campaign
from scripts.memory_sweep.init_campaign import RUNTIME
from scripts.memory_sweep.init_l_campaign import SCREENING, lanes
from scripts.memory_sweep.s_campaign import expected_config
from test_learned_arms import random_artifact

ROOT = Path(__file__).resolve().parents[2]


def _resolve(memory, runtime):
    return resolve_config(ROOT / 'configs/episodic/science_common.yaml', ROOT / 'configs/vln_empty.yaml',
                          memory, ROOT / 'configs/vggt/g3_joint_one_epoch.yaml', runtime)


@pytest.mark.parametrize('arm', l_campaign.ARMS)
def test_learned_recipe_equals_m5_e1_except_memory_and_loader(tmp_path, arm):
    reference = _resolve(ROOT / 'configs/episodic/arm_m5.yaml', ROOT / 'reports/vggt_five/M5_E1/operations/runtime.yaml')
    runtime = tmp_path / 'runtime.yaml'
    runtime.write_text(yaml.safe_dump(RUNTIME))
    artifact = random_artifact(arm, tuple(range(3, 32, 4)), 32)
    memory = arm_memory_config(arm, artifacts=dict(ranker=artifact, ranker_sha256=canonical_sha256(artifact)))
    path = tmp_path / f'arm_{arm}.yaml'
    path.write_text(yaml.safe_dump(memory, sort_keys=False))
    cfg = _resolve(path, runtime)
    assert cfg == expected_config(reference, yaml.safe_load(path.read_text())['memory'])
    assert {s for s in cfg if cfg[s] != reference[s]} == {'memory', 'training'}
    assert {k for k in cfg['training'] if cfg['training'][k] != reference['training'][k]} == {
        'dataloader_num_workers', 'dataloader_prefetch_factor'}
    validate_config(json.loads(json.dumps(cfg)), 4)
    spec = parse_episodic_spec(cfg)
    assert spec.is_learned and spec.ranker_sha256 == canonical_sha256(artifact)
    assert cfg['memory']['kv_guard_tokens'] == 4784
    assert yaml.safe_load(path.read_text()) == memory          # the YAML round trip keeps exact FP32 bytes


def test_lanes_train_every_learned_arm_once_after_the_s_lane_and_labels():
    plan = lanes('4925.172')
    arms = plan['A']['arms'] + plan['B']['arms']
    assert sorted(arms) == sorted(l_campaign.ARMS) == sorted(a for a in SCREENING if a != 'S0')
    assert plan['B']['qualifies'] and not plan['A']['qualifies']
    assert plan['B']['predecessor'] == dict(step='4925.172')
    assert plan['A']['predecessor']['final_status'] == 'lane-complete'
    assert plan['A']['predecessor']['step'] == '5248.27'


def test_fit_shards_cover_every_arm_and_control_once():
    arms = [a for shard in l_campaign.FIT_SHARDS for a in shard]
    assert sorted(arms) == sorted(l_campaign.ARMS + l_campaign.CONTROLS) and len(arms) == len(set(arms))


def test_fit_ledger_merge_refuses_inconsistent_shards():
    common = dict(target='t', seed=429, lrs=[3e-4, 1e-3], weight_decay=1e-2, batch=4096, epochs=20, clip=1.0,
                  patience=3, budget_rows=96, dataset=dict(rows=10), source_commit='abc', h_projection=dict(sha256='p'))
    a = dict(common, arms={'L0': {}}, fixed={'S0': {}})
    b = dict(common, arms={'L1': {}}, fixed={})
    merged = l_campaign.merge_fit_reports([a, b])
    assert set(merged['arms']) == {'L0', 'L1'} and merged['fixed'] == {'S0': {}}
    with pytest.raises(ValueError, match='twice'):
        l_campaign.merge_fit_reports([a, dict(b, arms={'L0': {}})])
    with pytest.raises(ValueError, match='disagree'):
        l_campaign.merge_fit_reports([a, dict(b, seed=430)])
    with pytest.raises(ValueError, match='Fixed scores'):
        l_campaign.merge_fit_reports([a, dict(b, fixed={'S0': {}})])


def test_predecessor_waits_for_final_status_and_step_end(tmp_path, monkeypatch):
    state = tmp_path / 'state.json'
    lane = object.__new__(l_campaign.Lane)
    lane.spec = dict(predecessor=dict(state=str(state), step='5248.27', final_status='lane-complete'))
    lane.own, lane.job, lane.dir = '5248.40', '5248', tmp_path
    observed = []
    sequence = iter([('training', ['5248.0', '5248.27', '5248.40']),
                     ('lane-complete', ['5248.0', '5248.27', '5248.40']),
                     ('lane-complete', ['5248.0', '5248.40'])])
    current = {}

    def steps():
        return current['steps']

    def record(status, **fields):
        observed.append(status)
        status_text, current['steps'] = next(sequence)
        state.write_text(json.dumps(dict(status=status_text)))
    status_text, current['steps'] = next(sequence)
    state.write_text(json.dumps(dict(status=status_text)))
    monkeypatch.setattr(l_campaign.time, 'sleep', lambda s: None)
    lane.active_steps, lane.record, lane.gpu_busy = steps, record, lambda: False
    lane.control = lambda name: False
    lane.wait_predecessor()
    assert observed == ['waiting-predecessor', 'waiting-predecessor']


def test_drift_bank_overlap_is_the_static_top_b_agreement():
    import numpy as np
    from scripts.memory_sweep.ranker_drift import bank_overlaps
    rng = np.random.default_rng(0)
    a = rng.random(10 * 300)
    assert bank_overlaps(a, a, 10) == [1.0, 1.0]          # eligible rows exceed B = 1200 only at decisions 8, 9
    assert bank_overlaps(a, -a, 10) == [0.75, 0.5]      # forced overlap (2B - eligible) / B of reversed rankings
    shuffled = bank_overlaps(a, rng.permutation(a), 10)
    assert 0.2 < np.mean(shuffled) < 0.9
