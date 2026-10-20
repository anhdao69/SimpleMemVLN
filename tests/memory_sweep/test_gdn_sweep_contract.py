"""Versioned sweep contract: legacy unchanged, sweep arms explicit and fail closed."""
from copy import deepcopy
from pathlib import Path
import pytest
import yaml
from qwen_vl.contracts import validate_config
from qwen_vl.research.episodic_contract import EpisodicSpec, QWEN35_4B_PAIRS, parse_episodic_spec
from qwen_vl.research.gdn_sweep_contract import (
    ARM_REGISTRY, GdnSweepSpec, arm_memory_config, canonical_sha256, drift_floor_record,
    slow_head_mask_record)
from qwen_vl.train.vln_runtime import resolve_config

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / 'configs/episodic/science_common.yaml'
ONE_EPOCH = ROOT / 'configs/vggt/g3_joint_one_epoch.yaml'
H4 = ROOT / 'configs/episodic/profile_h4.yaml'


def _base():
    return resolve_config(COMMON, ROOT / 'configs/vln_empty.yaml', ROOT / 'configs/episodic/arm_m5.yaml',
                          ONE_EPOCH, H4)


def _with(memory):
    cfg = _base()
    cfg['memory'] = deepcopy(memory['memory'])
    return cfg


MASK = {str(layer): [0, 3, 5, 8, 13, 21, 22, 31] for layer, _ in QWEN35_4B_PAIRS}
FLOOR = {str(layer): 1e-3 * (layer + 1) for layer, _ in QWEN35_4B_PAIRS}


def _artifacts(score):
    if score == 'S6':
        return dict(slow_head_mask=MASK, slow_head_mask_sha256=canonical_sha256(MASK))
    if score in ('S8N', 'S8W'):
        return dict(drift_denominator_floor=FLOOR, drift_denominator_floor_sha256=canonical_sha256(FLOOR))
    return None


def test_legacy_episodic_arms_parse_exactly_as_before():
    for arm, score, recent, budget, guard in (('m5', 'committed_mean_l2_v1', 4, 1200, 4784),
                                               ('m5c', 'random_hash_v1', 4, 1200, 4784),
                                               ('v0', 'none', 8, 0, 5120)):
        cfg = resolve_config(COMMON, ROOT / f'configs/episodic/arm_{arm}.yaml', None, ONE_EPOCH, H4)
        spec = parse_episodic_spec(cfg)
        backend = 'fp32_torch_wy_block64_v1' if score == 'committed_mean_l2_v1' else 'none'
        segmentation = 'prefix_obs_response_block64_v1' if score == 'committed_mean_l2_v1' else 'none'
        assert spec == EpisodicSpec('episodic_window_v1', recent, budget, 'token', score, QWEN35_4B_PAIRS,
                                    429, None, backend, segmentation,
                                    'episodic_fa2_varlen_gather_fp32acc_v1')
        validate_config(cfg, 4)


@pytest.mark.parametrize('score', list(ARM_REGISTRY))
def test_every_registry_arm_parses_and_validates(score):
    memory = arm_memory_config(score, artifacts=_artifacts(score))
    cfg = _with(memory)
    validate_config(deepcopy(cfg), 4)
    spec = parse_episodic_spec(cfg)
    assert isinstance(spec, GdnSweepSpec) and spec.score == score
    assert (spec.granularity, spec.head_reduction, spec.score_backend) == ARM_REGISTRY[score][:3]
    assert (spec.recent_groups, spec.budget_tokens, spec.attention_layers) == (4, 1200, tuple(range(3, 32, 4)))
    assert cfg['memory']['kv_guard_tokens'] == 4784
    assert spec.needs_write_scores == (not score.startswith('C_'))
    if score == 'S6':
        assert spec.slow_heads(2) == (0, 3, 5, 8, 13, 21, 22, 31)
    if score in ('S8N', 'S8W'):
        assert spec.floor(30) == pytest.approx(0.031)
    # The serializer contract stores the complete selector contract.
    assert spec.to_dict()['score'] == score


def test_committed_arm_yaml_files_match_the_generator():
    for path in sorted((ROOT / 'configs/memory_sweep').glob('arm_*.yaml')):
        score = path.stem[len('arm_'):]
        assert yaml.safe_load(path.read_text()) == arm_memory_config(score), path


@pytest.mark.parametrize('mutate,match', [
    (lambda m: m.pop('head_reduction'), 'missing'),
    (lambda m: m.update(extra_key=1), 'unknown'),
    (lambda m: m.update(scorer_backend='qualified_m5_legacy_v1'), 'backend'),
    (lambda m: m.update(selection_unit='frame300'), 'unit'),
    (lambda m: m.update(cosine_epsilon='1e-6'), 'cosine_epsilon'),
    (lambda m: m.update(zero_value_threshold=1e-5), 'zero_value_threshold'),
    (lambda m: m.update(priority_timing='dynamic_boundary'), 'priority_timing'),
    (lambda m: m.update(kv_guard_tokens=4000), 'guard'),
    (lambda m: m.update(episodic_budget_tokens=0, kv_guard_tokens=3584), 'budget'),
    (lambda m: m.update(score_id='S10'), 'Unknown'),
    (lambda m: m.update(evict_before_new_step=False), 'evicts'),
    (lambda m: m.update(artifacts=dict(slow_head_mask=MASK)), 'does not consume'),
])
def test_sweep_parser_fails_closed(mutate, match):
    memory = arm_memory_config('S1')
    mutate(memory['memory'])
    with pytest.raises(ValueError, match=match):
        parse_episodic_spec(_with(memory))


def test_frame_budget_must_be_whole_frames():
    memory = arm_memory_config('S7', budget=1000)
    with pytest.raises(ValueError, match='whole number'):
        parse_episodic_spec(_with(memory))


def test_calibration_artifacts_are_required_and_hash_verified():
    with pytest.raises(ValueError, match='slow-head mask'):
        parse_episodic_spec(_with(arm_memory_config('S6')))
    with pytest.raises(ValueError, match='drift floors'):
        parse_episodic_spec(_with(arm_memory_config('S8W')))
    bad = dict(_artifacts('S6')); bad['slow_head_mask'] = dict(MASK, **{'2': [0, 1, 2, 3, 4, 5, 6, 7]})
    with pytest.raises(ValueError, match='SHA-256'):
        parse_episodic_spec(_with(arm_memory_config('S6', artifacts=bad)))
    short = {k: v[:7] for k, v in MASK.items()}
    with pytest.raises(ValueError, match='sorted unique'):
        parse_episodic_spec(_with(arm_memory_config(
            'S6', artifacts=dict(slow_head_mask=short, slow_head_mask_sha256=canonical_sha256(short)))))
    tiny = dict(FLOOR, **{'2': 1e-9})
    with pytest.raises(ValueError, match='>='):
        parse_episodic_spec(_with(arm_memory_config(
            'S8N', artifacts=dict(drift_denominator_floor=tiny,
                                  drift_denominator_floor_sha256=canonical_sha256(tiny)))))
    assert slow_head_mask_record({2: [5, 1]}) == {'2': [1, 5]}
    assert drift_floor_record({2: 1}) == {'2': 1.0}


def test_sweep_keys_are_rejected_in_legacy_and_research_modes():
    cfg = resolve_config(COMMON, ROOT / 'configs/vln_memory_window8.yaml', None, ONE_EPOCH)
    cfg['memory']['kv_guard_tokens'] = 4096
    cfg['memory']['recent_groups_including_current'] = 4
    with pytest.raises(ValueError):
        validate_config(cfg)
    research = _with(arm_memory_config('S1'))
    research['research'] = dict(run_id='G3_M5')
    with pytest.raises(ValueError, match='research'):
        parse_episodic_spec(research)


def test_serializer_metadata_carries_the_full_sweep_contract():
    from types import SimpleNamespace
    from qwen_vl.data.episode_serializer import EpisodeSerializer
    cfg = _with(arm_memory_config('S8W', artifacts=_artifacts('S8W')))
    serializer = EpisodeSerializer.__new__(EpisodeSerializer)
    serializer.config, serializer.mode = cfg, 'qwen_text'
    serializer.template_hash, serializer.action_ids, serializer.eos, serializer.separator = 'h', [[1]], 2, [3]
    metadata = EpisodeSerializer.metadata(serializer)
    assert metadata['episodic']['score'] == 'S8W'
    assert metadata['episodic']['drift_floor_sha256'] == canonical_sha256(FLOOR)
    assert metadata['episodic']['drift_floor'][0] == [2, FLOOR['2']]
    assert metadata['config']['memory']['artifacts']['drift_denominator_floor'] == FLOOR
