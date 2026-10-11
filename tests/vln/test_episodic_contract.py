"""Versioned episodic memory contract and strict configuration validation."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from qwen_vl.contracts import validate_config
from qwen_vl.research.episodic_contract import (
    ATTENTION_BACKEND, EPISODIC_MODE, QWEN35_4B_PAIRS, derived_kv_guard,
    parse_episodic_spec, scoring_pairs_from_layer_types, spec_from_dict)


def _config(recent=8, budget=0, score='none'):
    cfg = dict(
        model=dict(backbone='Qwen/Qwen3.5-4B', output_mode='qwen_text', revision='x'),
        observations=dict(serializer_version='vln_append_only_chat_v1', append_action_tokens=True,
                          max_prefix_tokens=2048, max_step_group_tokens=384,
                          expected_rgb_shape=[480, 640, 3], expected_visual_tokens=300),
        memory=dict(mode=EPISODIC_MODE, recent_groups_including_current=recent,
                    episodic_budget_tokens=budget, episodic_granularity='token',
                    episodic_score=score, episodic_selector_seed=429,
                    episodic_per_source_cap=None, evict_before_new_step=True,
                    kv_guard_tokens=2048 + recent * 384 + budget),
        training=dict(microbatch_episodes_per_rank=1, use_cache=False,
                      supervise_assistant_terminator=True, nominal_episodes_per_update=8,
                      gradient_accumulation_steps=2),
        runtime=dict(text_attention='simplememvln_step_attention', vision_attention='flash_attention_2',
                     window_attention_batch_steps=8),
        generation=dict(max_response_tokens_including_eos=16, fixed_separator='\n'),
    )
    return cfg


@pytest.mark.parametrize('recent,budget,score', [
    (8, 0, 'none'), (9, 0, 'none'), (8, 300, 'committed_mean_l2_v1'),
    (8, 300, 'random_hash_v1'), (4, 1200, 'committed_mean_l2_v1'), (4, 1200, 'random_hash_v1')])
def test_phase_arms_validate(recent, budget, score):
    cfg = validate_config(_config(recent, budget, score), 4)
    spec = parse_episodic_spec(cfg)
    assert (spec.recent_groups, spec.budget_tokens, spec.score) == (recent, budget, score)
    assert spec.scoring_pairs == QWEN35_4B_PAIRS == tuple((a - 1, a) for a in range(3, 32, 4))
    assert spec.attention_backend == ATTENTION_BACKEND
    assert spec.needs_write_scores == (score == 'committed_mean_l2_v1')
    assert spec_from_dict(spec.to_dict()) == spec
    assert derived_kv_guard(cfg, spec) == cfg['memory']['kv_guard_tokens']


@pytest.mark.parametrize('mutate,message', [
    (lambda c: c['memory'].update(episodic_score='committed_mean_l2_v1'), 'B0 requires'),
    (lambda c: c['memory'].update(episodic_budget_tokens=300, kv_guard_tokens=5420), 'B0 requires'),
    (lambda c: c['memory'].update(episodic_per_source_cap=100), 'Per-source'),
    (lambda c: c['memory'].update(episodic_granularity='visual_group'), 'visual_group'),
    (lambda c: c['memory'].update(kv_guard_tokens=5000), 'derived'),
    (lambda c: c['memory'].update(recent_groups_including_current=0), 'recent'),
    (lambda c: c['memory'].update(evict_before_new_step=False), 'promotes'),
    (lambda c: c['memory'].update(episodic_score='beta_proxy'), 'Unknown episodic score'),
    (lambda c: c['memory'].update(episodic_attention_backend='old'), 'differs'),
    (lambda c: c['model'].update(step_lane=dict(enabled=True)), 'step lane'),
    (lambda c: c['runtime'].update(text_attention='flash_attention_2'), 'step attention'),
])
def test_invalid_specs_fail_closed(mutate, message):
    cfg = _config()
    mutate(cfg)
    with pytest.raises(ValueError, match=message):
        validate_config(cfg)


def test_legacy_modes_reject_stray_episodic_keys_and_keep_window8():
    cfg = _config()
    cfg['memory'] = dict(mode='window8', kv_window_steps_including_current=8,
                         evict_before_new_step=True, kv_guard_tokens=4096)
    assert parse_episodic_spec(validate_config(deepcopy(cfg))) is None
    cfg['memory']['episodic_budget_tokens'] = 300
    with pytest.raises(ValueError, match='only valid'):
        validate_config(cfg)


def test_scoring_pairs_follow_real_layer_types():
    types = (['linear_attention'] * 3 + ['full_attention']) * 8
    assert scoring_pairs_from_layer_types(types) == QWEN35_4B_PAIRS
    with pytest.raises(ValueError, match='preceding GDN'):
        scoring_pairs_from_layer_types(['full_attention', 'linear_attention'])
    spec = parse_episodic_spec(_config(), SimpleNamespace(layer_types=types))
    assert spec.scoring_layer_for(31) == 30


def test_staging_knobs_validated():
    cfg = _config()
    cfg['training'].update(dataloader_num_workers=0, dataloader_prefetch_factor=2)
    with pytest.raises(ValueError, match='prefetch'):
        validate_config(cfg)
    cfg['training'].update(dataloader_num_workers=1, activation_offload_backend='disk')
    with pytest.raises(ValueError, match='offload backend'):
        validate_config(cfg)
