from copy import deepcopy
from types import SimpleNamespace
import pytest
from qwen_vl.research.lane_contract import parse_step_lane_spec
from qwen_vl.contracts import validate_config
from test_candidate_contract import config


def lane_config(feedback='candidate_token'):
    cfg = config(feedback)
    cfg['model']['step_lane'] = {'enabled': True}
    return cfg


def text_config():
    return SimpleNamespace(hidden_size=2560, num_hidden_layers=32,
                           layer_types=['full_attention' if i % 4 == 3 else 'linear_attention' for i in range(32)])


def test_disabled_contract_is_unchanged():
    cfg = config()
    before = deepcopy(cfg)
    assert parse_step_lane_spec(cfg, text_config()) is None
    assert validate_config(cfg) == before
    cfg['model']['step_lane'] = {'enabled': False, 'layers': ['ignored']}
    assert parse_step_lane_spec(cfg, text_config()) is None


@pytest.mark.parametrize('feedback', ['candidate_token', 'canonical_action_text', 'none'])
def test_candidate_feedback_preserved(feedback):
    cfg = lane_config(feedback)
    before = deepcopy(cfg)
    spec = parse_step_lane_spec(cfg, text_config())
    assert spec.layers == (16, 20, 24, 28)
    assert spec.half_lives_steps == (8., 24., 64., 192.)
    assert cfg == before
    assert validate_config(cfg) is cfg


@pytest.mark.parametrize('field,value', [
    ('layers', [16, 16]), ('layers', [20, 16]), ('layers', [15]), ('layers', [32]),
    ('clock_mode', 'token_rate'), ('prefix_mode', 'write'), ('version', 'v0'),
    ('half_lives_steps', [8, 24, float('nan'), 192]),
    ('half_lives_steps', [8, 24, -1, 192]), ('beta_init', 1),
    ('key_dim', 64), ('key_dim', 128.0), ('num_value_heads', 4.0), ('num_key_heads', 3), ('eps', float('inf')),
])
def test_invalid_spec_rejected(field, value):
    cfg = lane_config()
    cfg['model']['step_lane'][field] = value
    with pytest.raises(ValueError):
        parse_step_lane_spec(cfg, text_config())


def test_classification_and_missing_calibration_rejected():
    cfg = lane_config()
    cfg['model']['output_mode'] = 'classification'
    with pytest.raises(ValueError, match='classification'):
        parse_step_lane_spec(cfg, text_config())
    cfg = lane_config()
    cfg['model']['step_lane']['clock_mode'] = 'token_rate_calibrated'
    with pytest.raises(ValueError, match='token_rate_reference'):
        parse_step_lane_spec(cfg, text_config())
    cfg['model']['step_lane']['token_rate_reference'] = 320
    assert parse_step_lane_spec(cfg, text_config()).token_rate_reference == 320


def test_validate_config_rejects_lane_even_without_text_config():
    cfg = lane_config()
    cfg['model']['step_lane']['clock_mode'] = 'bad'
    with pytest.raises(ValueError):
        validate_config(cfg)
