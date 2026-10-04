"""Metamorphic enabled/disabled tests, with an optional pinned real processor."""
from copy import deepcopy
import os
import pytest
import torch
from PIL import Image
from transformers import AutoProcessor
from qwen_vl.data.episode_serializer import EpisodeSerializer
from qwen_vl.contracts import deep_merge
from test_step_lane_contract import lane_config
import yaml


def episode(tmp_path, actions=('TURN_RIGHT', 'STOP')):
    path = tmp_path / 'rgb.png'
    Image.new('RGB', (640, 480), (50, 100, 150)).save(path)
    return dict(episode_uid='e', dataset='r2r', dataset_version='test', split='train',
                scene_id='s', episode_id='1', instruction_id='1', instruction='Turn and stop.',
                observation_action_alignment='observation_before_action',
                steps=[dict(step_id=i, rgb_path=str(path), action_name=a, is_valid=True)
                       for i, a in enumerate(actions)])


def fixture_serializer(cfg):
    # The dependency double replaces tokenization/image processing only. The complete
    # episode assembly, feedback, spans, supervision, and metadata code remains real.
    s = EpisodeSerializer.__new__(EpisodeSerializer)
    s.config, s.mode = cfg, cfg['model']['output_mode']
    s.eos, s.separator = 9, [8]
    s.candidate_token_ids = [32, 33, 34, 35]
    s.action_ids = [[100, 101, 9], [102, 9], [103, 104, 105, 9], [106, 9]]
    s.template_hash = 'test'
    s.encode_prefix = lambda instruction: s.text_block([8, 7, 8])
    s.encode_observation = lambda image, t=0: dict(
        s.text_block([8, 5, 8, 6]), pixel_values=torch.ones(1, 4),
        image_grid_thw=torch.tensor([[1, 30, 40]]))
    return s


@pytest.mark.parametrize('mode,feedback', [('qwen_text', 'canonical_action_text'),
    ('candidate_logits', 'candidate_token'), ('candidate_logits', 'canonical_action_text'),
    ('candidate_logits', 'none')])
def test_enabled_adds_only_roles_and_preserves_exact_serialization(tmp_path, mode, feedback):
    cfg = lane_config(feedback)
    cfg['model']['output_mode'] = mode
    enabled = fixture_serializer(cfg)
    disabled_cfg = deepcopy(cfg)
    disabled_cfg['model'].pop('step_lane')
    disabled = fixture_serializer(disabled_cfg)
    ep = episode(tmp_path)
    old, new = disabled.encode_episode(ep), enabled.encode_episode(ep)
    assert set(new) == set(old) | {'step_lane_roles'}
    for key, value in old.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, new[key]), key
        else:
            assert value == new[key], key
    roles = new['step_lane_roles']
    assert roles.dtype == torch.uint8
    assert roles[0, :3].tolist() == [0, 0, 0]
    for i, (start, end) in enumerate(new['step_plan'][1]):
        if mode == 'qwen_text':
            first_target = new['response_target_positions'][new['response_action_index'].eq(i)][0]
            decision = int(first_target) - 1
        else:
            decision = int(new['read_positions'][i])
        assert end - 1 > decision
        assert roles[0, start:decision + 1].tolist() == [1] * (decision - start + 1)
        assert roles[0, end - 1] == 3
        assert roles[0, decision + 1:end - 1].eq(2).all()
    assert int(roles.eq(3).sum()) == 2
    assert 'step_lane' in enabled.metadata()
    assert 'step_lane' not in disabled.metadata()


def test_no_history_labels_change_only_targets(tmp_path):
    s = fixture_serializer(lane_config('none'))
    ep = episode(tmp_path)
    changed = deepcopy(ep)
    changed['steps'][0]['action_name'] = 'TURN_LEFT'
    old, new = s.encode_episode(ep), s.encode_episode(changed)
    assert not torch.equal(old['action_class_ids'], new['action_class_ids'])
    for key in ('input_ids', 'mm_token_type_ids', 'pixel_values', 'image_grid_thw',
                'read_positions', 'step_lane_roles'):
        assert torch.equal(old[key], new[key]), key


def test_history_label_changes_only_feedback_suffix(tmp_path):
    s = fixture_serializer(lane_config('candidate_token'))
    ep = episode(tmp_path)
    changed = deepcopy(ep)
    changed['steps'][0]['action_name'] = 'TURN_LEFT'
    old, new = s.encode_episode(ep), s.encode_episode(changed)
    diffs = old['input_ids'].ne(new['input_ids']).nonzero().tolist()
    assert diffs == [[0, int(old['read_positions'][0]) + 1]]
    assert torch.equal(old['step_lane_roles'], new['step_lane_roles'])


def test_one_step_terminal_still_writes(tmp_path):
    data = fixture_serializer(lane_config('none')).encode_episode(episode(tmp_path, ('STOP',)))
    assert data['step_lane_roles'][0, -1] == 3
    assert int(data['step_lane_roles'].eq(3).sum()) == 1


@pytest.mark.parametrize('mode,feedback', [('qwen_text', None), ('candidate_logits', 'none'),
    ('candidate_logits', 'candidate_token'), ('candidate_logits', 'canonical_action_text')])
def test_real_tokenizer_roles_preserve_bytes(tmp_path, mode, feedback):
    path = os.environ.get('VLN_MODEL_PATH')
    if not path:
        pytest.skip('Pinned processor required; set VLN_MODEL_PATH')
    if mode == 'qwen_text':
        with open('configs/vln_r2r_v0_base.yaml') as f:
            cfg = yaml.safe_load(f)
        with open('configs/vln_r2r_v0_qwen_text.yaml') as f:
            cfg = deep_merge(cfg, yaml.safe_load(f))
        cfg['model']['step_lane'] = {'enabled': True}
    else:
        cfg = lane_config(feedback)
    proc = AutoProcessor.from_pretrained(path)
    s = EpisodeSerializer(proc, cfg)
    old_cfg = deepcopy(cfg)
    old_cfg['model'].pop('step_lane')
    old = EpisodeSerializer(proc, old_cfg)
    ep = episode(tmp_path)
    new_data, old_data = s.encode_episode(ep), old.encode_episode(ep)
    assert set(new_data) - set(old_data) == {'step_lane_roles'}
    for key, value in old_data.items():
        assert torch.equal(value, new_data[key]) if isinstance(value, torch.Tensor) else value == new_data[key]
    assert int(new_data['step_lane_roles'].eq(3).sum()) == 2
