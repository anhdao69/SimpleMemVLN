import numpy as np
import pytest
import torch
from PIL import Image

from qwen_vl.contracts import ACTIONS
from qwen_vl.rl.episode import encode_rollout


def frames(n, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(n)]


def test_canonical_rollout_matches_sft_serializer(serializer, tmp_path):
    images = frames(4)
    actions = [1, 0, 2, 3]
    steps = []
    for t, (rgb, a) in enumerate(zip(images, actions)):
        path = tmp_path / f'{t}.png'
        Image.fromarray(rgb).save(path)
        steps.append(dict(step_id=t, is_valid=True, action_name=ACTIONS[a], rgb_path=str(path)))
    episode = dict(episode_uid='u', dataset='r2r', dataset_version='v', split='train',
                   scene_id='s', episode_id='1', instruction_id='0', instruction='Walk ahead.',
                   observation_action_alignment='observation_before_action', steps=steps)
    reference = serializer.encode_episode(episode)
    rollout = encode_rollout(serializer, 'Walk ahead.', images,
                             [dict(tokens=serializer.action_ids[a], complete=True) for a in actions])
    for key in ('input_ids', 'mm_token_type_ids', 'pixel_values', 'image_grid_thw',
                'response_target_positions', 'response_action_index'):
        assert torch.equal(reference[key], rollout[key]), key
    assert reference['step_plan'] == rollout['step_plan']
    assert reference['num_actions'] == rollout['num_actions']


def test_unterminated_final_response_is_kept_without_separator(serializer):
    images = frames(2, 1)
    bad = serializer.ids('MOVE')
    batch = encode_rollout(serializer, 'Go.', images,
                           [dict(tokens=serializer.action_ids[0], complete=True),
                            dict(tokens=bad, complete=False)])
    prefix, spans = batch['step_plan']
    assert spans[-1][1] == batch['input_ids'].shape[1]
    tail = batch['input_ids'][0, -len(bad):].tolist()
    assert tail == bad
    assert batch['response_action_index'].tolist().count(1) == len(bad)


def test_only_final_response_may_be_unterminated(serializer):
    images = frames(2, 2)
    with pytest.raises(ValueError):
        encode_rollout(serializer, 'Go.', images,
                       [dict(tokens=serializer.ids('MOVE'), complete=False),
                        dict(tokens=serializer.action_ids[3], complete=True)])
    with pytest.raises(ValueError):
        encode_rollout(serializer, 'Go.', images[:1],
                       [dict(tokens=serializer.ids('STOP'), complete=True)])
