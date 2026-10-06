import json
from types import SimpleNamespace
import numpy as np
import pytest
from qwen_vl.eval.habitat_r2r import select_episodes, load_journal, rollout_episode
from qwen_vl.stream.transport import encode_rgb, decode_rgb


def test_rxr_guide_and_ground_truth_paths_resolve_from_split_and_role():
    from qwen_vl.eval.habitat_r2r import resolve_dataset_path
    template = '/datasets/rxr_15deg/{split}/{split}_{role}.json.gz'
    assert resolve_dataset_path(template, 'val_unseen', 'guide') == (
        '/datasets/rxr_15deg/val_unseen/val_unseen_guide.json.gz')
    ground_truth = '/datasets/rxr_15deg/{split}/{split}_{role}_gt.json.gz'
    assert resolve_dataset_path(ground_truth, 'val_unseen', 'guide') == (
        '/datasets/rxr_15deg/val_unseen/val_unseen_guide_gt.json.gz')


def test_raw_transport_preserves_all_pixels_and_rejects_bad_frame():
    rgb = np.random.default_rng(42).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    assert np.array_equal(np.asarray(decode_rgb(encode_rgb(rgb))), rgb)
    with pytest.raises(ValueError):
        encode_rgb(rgb.astype(np.float32))
    with pytest.raises(ValueError):
        decode_rgb({'rgb_raw': 'AAAA'})


def test_selection_and_resume_reject_duplicate_or_foreign_ids(tmp_path):
    episodes = [SimpleNamespace(episode_id=str(i), scene_id=s) for i,s in [(2,'b'),(1,'a'),(3,'b')]]
    assert [e.episode_id for e in select_episodes(episodes, None, 2)] == ['1','2']
    with pytest.raises(ValueError):
        select_episodes(episodes, ['1','1'], 0)
    with pytest.raises(ValueError):
        select_episodes(episodes, ['4'], 0)
    path = tmp_path/'episodes.jsonl'
    path.write_text('{"episode_id":"1"}\n{"episode_id":"1"}\n')
    with pytest.raises(ValueError):
        load_journal(path, ['1','2'])
    path.write_text('{"episode_id":"4"}\n')
    with pytest.raises(ValueError):
        load_journal(path, ['1','2'])


class Env:
    def __init__(self):
        self.steps=[]
        self.sim=SimpleNamespace(get_agent_state=lambda:SimpleNamespace(position=np.array([len(self.steps)*.25,0,0])))
        self.episode_over=False
    def reset(self):
        return {'rgb':np.zeros((480,640,3),dtype=np.uint8)}
    def get_metrics(self):
        return {'success':float(self.episode_over),'spl':.5,'distance_to_goal':2.0}
    def step(self, action):
        self.steps.append(action)
        self.episode_over=action==0
        return self.reset()


def test_capture_before_action_forced_stop_and_raw_response():
    env=Env()
    seen=[]
    class Model:
        def reset(self,*args): pass
        def observe(self,uid,step,rgb):
            seen.append(len(env.steps))
            return {'class_id':0,'action_name':'MOVE_FORWARD','habitat_action_id':1,'response_token_ids':[1,2], 'response_text':'MOVE_FORWARD','model_seconds':.1,'retained_kv_tokens':300*(step+1)}
    ep=SimpleNamespace(episode_id='1',scene_id='s',instruction=SimpleNamespace(instruction_text='go'))
    row=rollout_episode(env,Model(),ep,'val_unseen',max_steps=3)
    assert seen == [0,1,2]
    assert env.steps == [1,1,0]
    assert row['forced_stop'] and not row['failed']
    assert row['actions'][-1]['predicted']==1
    assert row['actions'][-1]['executed']==0
    assert row['actions'][0]['response_token_ids']==[1,2]
    assert row['metrics']['oracle_success']==1.0


def test_failed_reset_has_identity_and_no_stale_metrics():
    env=Env()
    def fail(): raise ValueError('reset failed')
    env.reset=fail
    ep=SimpleNamespace(episode_id='7',scene_id='s',instruction=SimpleNamespace(instruction_text='go'))
    row=rollout_episode(env,None,ep,'val_unseen')
    assert row['episode_id']=='7' and row['failed']
    assert row['metrics']=={}


def test_model_process_failure_is_fatal_but_invalid_action_is_not():
    from qwen_vl.eval.habitat_r2r import ModelProcessError
    ep=SimpleNamespace(episode_id='1',scene_id='s',instruction=SimpleNamespace(instruction_text='go'))
    class Model:
        def reset(self,*args): pass
        def observe(self,*args): raise ModelProcessError('dead child')
    row=rollout_episode(Env(),Model(),ep,'val_unseen')
    assert row['failed'] and row['fatal'] and not row['actions']
    class Invalid(Model):
        def observe(self,*args): raise ValueError('Invalid action text')
    row=rollout_episode(Env(),Invalid(),ep,'val_unseen')
    assert row['failed'] and not row['fatal'] and not row['actions']
    from qwen_vl.eval.metrics import summarize
    assert summarize([row])['sr']==0


def test_interrupted_tail_is_backed_up_and_retried_but_middle_corruption_rejected(tmp_path):
    path=tmp_path/'episodes.jsonl'
    complete='{"episode_id":"1"}\n'
    path.write_text(complete+'{"episode_id":"2", "actions": [')
    assert load_journal(path, {'1','2'})==[{'episode_id':'1'}]
    assert path.read_text()==complete
    assert path.with_suffix('.jsonl.incomplete-tail').read_text().startswith('{"episode_id":"2"')
    path.write_text('{broken}\n'+complete)
    with pytest.raises(ValueError):
        load_journal(path, {'1','2'})


def test_base_asset_identity_changes_when_preprocessing_changes(tmp_path):
    from qwen_vl.eval.habitat_r2r import base_asset_hashes
    (tmp_path/'config.json').write_text('{}')
    processor=tmp_path/'preprocessor_config.json'
    processor.write_text('{"image_mean":[0.5,0.5,0.5]}')
    first=base_asset_hashes(tmp_path)
    processor.write_text('{"image_mean":[0.1,0.2,0.3]}')
    assert base_asset_hashes(tmp_path)!=first
    assert first['config.json']


def test_instruction_whitespace_matches_training_preparation():
    ep=SimpleNamespace(episode_id='1',scene_id='s',instruction=SimpleNamespace(instruction_text='  go\n'))
    prompts=[]
    class Model:
        def reset(self, uid, instruction): prompts.append(instruction)
        def observe(self,*args): return {'class_id':3,'action_name':'STOP','habitat_action_id':0}
    row=rollout_episode(Env(),Model(),ep,'val_unseen')
    assert prompts==['go']
    assert row['instruction']=='go'
