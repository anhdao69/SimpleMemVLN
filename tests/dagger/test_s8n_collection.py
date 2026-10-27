import json
import pytest
import sys
from types import SimpleNamespace
import numpy as np

from s8n_dagger.common import RecoveryControl, accepted_trajectory, bind_contract, partition, training_row
from s8n_dagger.oracle import ReferenceOracle


def test_premature_stop_recovers_until_waypoint_advance():
    control = RecoveryControl(60, 4)
    assert control.choose(0, 1, 4.0) == (1, 'recovery')
    assert control.recovering and not control.model_success
    control.advance()
    assert not control.recovering
    assert control.choose(2, 2, 3.5) == (2, 'model')


def test_error_budget_and_nonterminal_stop():
    control = RecoveryControl(60, 4)
    for _ in range(12):
        assert control.choose(2, 1, 4)[0] == 2
    assert control.choose(2, 1, 4) == (1, 'recovery')
    control.advance()
    assert control.errors == 0
    # Even inside evaluation success radius, do not label unfinished route STOP.
    assert control.choose(0, 3, 2) == (3, 'stop_override')


def test_waypoint_advance_preserves_janus_accumulated_errors():
    control = RecoveryControl(60, 4)
    for _ in range(7):
        assert control.choose(2, 1, 4)[1] == 'model'
    control.advance()
    assert control.errors == 7
    for _ in range(5):
        assert control.choose(2, 1, 4)[1] == 'model'
    assert control.choose(2, 1, 4) == (1, 'recovery')


def test_save_filter_requires_real_terminal_stop():
    assert accepted_trajectory('oracle_stop', .2, .8, True)
    assert accepted_trajectory('oracle_stop', .2, .9, False)
    assert not accepted_trajectory('oracle_stop', .2, .9, True)
    assert not accepted_trajectory('max_steps', .2, .8, False)
    assert not accepted_trajectory('oracle_stop', .6, .8, False)


def test_resume_contract_cannot_change(tmp_path):
    bind_contract(tmp_path, {'teacher': 'a', 'ids': ['2']})
    bind_contract(tmp_path, {'teacher': 'a', 'ids': ['2']})
    with pytest.raises(ValueError, match='contract'):
        bind_contract(tmp_path, {'teacher': 'b', 'ids': ['2']})


def test_shards_exact_unique_cover():
    rows = [{'episode_id': str(i), 'scene_id': 'scene'} for i in range(11)]
    shards = [partition(rows, i, 4) for i in range(4)]
    ids = [x['episode_id'] for shard in shards for x in shard]
    assert sorted(ids) == sorted(str(i) for i in range(11))
    assert len(set(ids)) == 11
    with pytest.raises(ValueError):
        partition(rows + rows[:1], 0, 4)


def test_training_targets_are_executed_not_oracle(tmp_path):
    record = dict(episode_uid='r2r:train:s:1:recovery1', dataset='r2r',
                  episode_id='1', scene_id='s', instruction_id='1', instruction='Go.',
                  termination='oracle_stop', accepted=True, dataset_version='hash',
                  steps=[dict(step_id=0, executed_action=2, oracle_action=1, rgb_path='0.png'),
                         dict(step_id=1, executed_action=0, oracle_action=0, rgb_path='1.png')])
    row = training_row(record, tmp_path)
    assert [s['action_name'] for s in row['steps']] == ['TURN_LEFT', 'STOP']
    record['termination'] = 'max_steps'
    with pytest.raises(ValueError):
        training_row(record, tmp_path)


def test_two_point_route_uses_final_radius_and_pathfinder_errors_propagate():
    class Follower:
        def get_next_action(self, point):
            if point == 'bad':
                raise RuntimeError('unreachable')
            return 0 if point == 'reached' else 1
    radii = []
    def factory(radius):
        radii.append(radius)
        return Follower()
    oracle = ReferenceOracle(['start', 'goal'], factory)
    assert oracle.action() == (1, False)
    assert radii == [.25]
    oracle = ReferenceOracle(['start', 'reached', 'bad'], factory)
    with pytest.raises(RuntimeError, match='unreachable'):
        oracle.action()


def rollout_fixture(monkeypatch, tmp_path, max_steps=10, initial=4.):
    from s8n_dagger.collect import collect_episode, write_episode
    class Env:
        sim = None
        episode_over = False
        def __init__(self):
            self.sim = self
            self.t = 0
            self.x = 0
        def reset(self):
            return {'rgb': np.zeros((480, 640, 3), np.uint8)}
        def get_metrics(self):
            return {'distance_to_goal': initial if self.t == 0 else .1}
        def get_agent_state(self):
            return SimpleNamespace(position=np.array([self.x, 0, 0]),
                                   rotation=SimpleNamespace(x=0, y=0, z=0, w=1))
        def step(self, action):
            self.x += 0 if action == 0 else 5
            self.t += 1
            self.episode_over = action == 0
            return {'rgb': np.full((480, 640, 3), self.t, np.uint8)}
    env = Env()
    class Follower:
        def __init__(self, *args, **kwargs):
            pass
        def get_next_action(self, waypoint):
            if waypoint == 'middle':
                return 1 if env.t == 0 else 0
            return 3 if env.t < 2 else 0
    monkeypatch.setitem(sys.modules, 'habitat.tasks.nav.shortest_path_follower',
                        SimpleNamespace(ShortestPathFollower=Follower))
    token_ids = [[11, 99], [12, 99], [13, 99], [14, 99]]
    class Model:
        def call(self, operation, **kwargs):
            if operation == 'reset':
                return {'status': 'ok'}
            action = (0 if env.t == 0 else 3) if operation == 'propose' else kwargs['habitat_action_id']
            cls = {1: 0, 2: 1, 3: 2, 0: 3}[action]
            return dict(habitat_action_id=action, action_name={0:'STOP',1:'MOVE_FORWARD',2:'TURN_LEFT',3:'TURN_RIGHT'}[action],
                        class_id=cls, response_token_ids=token_ids[cls])
    args = SimpleNamespace(dataset='r2r', dataset_hash='hash', expert_only=False,
                           max_steps=max_steps, navigation=dict(eos=99, action_token_ids=token_ids, separator_ids=[198]))
    ep = SimpleNamespace(scene_id='/scenes/s/s.glb', episode_id='1', reference_path=['start','middle','goal'],
                         instruction=SimpleNamespace(instruction_text='Go.'))
    record, frames = collect_episode(env, Model(), ep, dict(scene_id='s', instruction={}), 60, args)
    (tmp_path / 'episodes').mkdir(exist_ok=True)
    (tmp_path / 'pending').mkdir(exist_ok=True)
    write_episode(tmp_path, record, frames, keep_rejected=True)
    return record


def test_rollout_recovers_releases_and_saves_pre_action_rgb(monkeypatch, tmp_path):
    from PIL import Image
    row = rollout_fixture(monkeypatch, tmp_path)
    assert row['accepted'] and not row['failed']
    assert [s['executed_action'] for s in row['steps']] == [1, 3, 0]
    assert [s['action_source'] for s in row['steps']] == ['recovery', 'model', 'oracle_stop']
    assert [int(np.asarray(Image.open(tmp_path / 'episodes/1' / s['rgb_path']))[0,0,0])
            for s in row['steps']] == [0, 1, 2]


def test_truncated_rollout_never_fabricates_stop(monkeypatch, tmp_path):
    row = rollout_fixture(monkeypatch, tmp_path, max_steps=1)
    assert row['termination'] == 'max_steps' and not row['accepted']
    assert [s['executed_action'] for s in row['steps']] == [1]
    assert not (tmp_path / 'episodes/1/training_episode.json').exists()


@pytest.mark.parametrize('value', [float('inf'), float('nan')])
def test_nonfinite_simulator_distance_is_durable_failure(monkeypatch, tmp_path, value):
    row = rollout_fixture(monkeypatch, tmp_path, initial=value)
    assert row['failed'] and row['initial_distance'] is None
    saved = json.loads((tmp_path / 'episodes/1/record.json').read_text())
    assert saved['failed'] and saved['steps'] == []
