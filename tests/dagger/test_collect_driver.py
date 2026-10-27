"""Habitat-side driver helpers with a fake Habitat environment (no simulator needed)."""
import gzip
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_expert_episode import H_FORWARD, H_LEFT, H_STOP, ScriptedModel  # noqa: E402

from qwen_vl.contracts import ACTIONS, validate_episode  # noqa: E402
from qwen_vl.dagger import collect_r2r  # noqa: E402
from qwen_vl.dagger.rules import STOP, DaggerRules  # noqa: E402
from qwen_vl.dagger.store import CollectionStore  # noqa: E402

FORWARD, LEFT = ACTIONS.index('MOVE_FORWARD'), ACTIONS.index('TURN_LEFT')


class FakeEnv:
    """Minimal habitat.Env surface used by HabitatWorld, on a 1-D corridor."""

    def __init__(self, goal=10.0, cap=500):
        self.goal, self.cap = goal, cap
        self.x, self.actions, self.episode_over, self.current_episode = 0.0, 0, False, None
        self.sim = types.SimpleNamespace(
            get_agent_state=lambda: types.SimpleNamespace(position=np.array([self.x, 0.0, 0.0], np.float32)))

    def _obs(self):
        rgb = np.zeros((480, 640, 3), np.uint8)
        rgb[0, 0, 0] = self.actions % 256
        return dict(rgb=rgb)

    def reset(self):
        self.x, self.actions, self.episode_over = 0.0, 0, False
        return self._obs()

    def step(self, action):
        assert not self.episode_over
        self.actions += 1
        if action == H_FORWARD:
            self.x += 0.25
        self.episode_over = action == H_STOP or self.actions >= self.cap
        return self._obs()

    def get_metrics(self):
        return dict(distance_to_goal=abs(self.goal - self.x), top_down_map=None)


def fake_factory(env):
    class Follower:
        def __init__(self, radius):
            self.radius = radius

        def get_next_action(self, point):
            gap = point[0] - env.x
            if abs(gap) <= self.radius:
                return H_STOP
            return H_FORWARD if gap > 0 else H_LEFT

    return Follower


SOURCE = dict(scene_id='mp3d/s/s.glb', instruction_id='5', instruction='Go.', dataset_version='v')


class DriverTests(unittest.TestCase):
    def run_one(self, proposals, root, oracle_length=40):
        env = FakeEnv()
        episode = types.SimpleNamespace(episode_id=7, reference_path=[[0, 0, 0], [5, 0, 0], [10, 0, 0]])
        args = types.SimpleNamespace(split='train', tag='t')
        store = CollectionStore(root, 0)
        with mock.patch.object(collect_r2r, 'habitat_follower_factory', lambda sim: fake_factory(env)):
            return collect_r2r.run_one(env, ScriptedModel(proposals), episode, DaggerRules(), SOURCE,
                                       oracle_length, args, store)

    def test_turn_only_takeover_is_dropped_like_streamvln(self):
        # Turning in place adds no path length: PL stays 1.0 and StreamVLN's rule drops the episode.
        with tempfile.TemporaryDirectory() as root:
            record = self.run_one(lambda t: LEFT if t < 13 else FORWARD, root)
            self.assertEqual((record['takeovers'], record['path_efficiency']), (1, 1.0))
            self.assertEqual((record['saved'], record['reason']), (False, 'near_oracle'))

    def test_takeover_episode_is_saved_with_a_valid_row(self):
        with tempfile.TemporaryDirectory() as root, \
                mock.patch.object(collect_r2r, 'path_efficiency', lambda start, travelled: 0.8):
            record = self.run_one(lambda t: LEFT if t < 13 else FORWARD, root)
            self.assertTrue(record['saved'], record)
            self.assertEqual(record['reason'], 'takeover')
            row = record['row']
            validate_episode(row)
            self.assertEqual(row['steps'][-1]['action_name'], 'STOP')
            self.assertEqual(len(row['steps']), record['steps'])
            self.assertEqual(row['dagger']['takeovers'], 1)
            self.assertLess(record['path_efficiency'], 0.93)
            json.dumps(record, allow_nan=False)

    def test_near_oracle_episode_is_journaled_not_saved(self):
        with tempfile.TemporaryDirectory() as root:
            record = self.run_one(lambda t: FORWARD, root)
            self.assertEqual((record['saved'], record['reason'], record['takeovers']), (False, 'near_oracle', 0))
            self.assertFalse(any(Path(root, 'frames').iterdir()))

    def test_clean_detour_is_not_saved(self):
        # A winding reference path can push PL below 0.85 with no deviation at all.
        with tempfile.TemporaryDirectory() as root, \
                mock.patch.object(collect_r2r, 'path_efficiency', lambda start, travelled: 0.8):
            record = self.run_one(lambda t: FORWARD, root)
            self.assertEqual((record['saved'], record['reason'], record['clean']), (False, 'clean', True))

    def test_check_response(self):
        ok = dict(episode_uid='u', step_id=3, class_id=0, action_name='MOVE_FORWARD', habitat_action_id=1)
        collect_r2r.check_response(dict(ok, source='model'), 'u', 3, None, 1)
        collect_r2r.check_response(dict(ok, source='forced'), 'u', 3, 0, None)
        collect_r2r.check_response(dict(ok, source='stop_intercepted'), 'u', 3, None, 0)
        for bad, args in [(dict(ok, source='model', step_id=4), (None, 1)),
                          (dict(ok, source='forced'), (2, None)),
                          (dict(ok, source='model'), (0, None)),
                          (dict(ok, source='stop_intercepted'), (None, 2)),
                          (dict(ok, source='other'), (None, 1))]:
            with self.assertRaises(collect_r2r.ModelProcessError):
                collect_r2r.check_response(bad, 'u', 3, *args)
        with self.assertRaises(ValueError):
            collect_r2r.check_response(dict(ok, source='model', habitat_action_id=2), 'u', 3, None, 1)

    def test_failures_are_journaled(self):
        with tempfile.TemporaryDirectory() as root:
            record = self.run_one(lambda t: 9, root)
            self.assertEqual((record['saved'], record['reason']), (False, 'failure'))
            self.assertTrue(record['failure'].startswith('abort'))

    def test_recoverable_model_failure_is_journaled(self):
        class Failing(ScriptedModel):
            def step(self, *args, **kwargs):
                raise ValueError('Invalid navigation response')

        env = FakeEnv()
        episode = types.SimpleNamespace(episode_id=8, reference_path=[[0, 0, 0], [10, 0, 0]])
        with tempfile.TemporaryDirectory() as root, \
                mock.patch.object(collect_r2r, 'habitat_follower_factory', lambda sim: fake_factory(env)):
            record = collect_r2r.run_one(env, Failing(None), episode, DaggerRules(), SOURCE, None,
                                         types.SimpleNamespace(split='train', tag='t'), CollectionStore(root, 0))
        self.assertEqual(record['reason'], 'failure')
        self.assertFalse(record['oracle_length_known'])

    def test_source_records_match_prepare_r2r(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root, 'train.json.gz')
            episodes = [dict(episode_id=1, scene_id='mp3d/a/a.glb', instruction=dict(instruction_text=' Walk. ')),
                        dict(episode_id=2, scene_id='mp3d/b/b.glb',
                             instruction=dict(instruction_text='Turn.', instruction_id=9))]
            with gzip.open(path, 'wt') as handle:
                json.dump(dict(episodes=episodes), handle)
            records = collect_r2r.source_records(path)
            self.assertEqual(records['1']['instruction_id'], '1')
            self.assertEqual(records['1']['instruction'], 'Walk.')
            self.assertEqual(records['2']['instruction_id'], '9')
            manifest = Path(root, 'm.jsonl')
            manifest.write_text(json.dumps(dict(dataset='r2rce', episode_id='1', steps=[{}] * 4)) + '\n'
                                + json.dumps(dict(dataset='rxrce', episode_id='1', steps=[{}] * 9)) + '\n')
            self.assertEqual(collect_r2r.oracle_lengths(manifest), {'1': 4})

    def test_rule_flags_cover_every_field(self):
        self.assertEqual(STOP, 3)
        for name in DaggerRules.__dataclass_fields__:
            self.assertTrue(name.islower())


if __name__ == '__main__':
    unittest.main()
