"""Reference-path expert and the collection episode on a 1-D corridor world."""
import unittest

import numpy as np

from qwen_vl.contracts import ACTIONS, HABITAT_IDS
from qwen_vl.dagger.episode import collect_episode
from qwen_vl.dagger.expert import ExpertError, ReferencePathExpert
from qwen_vl.dagger.rules import STOP, DaggerRules, EpisodeController

H_STOP, H_FORWARD, H_LEFT = 0, 1, 2
FORWARD, LEFT = ACTIONS.index('MOVE_FORWARD'), ACTIONS.index('TURN_LEFT')


class Corridor:
    """Agent on the x axis; MOVE_FORWARD advances 0.25 m, turns do not move, STOP ends."""

    def __init__(self, goal, cap=500):
        self.goal, self.cap = goal, cap
        self.x, self.actions, self.over = 0.0, 0, False

    def frame(self):
        rgb = np.zeros((480, 640, 3), np.uint8)
        rgb[0, 0, 0] = self.actions % 256
        return rgb

    def reset(self):
        self.x, self.actions, self.over = 0.0, 0, False
        return self.frame()

    def step(self, action):
        if self.over:
            raise AssertionError('step after the episode ended')
        self.actions += 1
        if action == H_FORWARD:
            self.x += 0.25
        if action == H_STOP or self.actions >= self.cap:
            self.over = True
            return None
        return self.frame()

    def distance_to_goal(self):
        return abs(self.goal - self.x)

    def position(self):
        return [self.x, 0.0, 0.0]


class Follower:
    def __init__(self, world, radius):
        self.world, self.radius = world, radius

    def get_next_action(self, point):
        gap = point[0] - self.world.x
        if abs(gap) <= self.radius:
            return H_STOP
        return H_FORWARD if gap > 0 else H_LEFT


class ScriptedModel:
    """Session semantics of DaggerSession.step with a scripted policy proposal per decision."""

    def __init__(self, proposals, escaped_stop_at=None):
        self.proposals, self.escaped_stop_at = proposals, escaped_stop_at
        self.calls = []

    def reset(self, uid, instruction):
        self.calls.clear()

    def step(self, uid, step_id, rgb, forced_class=None, intercept_with=None):
        self.calls.append((step_id, int(rgb[0, 0, 0]), forced_class, intercept_with))
        if forced_class is not None:
            return dict(class_id=forced_class, source='forced')
        if step_id == self.escaped_stop_at:
            return dict(class_id=STOP, source='model')
        proposal = self.proposals(step_id)
        if proposal == STOP and intercept_with is not None:
            return dict(class_id=intercept_with, source='stop_intercepted')
        return dict(class_id=proposal, source='model')


def run(proposals, goal=10.0, points=((0, 0, 0), (5, 0, 0), (10, 0, 0)), oracle=40, cap=500,
        max_actions=500, escaped_stop_at=None):
    world = Corridor(goal, cap)
    rules = DaggerRules()
    expert = ReferencePathExpert(points, lambda radius: Follower(world, radius), rules)
    controller = EpisodeController(rules, oracle, expert.waypoints)
    model = ScriptedModel(proposals, escaped_stop_at)
    result = collect_episode(world, model, expert, controller, 'uid', 'walk', max_actions=max_actions)
    return result, controller, model, world


class ExpertTests(unittest.TestCase):
    def test_waypoints_are_skipped_but_the_goal_is_not(self):
        world = Corridor(10.0)
        expert = ReferencePathExpert([(0, 0, 0), (1, 0, 0), (1.5, 0, 0), (2, 0, 0), (10, 0, 0)],
                                     lambda r: Follower(world, r), DaggerRules())
        self.assertEqual(expert.advance(), 2)                      # 1.0 and 1.5 m are within 1.8 m
        self.assertEqual(expert.index, 3)                          # 2.0 m is not
        self.assertEqual(expert.action(), H_FORWARD)
        world.x = 0.5
        self.assertEqual(expert.advance(), 1)                      # the final point is never skipped
        self.assertEqual(expert.index, 4)
        self.assertEqual(expert.action(), H_FORWARD)
        world.x = 9.8
        self.assertEqual(expert.advance(), 0)
        self.assertEqual(expert.action(), H_STOP)                  # within the 0.25 m goal radius

    def test_action_requires_advance_at_each_pose(self):
        world = Corridor(10.0)
        expert = ReferencePathExpert([(0, 0, 0), (1, 0, 0), (10, 0, 0)], lambda r: Follower(world, r),
                                     DaggerRules())
        with self.assertRaises(ExpertError):
            expert.action()
        expert.advance()
        with self.assertRaises(ExpertError):
            expert.advance()                                       # answer not consumed yet
        expert.action()
        with self.assertRaises(ExpertError):
            expert.action()                                        # consumed

    def test_one_follower_query_per_target_per_decision(self):
        # habitat-sim's GreedyFollower records every answer for thrashing detection.
        world = Corridor(10.0)
        calls = []

        class Counting(Follower):
            def get_next_action(self, point):
                calls.append((self.radius, point[0]))
                return super().get_next_action(point)

        expert = ReferencePathExpert([(0, 0, 0), (1, 0, 0), (5, 0, 0), (10, 0, 0)],
                                     lambda r: Counting(world, r), DaggerRules())
        expert.advance()
        expert.action()
        self.assertEqual(calls, [(1.8, 1.0), (1.8, 5.0)])         # reached waypoint, then the answer
        calls.clear()
        world.x = 0.25
        expert.advance()
        expert.action()
        self.assertEqual(calls, [(1.8, 5.0)])

    def test_follower_failures_raise(self):
        class Broken:
            def __init__(self, value):
                self.value = value

            def get_next_action(self, point):
                if isinstance(self.value, Exception):
                    raise self.value
                return self.value

        for value in (None, 7, RuntimeError('GreedyFollowerError')):
            expert = ReferencePathExpert([(0, 0, 0), (5, 0, 0)], lambda r, v=value: Broken(v), DaggerRules())
            with self.assertRaises(ExpertError):
                expert.advance()
        with self.assertRaises(ExpertError):
            ReferencePathExpert([(0, 0, 0)], lambda r: None, DaggerRules())


class EpisodeTests(unittest.TestCase):
    def test_on_track_policy_ends_with_one_expert_stop(self):
        result, controller, model, world = run(lambda t: FORWARD)
        self.assertTrue(result['final_reached'])
        self.assertIsNone(result['failure'])
        self.assertEqual(len(result['frames']), len(result['actions']))
        self.assertEqual(result['actions'][-1], STOP)
        self.assertNotIn(STOP, result['actions'][:-1])
        self.assertEqual(result['steps'][-1]['source'], 'expert_goal')
        self.assertEqual(controller.takeovers, 0)
        self.assertEqual(world.x, 9.75)                            # within the 0.25 m goal radius
        # Frame t is the observation of decision t (observation_before_action).
        for t, (step_id, marker, _, _) in enumerate(model.calls):
            self.assertEqual((step_id, marker), (t, t % 256))
            self.assertEqual(int(result['frames'][t][0, 0, 0]), t % 256)
        self.assertEqual(len(result['positions']), len(result['actions']))   # final STOP does not move

    def test_deviation_takeover_and_return_of_control(self):
        # Turn in place: segment = 40 / 2 = 20 actions, so the ratio rule needs 17 errors, but the
        # absolute cap (errors > 12) fires first, at the 13th error (decision 12).  The takeover
        # starts at decision 13 and ends when waypoint 1 (5 m, radius 1.8 m) is reached at x = 3.25.
        result, controller, model, _ = run(lambda t: LEFT if t < 30 else FORWARD)
        sources = [s['source'] for s in result['steps']]
        self.assertEqual(sources[:13], ['model'] * 13)
        self.assertEqual(sources[13:26], ['expert'] * 13)
        self.assertEqual(sources[26], 'model')
        self.assertEqual(result['steps'][26]['waypoint'], 2)
        self.assertEqual(controller.error_takeovers, 1)
        first_forced = next(call for call in model.calls if call[2] is not None)
        self.assertEqual(first_forced[0], 13)
        self.assertEqual(result['actions'][12], LEFT)              # the crossing decision is executed
        self.assertTrue(result['final_reached'])

    def test_far_policy_stop_is_overridden_and_starts_a_takeover(self):
        result, controller, _, _ = run(lambda t: STOP if t == 3 else FORWARD)
        self.assertEqual(result['steps'][3]['source'], 'stop_override')
        self.assertEqual(result['actions'][3], FORWARD)
        self.assertEqual(result['steps'][4]['source'], 'expert')
        self.assertEqual((controller.far_stop_takeovers, controller.takeovers), (1, 1))

    def test_near_policy_stop_is_overridden_without_takeover(self):
        result, controller, _, _ = run(lambda t: STOP if t == 30 else FORWARD)   # 2.5 m from the goal
        self.assertEqual(result['steps'][30]['source'], 'stop_override')
        self.assertEqual(result['steps'][31]['source'], 'model')
        self.assertEqual((controller.stop_overrides, controller.takeovers, controller.deviations), (1, 0, 1))

    def test_cap_and_aborts_are_failures(self):
        result, _, _, _ = run(lambda t: FORWARD, max_actions=5)
        self.assertEqual((result['failure'], result['final_reached']), ('action_cap', False))
        result, _, _, _ = run(lambda t: FORWARD, cap=5)
        self.assertEqual((result['failure'], result['final_reached']), ('action_cap', False))
        result, _, _, _ = run(lambda t: FORWARD, escaped_stop_at=2)
        self.assertTrue(result['failure'].startswith('abort'))
        result, _, _, _ = run(lambda t: 9)
        self.assertTrue(result['failure'].startswith('abort'))
        self.assertEqual(HABITAT_IDS[STOP], H_STOP)


if __name__ == '__main__':
    unittest.main()
