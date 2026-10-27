"""Deviation rules against the StreamVLN/JanusVLN reference logic."""
import itertools
import unittest

from qwen_vl.dagger.rules import (
    STOP, DaggerRules, EpisodeController, EscapedStop, path_efficiency, path_length, save_decision)

FORWARD, LEFT = 0, 1


def streamvln_keep(final_distance, model_success, pl):
    """streamvln_dagger.py / JanusVLN dagger.py ``episode_save``."""
    return final_distance < 0.5 and (((not model_success) and pl < 0.93) or pl < 0.85)


class RuleTests(unittest.TestCase):
    def test_segment_length_matches_streamvln(self):
        self.assertEqual(EpisodeController(DaggerRules(), 50, 6).segment_actions, 10)
        self.assertEqual(EpisodeController(DaggerRules(), 3, 6).segment_actions, 1)
        with self.assertRaises(ValueError):
            EpisodeController(DaggerRules(), 50, 1)

    def test_error_ratio_takeover_starts_at_the_next_decision(self):
        c = EpisodeController(DaggerRules(), 50, 6)                 # 10 actions per segment
        for _ in range(8):                                          # 8 / 10 = 0.8 is not > 0.8
            self.assertEqual(c.request(FORWARD), dict(forced_class=None, intercept_with=FORWARD))
            self.assertEqual(c.record('model', LEFT, FORWARD, 9.0), 'model')
        self.assertEqual(c.control, 'model')
        c.record('model', FORWARD, FORWARD, 9.0)                    # agreement does not count
        self.assertEqual(c.errors, 8)
        c.record('model', LEFT, FORWARD, 9.0)                       # 9 / 10 > 0.8
        self.assertEqual((c.control, c.errors), ('expert', 0))
        self.assertEqual((c.takeovers, c.model_failed), (0, False))   # counted once it forces the expert
        self.assertEqual(c.request(FORWARD), dict(forced_class=FORWARD, intercept_with=None))
        self.assertEqual(c.record('forced', FORWARD, FORWARD, 9.0), 'expert')
        self.assertEqual((c.takeovers, c.error_takeovers), (1, 1))
        c.waypoint_reached()
        self.assertEqual(c.control, 'model')
        self.assertTrue(c.model_failed)
        self.assertEqual((c.model_steps, c.expert_steps, c.deviations), (10, 1, 9))

    def test_absolute_error_cap(self):
        c = EpisodeController(DaggerRules(), 1000, 2)               # ratio never triggers
        for _ in range(12):
            c.record('model', LEFT, FORWARD, 9.0)
        self.assertEqual(c.control, 'model')
        c.record('model', LEFT, FORWARD, 9.0)                       # 13 > 12
        self.assertEqual(c.control, 'expert')

    def test_errors_persist_across_waypoints_like_streamvln(self):
        c = EpisodeController(DaggerRules(), 50, 6)
        for _ in range(5):
            c.record('model', LEFT, FORWARD, 9.0)
        c.waypoint_reached()
        self.assertEqual(c.errors, 5)

    def test_stop_interception(self):
        c = EpisodeController(DaggerRules(), 50, 6)
        self.assertEqual(c.record('stop_intercepted', FORWARD, FORWARD, 2.99), 'stop_override')
        # StreamVLN compares a policy STOP with the expert action: it is a deviation.
        self.assertEqual((c.control, c.stop_overrides, c.takeovers, c.errors), ('model', 1, 0, 1))
        c.record('stop_intercepted', LEFT, LEFT, 3.0)
        self.assertEqual((c.control, c.far_stop_takeovers, c.takeovers, c.errors), ('expert', 1, 1, 0))
        self.assertTrue(c.model_failed)                              # the override itself ran the expert
        with self.assertRaises(RuntimeError):
            c.record('stop_intercepted', LEFT, LEFT, 9.0)           # policy decision during takeover

    def test_repeated_near_stops_trigger_an_error_takeover(self):
        c = EpisodeController(DaggerRules(), 50, 6)                 # 10 actions per segment
        for _ in range(9):
            c.record('stop_intercepted', FORWARD, FORWARD, 1.0)
        self.assertEqual((c.control, c.takeovers), ('expert', 0))  # pending until a forced decision
        c.record('forced', FORWARD, FORWARD, 1.0)
        self.assertEqual((c.takeovers, c.error_takeovers), (1, 1))

    def test_deferred_takeover_cancelled_by_a_waypoint_is_not_counted(self):
        c = EpisodeController(DaggerRules(), 1000, 2)
        for _ in range(13):
            c.record('model', LEFT, FORWARD, 9.0)
        self.assertEqual(c.control, 'expert')
        c.waypoint_reached()
        self.assertEqual((c.control, c.takeovers, c.cancelled_takeovers, c.model_failed), ('model', 0, 1, False))

    def test_clean_episode(self):
        c = EpisodeController(DaggerRules(), 50, 6)
        c.record('model', FORWARD, FORWARD, 9.0)
        self.assertTrue(c.clean)
        c.record('model', LEFT, FORWARD, 9.0)
        self.assertFalse(c.clean)

    def test_invariants(self):
        c = EpisodeController(DaggerRules(), 50, 6)
        with self.assertRaises(RuntimeError):
            c.record('forced', FORWARD, FORWARD, 1.0)               # forced outside a takeover
        with self.assertRaises(EscapedStop):
            c.record('model', STOP, FORWARD, 1.0)
        with self.assertRaises(RuntimeError):
            c.record('stop_intercepted', LEFT, FORWARD, 1.0)        # must execute the expert action
        with self.assertRaises(ValueError):
            c.record('policy', FORWARD, FORWARD, 1.0)
        with self.assertRaises(ValueError):
            c.request(STOP)
        c._takeover('error', realized=True)
        with self.assertRaises(RuntimeError):
            c.record('forced', LEFT, FORWARD, 1.0)
        with self.assertRaises(RuntimeError):
            c.record('model', FORWARD, FORWARD, 1.0)

    def test_save_decision_reproduces_streamvln(self):
        rules = DaggerRules()
        for distance, model_failed, pl in itertools.product(
                (0.0, 0.3, 0.49, 0.5, 0.8), (False, True), (0.3, 0.84, 0.85, 0.9, 0.92, 0.93, 0.99, 1.0)):
            keep, _ = save_decision(rules, True, distance, 40, pl, model_failed)
            self.assertEqual(keep, streamvln_keep(distance, not model_failed, pl), (distance, model_failed, pl))
        self.assertEqual(save_decision(rules, False, 0.1, 40, 0.5, True), (False, 'goal_not_reached'))
        self.assertEqual(save_decision(rules, True, 0.1, 184, 0.5, True), (False, 'too_long'))
        self.assertEqual(save_decision(rules, True, 0.1, 183, 0.5, True), (True, 'takeover'))
        self.assertEqual(save_decision(rules, True, 0.1, 40, 0.8, False), (True, 'detour'))
        self.assertEqual(save_decision(rules, True, 0.1, 40, 0.97, True), (False, 'near_oracle'))
        self.assertEqual(save_decision(rules, True, 0.1, 40, 0.8, False, clean=True), (False, 'clean'))
        self.assertEqual(save_decision(DaggerRules(save_clean=True), True, 0.1, 40, 0.8, False, clean=True),
                         (True, 'detour'))

    def test_path_metrics(self):
        self.assertAlmostEqual(path_length([[0, 0, 0], [3, 0, 4], [3, 0, 4], [3, 1, 4]]), 6.0)
        self.assertEqual(path_efficiency(5.0, 10.0), 0.5)
        self.assertEqual(path_efficiency(5.0, 4.0), 1.0)
        self.assertEqual(path_efficiency(0.0, 0.0), 1.0)

    def test_rule_validation(self):
        for bad in (dict(goal_radius=0), dict(max_errors=0), dict(max_saved_steps=1.5),
                    dict(save_pl_without_takeover=0.95), dict(save_pl_with_takeover=1.2), dict(save_clean=1)):
            with self.assertRaises(ValueError):
                DaggerRules(**bad)
        self.assertEqual(DaggerRules().to_dict()['waypoint_radius'], 1.8)


if __name__ == '__main__':
    unittest.main()
