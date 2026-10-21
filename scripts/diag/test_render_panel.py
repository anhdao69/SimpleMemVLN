"""CPU contracts for observation-before-action rendering and strict RGB parity."""
import unittest
import numpy as np
from scripts.diag.render_panel import compare_rgb, render_episode, summarize_frames


class FakeEnv:
    def __init__(self):
        self.position = 0
        self.episode_over = False
        self.actions = []
    def reset(self):
        self.position = 0
        self.episode_over = False
        return {'rgb': np.full((480, 640, 3), self.position, np.uint8)}
    def step(self, action):
        self.actions.append(action)
        self.position += 1
        self.episode_over = action == 0
        return {'rgb': np.full((480, 640, 3), self.position, np.uint8)}


class RenderingTests(unittest.TestCase):
    def test_observations_precede_actions_and_stop_has_frame(self):
        env = FakeEnv()
        values = []
        render_episode(env, object(), [1, 3, 0],
                       lambda step, action, rgb: values.append((step, action, int(rgb[0, 0, 0]))))
        self.assertEqual(values, [(0, 1, 0), (1, 3, 1), (2, 0, 2)])
        self.assertEqual(env.actions, [1, 3, 0])

    def test_reject_nonterminal_or_missing_stop(self):
        for actions in ([1, 0, 3], [1, 3], [], [7, 0]):
            with self.assertRaises(ValueError):
                render_episode(FakeEnv(), object(), actions, lambda *args: None)

    def test_reject_early_episode_end(self):
        class EarlyEnv(FakeEnv):
            def step(self, action):
                obs = super().step(action)
                self.episode_over = True
                return obs
        with self.assertRaises(ValueError):
            render_episode(EarlyEnv(), object(), [1, 0], lambda *args: None)

    def test_channel_mae_avoids_uint8_wrap_and_normalizes_once(self):
        a = np.zeros((480, 640, 3), np.uint8)
        b = a.copy()
        b[:, :, 0] = 255
        result = compare_rgb(a, b)
        self.assertEqual(result['mae_per_channel'], [1.0, 0.0, 0.0])
        self.assertEqual(result['max_abs_pixel'], 255)
        self.assertAlmostEqual(result['mean_absolute_error'], 1 / 3)

    def test_shape_and_dtype_fail_without_conversion(self):
        rgb = np.zeros((480, 640, 3), np.uint8)
        for bad in (rgb.transpose(1, 0, 2), rgb[:, :, :1], rgb.astype(float)):
            with self.assertRaises(ValueError):
                compare_rgb(rgb, bad)

    def test_each_channel_must_pass_not_only_rgb_average(self):
        a = np.zeros((480, 640, 3), np.uint8)
        b = a.copy(); b[:, :, 0] = 2
        result = summarize_frames([compare_rgb(a, b)])
        self.assertFalse(result['passed'])
        self.assertLess(result['mean_absolute_error'], 1 / 255)
        b[:, :, 0] = 1
        self.assertTrue(summarize_frames([compare_rgb(a, b)])['passed'])

    def test_equal_frame_weights_and_distribution(self):
        a = np.zeros((480, 640, 3), np.uint8)
        b = a.copy(); b[:] = 2
        result = summarize_frames([compare_rgb(a, a), compare_rgb(a, b)])
        np.testing.assert_allclose(result['mae_per_channel'], [1/255]*3)
        self.assertTrue(result['passed'])
        self.assertEqual(result['frame_count'], 2)
        self.assertEqual(result['max_abs_pixel'], 2)

if __name__ == '__main__':
    unittest.main()
