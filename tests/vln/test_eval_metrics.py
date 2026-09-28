import unittest


class EvaluationFailureTests(unittest.TestCase):
    def test_failed_episode_stays_in_denominator_without_zero_navigation_error(self):
        from qwen_vl.eval.metrics import summarize

        result = summarize(
            [
                {
                    "failed": False,
                    "metrics": {"success": 1.0, "spl": 0.8, "distance_to_goal": 1.0},
                },
                {
                    "failed": True,
                    "metrics": {"success": 1.0, "spl": 0.9, "distance_to_goal": 2.0},
                },
                {"failed": True, "metrics": {}},
            ]
        )
        self.assertAlmostEqual(result["sr"], 1 / 3)
        self.assertAlmostEqual(result["spl"], 0.8 / 3)
        self.assertEqual(result["navigation_error"], 1.5)
        self.assertEqual(result["navigation_error_coverage"], 2)
        self.assertEqual(result["failures"], 2)

    def test_stop_and_latency_accounting_distinguishes_forced_action(self):
        from qwen_vl.eval.metrics import summarize

        result = summarize(
            [
                {
                    "failed": False,
                    "forced_stop": True,
                    "metrics": {"success": 0.0, "spl": 0.0},
                    "actions": [
                        {"predicted": 1, "executed": 1, "model_seconds": 0.1},
                        {"predicted": 1, "executed": 0, "model_seconds": 0.3},
                    ],
                }
            ]
        )
        self.assertEqual(result["predicted_action_counts"], {"1": 2})
        self.assertEqual(result["executed_action_counts"], {"1": 1, "0": 1})
        self.assertEqual(result["predicted_stops"], 0)
        self.assertEqual(result["forced_stops"], 1)
        self.assertAlmostEqual(result["model_latency_p50_seconds"], 0.2)
