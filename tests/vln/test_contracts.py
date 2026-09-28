"""Catch invalid action mapping, broken chronology and silent config changes."""
import unittest
import importlib.util


class ContractTests(unittest.TestCase):
    def test_navigation_contract_available(self):
        self.assertIsNotNone(importlib.util.find_spec("qwen_vl.contracts"))

    def test_action_mapping_and_strict_parse(self):
        from qwen_vl.contracts import action_result, parse_action

        self.assertEqual(action_result(0)["habitat_action_id"], 1)
        self.assertEqual(action_result(3)["habitat_action_id"], 0)
        self.assertEqual(parse_action("  TURN_LEFT\n"), 1)
        for text in ["STOP now", "MOVE_FORWARD<|im_end|>", "left", ""]:
            with self.assertRaises(ValueError):
                parse_action(text)

    def test_chronology_requires_real_terminal_observation(self):
        from qwen_vl.contracts import validate_episode

        ep = dict(
            episode_uid="r2r:train:scene:1:1",
            scene_id="scene",
            episode_id="1",
            instruction_id="1",
            instruction="Walk.",
            dataset="r2r",
            dataset_version="test",
            split="train",
            observation_action_alignment="observation_before_action",
            steps=[
                dict(
                    step_id=0,
                    rgb_path="a.png",
                    action_name="MOVE_FORWARD",
                    is_valid=True,
                ),
                dict(step_id=1, rgb_path="b.png", action_name="STOP", is_valid=True),
            ],
        )
        validate_episode(ep, check_images=False)
        ep["steps"][1]["step_id"] = 2
        with self.assertRaises(ValueError):
            validate_episode(ep, check_images=False)
        ep["steps"][1]["step_id"] = 1
        ep["steps"][1]["action_name"] = "TURN_LEFT"
        with self.assertRaises(ValueError):
            validate_episode(ep, check_images=False)

    def test_deep_merge_preserves_nested_fields(self):
        from qwen_vl.contracts import deep_merge

        self.assertEqual(
            deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}),
            {"a": {"b": 3, "c": 2}},
        )

    def test_resume_rejects_visibility_and_data_changes(self):
        from qwen_vl.contracts import validate_resume_contract

        saved = {"navigation": {"mode": "full_context"}, "data": {"ids": ["a", "b"]}}
        validate_resume_contract(saved, {"mode": "full_context"}, {"ids": ["a", "b"]})
        with self.assertRaises(ValueError):
            validate_resume_contract(saved, {"mode": "window8"}, {"ids": ["a", "b"]})
        with self.assertRaises(ValueError):
            validate_resume_contract(
                saved, {"mode": "full_context"}, {"ids": ["b", "a"]}
            )

    def test_parity_selection_rejects_empty_and_missing_ids(self):
        from qwen_vl.contracts import select_episode_ids

        eps = [{"episode_uid": "a"}, {"episode_uid": "b"}]
        self.assertEqual(select_episode_ids(eps, ["b"]), [eps[1]])
        for ids in ([], ["missing"], ["a", "missing"]):
            with self.assertRaises(ValueError):
                select_episode_ids(eps, ids)


if __name__ == "__main__":
    unittest.main()
