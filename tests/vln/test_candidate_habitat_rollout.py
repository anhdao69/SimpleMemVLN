"""Candidate-logit decisions must retain their exact Habitat rollout evidence."""

from types import SimpleNamespace

import numpy as np
import pytest

from qwen_vl.eval.habitat_r2r import (
    load_journal, rollout_episode, select_episodes, validate_prediction,
)
from qwen_vl.stream.transport import decode_rgb, encode_rgb


def test_candidate_decision_executes_stop_and_preserves_logits():
    class Env:
        episode_over = False

        def __init__(self):
            self.actions = []
            self.sim = SimpleNamespace(
                get_agent_state=lambda: SimpleNamespace(position=np.zeros(3))
            )

        def reset(self):
            return {"rgb": np.zeros((480, 640, 3), dtype=np.uint8)}

        def step(self, action):
            self.actions.append(action)
            self.episode_over = action == 0
            return self.reset()

        def get_metrics(self):
            return {"success": 1.0, "spl": 1.0, "distance_to_goal": 1.0}

    class Model:
        def reset(self, uid, instruction):
            assert instruction == "stop here"

        def observe(self, uid, step, rgb):
            assert step == 0
            return {
                "status": "ok",
                "class_id": 3,
                "habitat_action_id": 0,
                "action_name": "STOP",
                "action_logits": [-3.0, -2.0, -1.0, 4.0],
                "generated_tokens": 0,
                "model_seconds": 0.1,
                "retained_kv_tokens": 500,
                "peak_allocated_gib": 9.0,
            }

    env = Env()
    episode = SimpleNamespace(
        scene_id="scene", episode_id="7",
        instruction=SimpleNamespace(instruction_text="stop here"),
    )
    row = rollout_episode(env, Model(), episode, "val_unseen")
    assert env.actions == [0]
    assert row["metrics"]["success"] == 1.0
    assert row["actions"][0]["class_id"] == 3
    assert row["actions"][0]["action_logits"] == [-3.0, -2.0, -1.0, 4.0]
    assert row["actions"][0]["generated_tokens"] == 0
    assert row["forced_stop"] is False


def test_forced_stop_preserves_model_prediction():
    class Env:
        episode_over = False

        def __init__(self):
            self.actions = []
            self.sim = SimpleNamespace(
                get_agent_state=lambda: SimpleNamespace(position=np.zeros(3))
            )

        def reset(self):
            return {"rgb": np.zeros((480, 640, 3), dtype=np.uint8)}

        def step(self, action):
            self.actions.append(action)
            self.episode_over = action == 0
            return self.reset()

        def get_metrics(self):
            return {"success": 0.0, "spl": 0.0, "distance_to_goal": 5.0}

    class Model:
        def reset(self, uid, instruction):
            pass

        def observe(self, uid, step, rgb):
            return {"habitat_action_id": 1, "class_id": 0,
                    "action_name": "MOVE_FORWARD", "model_seconds": 0.1,
                    "retained_kv_tokens": 100, "peak_allocated_gib": 9.0}

    env = Env()
    episode = SimpleNamespace(
        scene_id="scene", episode_id="8",
        instruction=SimpleNamespace(instruction_text="go"),
    )
    row = rollout_episode(env, Model(), episode, "val_unseen", max_steps=2)
    assert env.actions == [1, 0]
    assert row["forced_stop"] is True
    assert row["actions"][-1]["predicted"] == 1
    assert row["actions"][-1]["executed"] == 0


def test_journal_rejects_duplicate_and_keeps_complete_prior_rows(tmp_path):
    path = tmp_path / "episodes.jsonl"
    path.write_text('{"episode_id":"1"}\n{"episode_id":"2","actions":[')
    assert load_journal(path, {"1", "2"}) == [{"episode_id": "1"}]
    assert path.read_text() == '{"episode_id":"1"}\n'
    assert path.with_suffix(".jsonl.incomplete-tail").exists()
    path.write_text('{"episode_id":"1"}\n{"episode_id":"1"}\n')
    with pytest.raises(ValueError, match="Duplicate"):
        load_journal(path, {"1", "2"})


def test_selection_and_lossless_rgb_transport():
    episodes = [SimpleNamespace(episode_id="2", scene_id="b"),
                SimpleNamespace(episode_id="1", scene_id="a")]
    assert [e.episode_id for e in select_episodes(episodes)] == ["1", "2"]
    with pytest.raises(ValueError, match="unique"):
        select_episodes(episodes, ["1", "1"])
    rgb = np.random.default_rng(7).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    assert np.array_equal(rgb, np.asarray(decode_rgb(encode_rgb(rgb))))


def test_candidate_action_metadata_must_match_habitat_action():
    valid = {"class_id": 3, "action_name": "STOP", "habitat_action_id": 0}
    assert validate_prediction(valid) == 0
    with pytest.raises(ValueError, match="inconsistent"):
        validate_prediction({**valid, "habitat_action_id": 1})
    with pytest.raises(ValueError, match="inconsistent"):
        validate_prediction({**valid, "action_name": "MOVE_FORWARD"})
