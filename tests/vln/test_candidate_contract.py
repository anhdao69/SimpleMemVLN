import json
from copy import deepcopy
import os
import pytest
import torch
from PIL import Image
from transformers import AutoProcessor
from qwen_vl.contracts import (
    ACTIONS,
    HABITAT_IDS,
    action_result,
    deep_merge,
    validate_config,
)
from qwen_vl.data.episode_serializer import EpisodeSerializer
from qwen_vl.data.episode_dataset import EpisodeDataset
import yaml


def config(feedback="candidate_token"):
    with open("configs/vln_r2r_v0_base.yaml") as f:
        base = yaml.safe_load(f)
    return deep_merge(
        base,
        dict(
            model=dict(
                output_mode="candidate_logits", action_head_mode="lm_rows_trainable"
            ),
            observations=dict(
                serializer_version=(
                    "vln_candidate_logits_no_action_history_v1"
                    if feedback == "none" else "vln_candidate_logits_v1"
                ),
                append_action_tokens=feedback != "none",
                max_step_group_tokens=384,
                feedback_format=feedback,
            ),
            training=dict(class_weighting="none"),
        ),
    )


@pytest.fixture(params=["candidate_token", "canonical_action_text", "none"])
def candidate_serializer(request):
    path = os.environ.get("VLN_MODEL_PATH")
    if not path:
        pytest.skip("Pinned tokenizer required")
    return EpisodeSerializer(
        AutoProcessor.from_pretrained(path), validate_config(config(request.param))
    )


def test_candidate_config_and_mapping():
    validate_config(config())
    for i, (name, habitat) in enumerate(zip(ACTIONS, HABITAT_IDS)):
        assert action_result(i)["action_name"] == name
        assert action_result(i)["habitat_action_id"] == habitat
    for field, value in [("feedback_format", "bad")]:
        cfg = config()
        cfg["observations"][field] = value
        with pytest.raises(ValueError):
            validate_config(cfg)


def test_policy_overlay_preserves_joint_schedule_and_admission():
    from qwen_vl.train.vln_runtime import resolve_config

    cfg = resolve_config(
        "configs/vln_r2r_v0_base.yaml",
        "configs/vln_joint_b_2epoch_bs8.yaml",
        "configs/vln_memory_window8.yaml",
        "configs/vln_r2r_v0_candidate_logits.yaml",
    )
    assert cfg["model"]["output_mode"] == "candidate_logits"
    assert cfg["training"]["expected_total_steps"] == 7704
    assert cfg["training"]["model_max_length"] == 262144
    assert cfg["memory"]["mode"] == "window8"


def test_exact_token_boundary_and_history(candidate_serializer, tmp_path):
    s = candidate_serializer
    assert s.candidate_token_ids == [32, 33, 34, 35]
    assert len(set(s.candidate_token_ids)) == 4
    obs = s.observation_text
    assert obs.endswith("Action:\n")
    for label, token in zip("ABCD", s.candidate_token_ids):
        assert s.ids(label) == [token]
        assert (
            s.ids(obs + label + "<|im_end|>\n")
            == s.ids(obs) + [token, s.eos] + s.separator
        )
    image = Image.new("RGB", (640, 480))
    path = tmp_path / "rgb.png"
    image.save(path)
    ep = dict(
        episode_uid="e",
        dataset="r2r",
        dataset_version="test",
        split="train",
        scene_id="s",
        episode_id="1",
        instruction_id="1",
        instruction="Go forward then turn and stop.",
        observation_action_alignment="observation_before_action",
        steps=[
            dict(step_id=i, rgb_path=str(path), action_name=a, is_valid=True)
            for i, a in enumerate(ACTIONS)
        ],
    )
    encoded = s.encode_episode(ep)
    assert encoded["action_class_ids"].tolist() == [0, 1, 2, 3]
    assert encoded["response_target_positions"].numel() == 0
    for i, (start, end) in enumerate(encoded["step_plan"][1]):
        read = int(encoded["read_positions"][i])
        expected = (
            (
                [32 + i]
                if s.config["observations"]["feedback_format"] == "candidate_token"
                else ([] if s.config["observations"]["feedback_format"] == "none"
                      else s.ids(ACTIONS[i]))
            )
            + [s.eos]
            + s.separator
        )
        assert s.feedback_ids(i) == expected
        assert encoded["input_ids"][0, read + 1 : end].tolist() == expected
        assert read == start + s.encode_observation(image)["input_ids"].numel() - 1
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(ep) + "\n")
    ds = EpisodeDataset(manifest, s)
    assert ds.encoded_lengths[0] == encoded["input_ids"].numel()
    assert s.metadata()["candidate_token_ids"] == [32, 33, 34, 35]
    # Metamorphic guard: teacher-forcing targets must not leak into no-history inputs.
    changed = deepcopy(ep)
    for step in changed["steps"][:-1]:
        step["action_name"] = ACTIONS[(ACTIONS.index(step["action_name"]) + 1) % 3]
    altered = s.encode_episode(changed)
    assert not torch.equal(encoded["action_class_ids"], altered["action_class_ids"])
    if s.config["observations"]["feedback_format"] == "none":
        for key in ("input_ids", "mm_token_type_ids", "pixel_values", "image_grid_thw", "read_positions"):
            assert torch.equal(encoded[key], altered[key]), key
        assert encoded["step_plan"] == altered["step_plan"]
        assert s.ids(obs + "<|im_end|>\n") == s.ids(obs) + [s.eos] + s.separator
    else:
        assert not torch.equal(encoded["input_ids"], altered["input_ids"])


def test_reject_multitoken_or_context_merging_candidate(
    candidate_serializer, monkeypatch
):
    s = candidate_serializer
    from qwen_vl.data.candidates import validate_candidates

    with pytest.raises(ValueError):
        validate_candidates(
            s.tokenizer, ["A", "B", "C", "MOVE_FORWARD"], s.observation_text
        )
    with pytest.raises(ValueError):
        validate_candidates(s.tokenizer, ["A", "B", "C", "D"], "Action:")
    with pytest.raises(ValueError):
        validate_candidates(s.tokenizer, ["A", "A", "B", "C"], s.observation_text)
