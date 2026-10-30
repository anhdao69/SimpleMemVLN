import os
import torch
import pytest
from PIL import Image
from transformers import AutoProcessor
from qwen_vl.contracts import deep_merge
from qwen_vl.data.episode_serializer import EpisodeSerializer
import yaml


@pytest.fixture(params=["classification", "qwen_text"])
def serializer(request):
    model = os.environ.get("VLN_MODEL_PATH")
    if not model:
        pytest.skip("Set VLN_MODEL_PATH to the pinned processor snapshot")
    with open("configs/vln_r2r_v0_base.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(f"configs/vln_r2r_v0_{request.param}.yaml") as f:
        cfg = deep_merge(cfg, yaml.safe_load(f))
    return EpisodeSerializer(AutoProcessor.from_pretrained(model), cfg)


def test_offline_stream_fragments_and_supervision_agree(serializer, tmp_path):
    image = Image.new("RGB", (640, 480), (50, 100, 150))
    path = tmp_path / "rgb.png"
    image.save(path)
    ep = dict(
        episode_uid="r2r:train:s:1:1",
        dataset="r2r",
        dataset_version="test",
        split="train",
        scene_id="s",
        episode_id="1",
        instruction_id="1",
        instruction="Turn right and stop.",
        observation_action_alignment="observation_before_action",
        steps=[
            dict(step_id=i, rgb_path=str(path), action_name=a, is_valid=True)
            for i, a in enumerate(["TURN_RIGHT", "STOP"])
        ],
    )
    encoded = serializer.encode_episode(ep)
    ids = [serializer.encode_prefix(ep["instruction"])["input_ids"]]
    for t, cls in enumerate([2, 3]):
        block = serializer.encode_observation(image, t)
        ids.append(block["input_ids"])
        if serializer.mode == "qwen_text":
            ids.append(
                serializer.text_block(
                    serializer.action_ids[cls] + serializer.separator
                )["input_ids"]
            )
    assert torch.equal(encoded["input_ids"], torch.cat(ids, 1))
    assert encoded["num_actions"] == 2
    if serializer.mode == "classification":
        changed = dict(
            ep,
            steps=[
                dict(s, action_name="TURN_LEFT" if i == 0 else "STOP")
                for i, s in enumerate(ep["steps"])
            ],
        )
        assert torch.equal(
            encoded["input_ids"], serializer.encode_episode(changed)["input_ids"]
        )
        for pos in encoded["read_positions"]:
            assert serializer.tokenizer.decode(
                encoded["input_ids"][0, pos - 1 : pos + 1]
            ).endswith("Action:")
    else:
        targets = encoded["input_ids"][0, encoded["response_target_positions"]]
        assert targets.tolist() == serializer.action_ids[2] + serializer.action_ids[3]
        assert int(targets.eq(serializer.eos).sum()) == 2


def test_oracle_targets_keep_executed_history(serializer, tmp_path):
    if serializer.mode != "qwen_text":
        pytest.skip("Oracle relabeling is defined for qwen_text responses")
    path = tmp_path / "rgb.png"
    Image.new("RGB", (640, 480), (50, 100, 150)).save(path)
    executed = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "TURN_LEFT", "STOP"]
    oracle = ["TURN_LEFT", "MOVE_FORWARD", "TURN_LEFT", "TURN_LEFT", "STOP"]
    ep = dict(
        episode_uid="r2r:train:s:1:1",
        dataset="r2r",
        dataset_version="test",
        split="train",
        scene_id="s",
        episode_id="1",
        instruction_id="1",
        instruction="Turn left and stop.",
        observation_action_alignment="observation_before_action",
        steps=[
            dict(step_id=i, rgb_path=str(path), action_name=a, is_valid=True)
            for i, a in enumerate(executed)
        ],
    )
    plain = serializer.encode_episode(ep)
    assert "response_target_ids" not in plain
    relabeled = dict(
        ep,
        steps=[dict(s, target_action_name=o) for s, o in zip(ep["steps"], oracle)],
    )
    encoded = serializer.encode_episode(relabeled)
    # History, images and positions are those of the executed rollout.
    for key in ("input_ids", "mm_token_type_ids", "pixel_values", "image_grid_thw"):
        assert torch.equal(encoded[key], plain[key])
    assert encoded["step_plan"] == plain["step_plan"]
    ids = dict(zip(("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"), serializer.action_ids))
    move, left, right = ids["MOVE_FORWARD"], ids["TURN_LEFT"], ids["TURN_RIGHT"]
    # Oracle tokens up to and including the first one that differs from the executed response.
    expected = [left[:1], move[:1], left[:2], left, ids["STOP"]]
    assert encoded["response_target_ids"].tolist() == sum(expected, [])
    assert encoded["response_action_index"].tolist() == sum(
        ([t] * len(x) for t, x in enumerate(expected)), []
    )
    plain_positions = plain["response_target_positions"].tolist()
    starts = [plain_positions[plain["response_action_index"].tolist().index(t)] for t in range(5)]
    assert encoded["response_target_positions"].tolist() == sum(
        (list(range(s, s + len(x))) for s, x in zip(starts, expected)), []
    )
    assert encoded["response_action_lengths"].tolist() == [len(ids[o]) for o in oracle]
    assert encoded["action_class_ids"].tolist() == [
        ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP").index(o) for o in oracle
    ]
    agreeing = dict(
        ep,
        steps=[dict(s, target_action_name=s["action_name"]) for s in ep["steps"]],
    )
    same = serializer.encode_episode(agreeing)
    assert same["response_target_ids"].tolist() == plain["input_ids"][
        0, plain["response_target_positions"]
    ].tolist()
    assert torch.equal(same["response_target_positions"], plain["response_target_positions"])
