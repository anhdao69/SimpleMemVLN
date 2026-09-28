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
