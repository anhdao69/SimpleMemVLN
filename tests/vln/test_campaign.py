"""Campaign gates: catch wrong schedules, checkpoint pruning and unsafe uploads."""
import importlib
import hashlib
from types import SimpleNamespace

import pytest


def campaign():
    spec = importlib.util.find_spec("qwen_vl.train.campaign")
    assert spec is not None, "Campaign safety gates are missing"
    return importlib.import_module("qwen_vl.train.campaign")


def test_actual_schedule_rejects_wrong_step_or_warmup_count():
    gate = campaign().ScheduleGuard(4059, 122)
    args = SimpleNamespace(get_warmup_steps=lambda n: 122)
    gate.on_train_begin(args, SimpleNamespace(max_steps=4059), None)
    with pytest.raises(ValueError):
        gate.on_train_begin(args, SimpleNamespace(max_steps=4056), None)
    with pytest.raises(ValueError):
        gate.on_train_begin(
            SimpleNamespace(get_warmup_steps=lambda n: 121),
            SimpleNamespace(max_steps=4059),
            None,
        )


def test_epoch_policy_retains_all_snapshots_and_uses_four_workers():
    policy = campaign().checkpoint_policy(
        {
            "save_strategy": "epoch",
            "save_total_limit": None,
            "dataloader_num_workers": 4,
        },
        False,
        500,
    )
    assert policy == {
        "save_strategy": "epoch",
        "save_total_limit": None,
        "save_steps": 500,
        "dataloader_num_workers": 4,
    }
    assert campaign().checkpoint_policy({}, True, 500)["save_strategy"] == "no"


def test_memory_gate_checks_every_rank_and_all_representatives(tmp_path):
    import json

    root = tmp_path
    (root / "representatives.json").write_text(
        json.dumps(
            [
                {"bucket": b, "episode_uid": b}
                for b in ("short", "median", "p95", "longest")
            ]
        )
    )
    for rank in range(4):
        (root / f"profile_rank{rank}.jsonl").write_text(
            "".join(
                json.dumps({"update": i, "peak_reserved_gib": 77.9}) + "\n"
                for i in (1, 2)
            )
        )
        (root / f"exposures_rank{rank}.jsonl").write_text(
            json.dumps({"episode_uid": ["short", "median", "p95", "longest"][rank]})
            + "\n"
        )
    assert campaign().check_memory_profile(root)["peak_reserved_gib"] == 77.9
    (root / "profile_rank3.jsonl").write_text(
        '{"update": 1, "peak_reserved_gib": 78.1}\n'
    )
    with pytest.raises(ValueError):
        campaign().check_memory_profile(root)


def test_sha256_verification_detects_corruption(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"abc")
    digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert campaign().sha256_file(path) == digest
    campaign().verify_sha256(path, digest)
    path.write_bytes(b"abd")
    with pytest.raises(ValueError):
        campaign().verify_sha256(path, digest)


def test_epoch_save_fixture_uses_production_episode_sampler():
    from scripts.vln import check_epoch_saving
    from qwen_vl.train.sampler import _get_train_sampler
    from qwen_vl.data.episode_dataset import EpisodeLengthSampler

    assert hasattr(
        check_epoch_saving, "build_dataset"
    ), "Epoch fixture lacks episode lengths"
    trainer = SimpleNamespace(
        args=SimpleNamespace(
            world_size=4,
            gradient_accumulation_steps=2,
            seed=429,
            group_by_modality_length=True,
        )
    )
    sampler = _get_train_sampler(trainer, check_epoch_saving.build_dataset())
    assert isinstance(sampler, EpisodeLengthSampler)
    assert sorted(sampler) == list(range(9))
