"""No-history removes action content, not observation or recurrent memory."""
from copy import deepcopy
from types import MethodType

import pytest
import torch

from qwen_vl.contracts import validate_config, validate_resume_contract
from qwen_vl.data.episode_serializer import EpisodeSerializer
from test_candidate_contract import config
from test_session import make_session


def no_history_config():
    cfg = config("none")
    cfg["observations"].update(
        append_action_tokens=False,
        serializer_version="vln_candidate_logits_no_action_history_v1",
    )
    return cfg


def bare_serializer():
    s = EpisodeSerializer.__new__(EpisodeSerializer)
    s.config = no_history_config()
    s.mode = "candidate_logits"
    s.eos, s.separator = 9, [8]
    s.candidate_token_ids = [32, 33, 34, 35]
    s.action_ids = [[100 + i, 9] for i in range(4)]
    s.template_hash = "test"
    return s


def test_no_history_contract_is_explicit_and_incompatible():
    cfg = no_history_config()
    assert validate_config(cfg) is cfg
    for key, value in [("append_action_tokens", True),
                       ("serializer_version", "vln_candidate_logits_v1")]:
        bad = deepcopy(cfg)
        bad["observations"][key] = value
        with pytest.raises(ValueError):
            validate_config(bad)
    old = bare_serializer()
    old.config = config()
    current = bare_serializer().metadata()
    assert current["serializer"] == "vln_candidate_logits_no_action_history_v1"
    assert current["feedback_token_ids"] == [[9, 8]] * 4
    with pytest.raises(ValueError, match="navigation"):
        validate_resume_contract({"navigation": old.metadata(), "data": {}}, current, {})


def test_no_history_feedback_is_class_independent_boundary_only():
    s = bare_serializer()
    assert [s.feedback_ids(i) for i in range(4)] == [[9, 8]] * 4
    for invalid in [-1, 4]:
        with pytest.raises(ValueError):
            s.feedback_ids(invalid)


def test_no_history_overlay_preserves_production_recipe():
    from qwen_vl.train.vln_runtime import resolve_config
    cfg = resolve_config(
        "configs/vln_r2r_v0_base.yaml", "configs/vln_joint_b_2epoch_bs8.yaml",
        None, "configs/vln_candidate_logits_no_history.yaml",
    )
    validate_config(cfg, world_size=4)
    assert cfg["memory"]["mode"] == "full_context"
    assert cfg["training"]["backbone_lr"] == 5e-6
    assert cfg["training"]["expected_total_steps"] == 7704
    assert cfg["observations"]["append_action_tokens"] is False


@pytest.mark.parametrize("memory", ["full_context", "window8"])
def test_no_history_stream_commits_same_inputs_for_different_actions(monkeypatch, memory):
    committed = []
    for cls in range(4):
        s = make_session(monkeypatch, [])
        serializer = bare_serializer()
        s.serializer.mode = serializer.mode
        s.serializer.candidate_token_ids = serializer.candidate_token_ids
        s.serializer.feedback_ids = MethodType(EpisodeSerializer.feedback_ids, serializer)
        s.cfg["memory"]["mode"] = memory
        s.prefix_length = 0
        s.model.action_logits = lambda h: torch.nn.functional.one_hot(torch.tensor(cls), 4).float()
        for step in range(10):
            result = s.observe("e", step, None)
            assert result["class_id"] == cls
            assert result["generated_tokens"] == 0
            assert result["feedback_token_ids"] == [9, 8]
            before = list(s.committed)
            assert s.observe("e", step, None) == result
            assert s.committed == before
        assert s.positions.logical_token_count == 50
        expected_steps = range(2, 10) if memory == "window8" else []
        assert list(s.resident) == [(i, 5) for i in expected_steps]
        committed.append(s.committed)
    assert committed == [[5, 6, 7, 9, 8] * 10] * 4
