from types import SimpleNamespace
import pytest
import torch
from test_session import make_session
from qwen_vl.stream.session import StreamSession


@pytest.mark.parametrize("feedback", [[33, 9, 8], [1, 2, 9, 8]])
def test_candidate_feedback_no_decode_and_retry(monkeypatch, feedback):
    s = make_session(monkeypatch, [])
    s.serializer.mode = "candidate_logits"
    s.serializer.candidate_token_ids = [32, 33, 34, 35]
    s.serializer.feedback_ids = lambda cls: feedback
    s.model.action_logits = lambda h: torch.tensor([-2.0, 3.0, 0.0, 1.0])
    s.cfg["runtime"]["action_diagnostics"] = True
    result = s.observe("e", 0, None)
    assert result["class_id"] == 1 and result["action_name"] == "TURN_LEFT"
    assert result["habitat_action_id"] == 2 and result["generated_tokens"] == 0
    assert s.committed == [5, 6, 7] + feedback
    assert result["candidate_token_ids"] == [32, 33, 34, 35]
    assert result["margin_top1_top2"] == 2
    assert s.observe("e", 0, None) == result
    assert s.committed == [5, 6, 7] + feedback


def test_window_evicts_whole_feedback_group_without_reset(monkeypatch):
    s = make_session(monkeypatch, [])
    s.serializer.mode = "candidate_logits"
    s.serializer.candidate_token_ids = [32, 33, 34, 35]
    s.serializer.feedback_ids = lambda cls: [32, 9, 8]
    s.model.action_logits = lambda h: torch.tensor([3.0, 0.0, 0.0, 0.0])
    s.cfg["memory"]["mode"] = "window8"
    s.prefix_length = 10
    # Actual cache eviction on tensor KV; recurrent state must stay intact.
    recurrent = torch.randn(2, 3)
    kv = SimpleNamespace(keys=torch.zeros(1, 1, 10, 1), values=torch.zeros(1, 1, 10, 1))
    s.cache.layers = [kv, SimpleNamespace(recurrent_states=recurrent)]
    append = s._append

    def with_kv(block):
        n = block["input_ids"].numel()
        for field in ("keys", "values"):
            setattr(
                kv, field, torch.cat([getattr(kv, field), torch.ones(1, 1, n, 1)], -2)
            )
        return append(block)

    s._append = with_kv
    for step in range(10):
        result = s.observe("e", step, None)
    assert list(s.resident) == [(i, 6) for i in range(2, 10)]
    assert result["retained_kv_tokens"] == 10 + 8 * 6
    assert s.positions.logical_token_count == 60
    assert s.cache.layers[1].recurrent_states is recurrent


def test_candidate_reserves_feedback_before_any_append(monkeypatch):
    s = make_session(monkeypatch, [])
    s.serializer.mode = "candidate_logits"
    s.serializer.feedback_ids = lambda cls: [32, 9, 8]
    s.cfg["runtime"]["max_logical_context_tokens"] = 5
    with pytest.raises(ValueError, match="Complete step"):
        s.observe("e", 0, None)
    assert s.committed == [] and not s.valid
