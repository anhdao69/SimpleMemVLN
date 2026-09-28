"""The real request/decoder controller with a deterministic model boundary.

The test double only supplies token predictions; observe() must commit EOS,
separator, and body tokens itself and enforce retry/failure behavior.
"""
from types import SimpleNamespace
from collections import deque
import torch
import pytest
from qwen_vl.stream.session import StreamSession
from qwen_vl.stream.positions import PositionLedger


def make_session(monkeypatch, predictions):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    predictions = iter(predictions)

    class Tokenizer:
        all_special_ids = [9, 10]

        def decode(self, tokens, **kwargs):
            return "".join({1: "TURN_", 2: "LEFT", 3: "bad"}[t] for t in tokens)

    def head(hidden):
        logits = torch.zeros(11)
        logits[next(predictions)] = 1
        return logits

    serializer = SimpleNamespace(
        mode="qwen_text",
        eos=9,
        separator=[8],
        tokenizer=Tokenizer(),
        encode_observation=lambda rgb, step: {"input_ids": torch.tensor([[5, 6, 7]])},
        text_block=lambda ids: {"input_ids": torch.tensor([ids])},
    )
    model = SimpleNamespace(
        navigation_config={
            "memory": {"mode": "full_context"},
            "runtime": {"max_logical_context_tokens": 1000},
        },
        eval=lambda: None,
        backbone=SimpleNamespace(lm_head=head),
    )
    # Exercise the real controller; replace only expensive native appends.
    session = StreamSession.__new__(StreamSession)
    session.model, session.serializer, session.cfg = (
        model,
        serializer,
        model.navigation_config,
    )
    session.valid = True
    session.busy = False
    session.episode_uid = "e"
    session.next_step = 0
    session.last_result = None
    session.positions = PositionLedger()
    session.resident = deque()
    session.cache = SimpleNamespace(layers=[])
    session.committed = []

    def append(block):
        ids = block["input_ids"].flatten().tolist()
        session.committed.extend(ids)
        session.positions.logical_token_count += len(ids)
        return torch.zeros(1, len(ids), 4)

    session._append = append
    return session


def test_decoder_commits_body_eos_separator_once_and_retry_is_free(monkeypatch):
    session = make_session(monkeypatch, [1, 2, 9])
    result = session.observe("e", 0, None)
    assert result["action_name"] == "TURN_LEFT" and result["habitat_action_id"] == 2
    assert session.committed == [5, 6, 7, 1, 2, 9, 8]
    assert session.observe("e", 0, None) == result
    assert session.committed == [5, 6, 7, 1, 2, 9, 8]
    with pytest.raises(ValueError):
        session.observe("e", 2, None)


def test_invalid_response_poisoned_session_cannot_continue(monkeypatch):
    session = make_session(monkeypatch, [3, 9])
    with pytest.raises(ValueError):
        session.observe("e", 0, None)
    assert session.committed == [5, 6, 7, 3, 9, 8]
    assert not session.valid
    with pytest.raises(ValueError):
        session.observe("e", 1, None)


def test_decode_cap_and_structural_token_fail_without_stop_fallback(monkeypatch):
    session = make_session(monkeypatch, [1] * 16)
    with pytest.raises(ValueError, match="Response limit"):
        session.observe("e", 0, None)
    assert not session.valid and session.last_result is None
    assert session.committed == [5, 6, 7] + [1] * 16
    session = make_session(monkeypatch, [10])
    with pytest.raises(ValueError, match="Forbidden structural token"):
        session.observe("e", 0, None)
    assert not session.valid and session.committed == [5, 6, 7]
