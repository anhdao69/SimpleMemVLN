"""Transaction controller tests with real frozen observation/decode/episodic logic.

Only expensive model tensor appends and tokenizer text are replaced. Native,
conv, scorer, pending, rope, selector, and group state remain observable.
"""
from collections import deque
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from qwen_vl.stream.gdn_selectors import RunningTopFrames
from qwen_vl.stream.gdn_sweep_state import GdnSweepSessionState
from qwen_vl.stream.positions import PositionLedger
from qwen_vl.stream.session import StreamSession

from s8n_dagger.policy import RecoverySession


class Decoder(torch.nn.Module):
    def __deepcopy__(self, memo):
        raise AssertionError("Snapshot copied a decoder and its weights")


class TinyModel(torch.nn.Module):
    def __init__(self, predictions):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.decoder = Decoder()
        self.navigation_config = {
            "memory": {"mode": "episodic_window_gdn_sweep_v1"},
            "runtime": {"max_logical_context_tokens": 100000},
        }
        self.predictions = iter(predictions)
        self.decode_calls = 0
        self.backbone = SimpleNamespace(model=SimpleNamespace(rope_deltas=torch.zeros(1)), lm_head=self.head)

    def head(self, hidden):
        self.decode_calls += 1
        logits = torch.zeros(20)
        logits[next(self.predictions)] = 1
        return logits


class Tokenizer:
    all_special_ids = [9, 10]

    def decode(self, tokens, **kwargs):
        return "".join({1: "TURN_", 2: "LEFT", 3: "RIGHT", 4: " ", 5: "STOP", 6: "bad", 7: "MOVE_FORWARD"}[t] for t in tokens)


def make_session(monkeypatch, predictions, cls=RecoverySession):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    model = TinyModel(predictions)
    serializer = SimpleNamespace(
        mode="qwen_text", eos=9, separator=[8], image_id=11,
        action_ids=[[7, 9], [1, 2, 9], [1, 3, 9], [5, 9]], tokenizer=Tokenizer(),
        encode_observation=lambda rgb, step: {"input_ids": torch.tensor([[12] + [11] * 300 + [13]])},
        text_block=lambda ids: {"input_ids": torch.tensor([ids])},
    )
    session = cls(model, serializer)
    session.valid, session.busy, session.episode_uid = True, False, "e"
    session.next_step, session.last_result = 0, None
    session.positions, session.resident = PositionLedger(), deque()
    session.step_lane_cache = None
    session.prefix_length = 0
    session.cache = SimpleNamespace(layers=[SimpleNamespace(
        keys=torch.empty(1, 1, 0, 1), values=torch.empty(1, 1, 0, 1),
        recurrent_states=torch.zeros(1), conv_states=torch.zeros(3),
    )])
    ep = GdnSweepSessionState.__new__(GdnSweepSessionState)
    ep.spec = SimpleNamespace(recent_groups=4, needs_write_scores=True, score="S8N", is_frame=True)
    ep.attention_layers = (0,)
    ep.scorers = {0: SimpleNamespace(layer=model.decoder, state=torch.zeros(1), conv_tail=torch.zeros(3), pending=[])}
    ep.running, ep.resident = {0: RunningTopFrames(4)}, {0: np.empty(0, np.int64)}
    ep.groups, ep.promotions = deque(), []
    ep.prefix_length, ep.segment_kind, ep.visual_offset = 0, None, None
    session.episodic = ep

    def append(block):
        ids = block["input_ids"].flatten().float()
        cache = session.cache.layers[0]
        extra = ids.reshape(1, 1, -1, 1)
        cache.keys = torch.cat([cache.keys, extra], -2)
        cache.values = torch.cat([cache.values, extra + 1], -2)
        for token in ids:
            cache.recurrent_states.mul_(1.001).add_(token)
            cache.conv_states = torch.cat([cache.conv_states[1:], token.reshape(1)])
        scorer = session.episodic.scorers[0]
        kind = session.episodic.segment_kind
        scorer.conv_tail = cache.conv_states.clone()
        if kind in ("response_token", "response_close"):
            scorer.pending.append(ids.clone())
            if kind == "response_close":
                scorer.state.add_(torch.cat(scorer.pending).sum())
                scorer.pending.clear()
        else:
            scorer.state.add_(ids.sum())
            session.episodic.groups[-1].scores[0] = np.float64(float(scorer.state[0]))
        session.positions.logical_token_count += len(ids)
        session.positions.mrope_cursor += len(ids) + 2
        session.model.backbone.model.rope_deltas.add_(len(ids))
        return torch.zeros(1, len(ids), 4)

    session._append = append
    return session


def rgb(value=0):
    return np.full((480, 640, 3), value, np.uint8)


def state(session):
    layer = session.cache.layers[0]
    ep = session.episodic
    scorer = ep.scorers[0]
    return dict(
        keys=layer.keys.clone(), values=layer.values.clone(),
        native=layer.recurrent_states.clone(), conv=layer.conv_states.clone(),
        scorer=scorer.state.clone(), scorer_conv=scorer.conv_tail.clone(),
        pending=[x.clone() for x in scorer.pending],
        positions=deepcopy(session.positions), resident=ep.resident[0].copy(),
        groups=[(x.step_id, x.start, x.end, x.visual_start, dict(x.scores)) for x in ep.groups],
        running=ep.running[0].ids.copy(), promotions=deepcopy(ep.promotions),
        rope=session.model.backbone.model.rope_deltas.clone(), next_step=session.next_step,
    )


def assert_state_equal(first, second):
    for key in first:
        a, b = first[key], second[key]
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), key
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b, err_msg=key)
        else:
            assert a == b, key


def test_unchanged_commit_preserves_noncanonical_generated_tokens_and_original_state(monkeypatch):
    # Re-encoding a learner response would erase its accepted leading space.
    adapter = make_session(monkeypatch, [4, 1, 2, 9] * 7)
    original = make_session(monkeypatch, [4, 1, 2, 9] * 7, cls=StreamSession)
    for step in range(7):
        expected = original.observe("e", step, rgb())
        proposal = adapter.propose("e", step, rgb())
        assert proposal["response_token_ids"] == [4, 1, 2, 9]
        result = adapter.commit("e", step, 2)
        assert result["response_token_ids"] == expected["response_token_ids"]
        assert result["habitat_action_id"] == 2
        assert_state_equal(state(adapter), state(original))
    assert adapter.episodic.promotions


def test_override_discards_proposed_gdn_conv_scorer_pending_and_rope(monkeypatch):
    adapter = make_session(monkeypatch, [4, 1, 2, 9] * 7)
    forced = make_session(monkeypatch, [])
    for step in range(7):
        adapter.propose("e", step, rgb(step))
        committed = adapter.commit("e", step, 3)
        baseline = forced.force("e", step, rgb(step), 3)
        assert committed["response_token_ids"] == baseline["response_token_ids"] == [1, 3, 9]
        assert committed["response_text"] == "TURN_RIGHT"
        assert_state_equal(state(adapter), state(forced))
        assert adapter.episodic.scorers[0].layer is adapter.model.decoder
    assert forced.model.decode_calls == 0
    assert adapter.episodic.promotions


def test_transaction_retries_are_free_and_conflicting_requests_rejected(monkeypatch):
    session = make_session(monkeypatch, [1, 2, 9] * 2)
    proposal = session.propose("e", 0, rgb())
    after = state(session)
    assert session.propose("e", 0, rgb()) == proposal
    with pytest.raises(ValueError):
        session.propose("e", 0, rgb(1))
    with pytest.raises(ValueError):
        session.propose("e", 1, rgb())
    with pytest.raises(ValueError):
        session.force("e", 0, rgb(), 2)
    with pytest.raises(ValueError):
        session.commit("other", 0, 2)
    with pytest.raises(ValueError):
        session.commit("e", 0, True)
    assert_state_equal(after, state(session))
    committed = session.commit("e", 0, 3)
    assert session.commit("e", 0, 3) == committed
    with pytest.raises(ValueError):
        session.commit("e", 0, 2)
    with pytest.raises(ValueError):
        session.force("e", 0, rgb(), 3)
    with pytest.raises(ValueError):
        session.commit("e", 1, 2)
    session.propose("e", 1, rgb())
    with pytest.raises(ValueError):
        session.propose("e", 0, rgb())


def test_force_idempotence_requires_same_rgb_action_and_operation(monkeypatch):
    session = make_session(monkeypatch, [])
    committed = session.force("e", 0, rgb(), 1)
    after = state(session)
    assert session.force("e", 0, rgb(), 1) == committed
    for operation in (
        lambda: session.force("e", 0, rgb(1), 1),
        lambda: session.force("e", 0, rgb(), 2),
        lambda: session.propose("e", 0, rgb()),
        lambda: session.commit("e", 0, 1),
    ):
        with pytest.raises(ValueError):
            operation()
    assert_state_equal(after, state(session))
    assert session.model.decode_calls == 0


@pytest.mark.parametrize("tokens,match", [([6, 9], "Invalid navigation"), ([10], "Forbidden"), ([1] * 16, "Response limit")])
def test_invalid_model_response_poisons_episode_without_stop(monkeypatch, tokens, match):
    session = make_session(monkeypatch, tokens)
    with pytest.raises(ValueError, match=match):
        session.propose("e", 0, rgb())
    assert not session.valid
    with pytest.raises(ValueError):
        session.commit("e", 0, 0)
    with pytest.raises(ValueError):
        session.force("e", 0, rgb(), 0)


def test_reset_discards_pending_and_failed_episode_state(monkeypatch):
    session = make_session(monkeypatch, [1, 2, 9])
    session.propose("e", 0, rgb())

    def reset_boundary(self, uid, instruction):
        self.episode_uid, self.valid, self.busy = uid, True, False
        self.next_step, self.last_result = 0, None

    monkeypatch.setattr(StreamSession, "reset", reset_boundary)
    session.valid = False
    assert session.reset("new", "go") == {"status": "ok", "episode_uid": "new"}
    with pytest.raises(ValueError):
        session.commit("new", 0, 2)
    assert session.valid


def test_json_lines_protocol_decodes_lossless_rgb_and_commits_only_executed_action(monkeypatch):
    import io
    import json
    from qwen_vl.stream.transport import encode_rgb
    from s8n_dagger.model_server import serve

    session = make_session(monkeypatch, [1, 2, 9])
    requests = [
        {"operation": "propose", "episode_uid": "e", "step_id": 0, **encode_rgb(rgb(35))},
        {"operation": "commit", "episode_uid": "e", "step_id": 0, "habitat_action_id": 3},
        {"operation": "force", "episode_uid": "e", "step_id": 1, "habitat_action_id": 0, **encode_rgb(rgb(36))},
        {"operation": "observe", "episode_uid": "e", "step_id": 2},
    ]
    output = io.StringIO()
    serve(session, io.StringIO("\n".join(json.dumps(x) for x in requests) + "\n"), output)
    results = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [x["status"] for x in results] == ["ok", "ok", "ok", "failure"]
    assert [x["habitat_action_id"] for x in results[:3]] == [2, 3, 0]
    assert results[1]["response_token_ids"] == [1, 3, 9]
    assert results[2]["response_token_ids"] == [5, 9]
    assert session.next_step == 2


def test_protocol_reports_bad_json_without_corrupting_next_request(monkeypatch):
    import io
    import json
    from s8n_dagger.model_server import serve

    session = make_session(monkeypatch, [])
    output = io.StringIO()
    serve(session, io.StringIO('bad json\n{"operation": "commit", "episode_uid": "e", "step_id": 0, "habitat_action_id": 0}\n'), output)
    results = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [x["status"] for x in results] == ["failure", "failure"]
    assert session.valid and session.next_step == 0


def test_gpu_audit_fingerprints_detect_all_mutable_rollback_components(monkeypatch):
    from s8n_dagger.check_policy import state_fingerprints

    first = make_session(monkeypatch, [1, 2, 9])
    second = make_session(monkeypatch, [1, 2, 9])
    first.propose("e", 0, rgb())
    second.propose("e", 0, rgb())
    first.commit("e", 0, 2)
    second.commit("e", 0, 2)
    assert state_fingerprints(first) == state_fingerprints(second)
    before = state_fingerprints(first)
    with torch.inference_mode():
        first.cache.layers[0].conv_states.add_(1)
        first.episodic.scorers[0].pending.append(torch.tensor([3]))
        first.positions.mrope_cursor += 1
        first.model.backbone.model.rope_deltas.add_(1)
    after = state_fingerprints(first)
    assert {key for key in before if before[key] != after[key]} == {"cache", "episodic", "positions", "rope_deltas"}
