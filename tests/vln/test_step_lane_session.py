import torch
from test_session import make_session


def test_text_append_writer_is_last_closing_token_and_retry_no_write(monkeypatch):
    s = make_session(monkeypatch, [1,2,9,1,2,9])
    s.model.step_lane_spec = object()
    original = s._append
    trace = []
    def append(block, *, step_lane_roles=None):
        trace.extend(step_lane_roles.flatten().tolist())
        return original(block)
    s._append = append
    s.observe('e',0,None)
    assert trace == [1,1,1,2,2,2,3]
    s.observe('e',0,None)
    assert trace.count(3) == 1
    s.observe('e',1,None)
    assert trace.count(3) == 2


def test_none_feedback_closure_contains_one_writer_no_action(monkeypatch):
    s = make_session(monkeypatch, [])
    s.model.step_lane_spec = object()
    s.model.action_logits = lambda h: torch.tensor([0.,1.,0.,0.])
    s.serializer.mode = 'candidate_logits'
    s.serializer.feedback_ids = lambda c: [9,8]
    original = s._append
    trace=[]
    def append(block, *, step_lane_roles=None):
        trace.extend(step_lane_roles.flatten().tolist())
        return original(block)
    s._append = append
    s.observe('e',0,None)
    assert s.committed == [5,6,7,9,8]
    assert trace == [1,1,1,2,3]
