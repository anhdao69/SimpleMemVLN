import pytest
import torch
from qwen_vl.data.lane_metadata import build_lane_roles


def test_unequal_feedback_and_span_based_roles():
    roles = build_lane_roles(2, ((2, 5), (8, 11)), ((2, 8), (8, 13)), 13)
    assert roles.dtype == torch.uint8
    assert roles.tolist() == [[0, 0, 1, 1, 1, 2, 2, 3, 1, 1, 1, 2, 3]]
    # Newline IDs are deliberately irrelevant: only the complete-group boundary writes.
    ids = torch.full((1, 13), 198)
    assert roles.eq(3).nonzero().tolist() == [[0, 7], [0, 12]]
    assert ids[0, 1] == ids[0, 7]


@pytest.mark.parametrize('prefix,obs,spans,total', [
    (2, ((3, 5),), ((3, 7),), 7),  # gap
    (2, ((2, 5), (6, 8)), ((2, 7), (6, 10)), 10),  # overlap
    (2, ((2, 5),), ((2, 7),), 8),  # trailing unassigned token
    (2, ((2, 7),), ((2, 7),), 7),  # no post-decision closing block
    (2, ((2, 5),), ((2, 8),), 7),  # truncated
    (2, (), ((2, 7),), 7),
    (-1, (), (), 0),
])
def test_invalid_span_layout_rejected(prefix, obs, spans, total):
    with pytest.raises(ValueError):
        build_lane_roles(prefix, obs, spans, total)


def test_prefix_only_and_one_step_terminal():
    assert build_lane_roles(3, (), (), 3).tolist() == [[0, 0, 0]]
    assert build_lane_roles(0, ((0, 2),), ((0, 3),), 3).tolist() == [[1, 1, 3]]
