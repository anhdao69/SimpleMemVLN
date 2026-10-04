"""Token roles derived exclusively from immutable serializer spans."""
from enum import IntEnum
import torch


class LaneRole(IntEnum):
    PREFIX = 0
    OBSERVATION = 1
    FEEDBACK = 2
    STEP_END = 3


def build_lane_roles(prefix_length: int, observation_spans: tuple[tuple[int, int], ...],
                     complete_spans: tuple[tuple[int, int], ...], total_tokens: int) -> torch.Tensor:
    if type(prefix_length) is not int or type(total_tokens) is not int or not 0 <= prefix_length <= total_tokens:
        raise ValueError('Invalid prefix/total token count')
    if len(observation_spans) != len(complete_spans):
        raise ValueError('Observation and complete spans must match')
    roles = torch.zeros((1, total_tokens), dtype=torch.uint8)
    cursor = prefix_length
    for observation, complete in zip(observation_spans, complete_spans):
        if len(observation) != 2 or len(complete) != 2 or any(type(i) is not int for i in (*observation, *complete)):
            raise ValueError('Spans must contain two integer offsets')
        start, observation_end = observation
        complete_start, end = complete
        if start != cursor or complete_start != start or not start < observation_end < end <= total_tokens:
            raise ValueError('Spans must be contiguous, untruncated and have a post-decision closing block')
        # Decision is observation_end - 1. A nonempty closing block proves writer > decision.
        roles[0, start:observation_end] = LaneRole.OBSERVATION
        roles[0, observation_end:end] = LaneRole.FEEDBACK
        roles[0, end - 1] = LaneRole.STEP_END
        cursor = end
    if cursor != total_tokens:
        raise ValueError('Spans must cover every nonprefix token')
    return roles
