"""Original-token episodic layout: metadata only, never new tokens or targets."""
from dataclasses import dataclass
import torch
from qwen_vl.research.episodic_contract import VISUAL_TOKENS_PER_GROUP


@dataclass(frozen=True)
class EpisodicLayout:
    """Immutable token geometry of one complete serialized episode.

    ``group_spans[t]`` covers the observation framing, the 300 expanded image
    tokens, the canonical action text, EOS and the fixed separator of group t.
    ``observation_ends[t]`` is the first response token of group t.  Visual
    token IDs of group t are ``visual_starts[t] + 0..299``.
    """
    prefix: int
    group_spans: tuple[tuple[int, int], ...]
    observation_ends: tuple[int, ...]
    visual_starts: tuple[int, ...]
    visual_tokens: int = VISUAL_TOKENS_PER_GROUP

    @property
    def groups(self):
        return len(self.group_spans)

    @property
    def length(self):
        return self.group_spans[-1][1] if self.group_spans else self.prefix

    def visual_ids(self, group):
        start = self.visual_starts[group]
        return torch.arange(start, start + self.visual_tokens, dtype=torch.long)

    def visual_matrix(self):
        """[G, 300] original visual token IDs, CPU int64."""
        starts = torch.tensor(self.visual_starts, dtype=torch.long)
        return starts[:, None] + torch.arange(self.visual_tokens, dtype=torch.long)

    def score_segments(self):
        """Canonical score-helper segments: prefix, then (observation, response) per group."""
        segments = [(0, self.prefix)] if self.prefix else []
        for (start, end), observed in zip(self.group_spans, self.observation_ends):
            segments.append((start, observed))
            segments.append((observed, end))
        return tuple(segments)

    def to_dict(self):
        return dict(prefix=self.prefix, group_spans=[list(s) for s in self.group_spans],
                    observation_ends=list(self.observation_ends),
                    visual_starts=list(self.visual_starts))


def build_episodic_layout(input_ids, image_token_id, prefix, observation_spans, group_spans):
    """Derive and validate layout from actual serialized IDs (fail closed)."""
    ids = input_ids.reshape(-1)
    if prefix <= 0 or len(observation_spans) != len(group_spans) or not group_spans:
        raise ValueError('Episodic layout requires a prefix and complete groups')
    if bool(ids[:prefix].eq(image_token_id).any()):
        raise ValueError('Prefix cannot contain image tokens')
    cursor, ends, starts = prefix, [], []
    for (obs_start, obs_end), (start, end) in zip(observation_spans, group_spans):
        if start != cursor or obs_start != start or not start < obs_end < end:
            raise ValueError('Groups must be contiguous with observation before response')
        image = ids[start:obs_end].eq(image_token_id).nonzero().reshape(-1)
        if image.numel() != VISUAL_TOKENS_PER_GROUP:
            raise ValueError(f'Group at {start} has {image.numel()} image tokens, expected 300')
        first = start + int(image[0])
        if int(image[-1]) - int(image[0]) != VISUAL_TOKENS_PER_GROUP - 1:
            raise ValueError('Expanded image tokens must be contiguous')
        if bool(ids[obs_end:end].eq(image_token_id).any()):
            raise ValueError('Response cannot contain image tokens')
        starts.append(first)
        ends.append(obs_end)
        cursor = end
    if cursor != ids.numel():
        raise ValueError('Layout must cover the complete episode exactly')
    return EpisodicLayout(prefix=int(prefix), group_spans=tuple((int(a), int(b)) for a, b in group_spans),
                          observation_ends=tuple(int(x) for x in ends),
                          visual_starts=tuple(int(x) for x in starts))
