"""Causal token/frame selection for the GDN-only S sweep (offline and serving).

Same visibility and admission rule as ``episodic.build_layer_plan``: before
group ``t`` the source ``t - R`` leaves the recent window and its 300 visual
rows become candidates; every selector sees only the resident old bank plus
those newly eligible rows, so a deleted row can never return and a future
group never influences an earlier plan.  Priorities are fixed at birth.

* token arms: exact running top-B (``RunningTopB``), priority descending then
  original token ID ascending;
* frame arms (S7/S8): running top-``B/300`` whole frames, priority descending
  then source group ascending; every retained source keeps all 300 original
  rows, rejected sources keep none (no pooling, merging or partial frames);
* C_RECENCY: priority = original token ID, i.e. the latest eligible rows;
* C_RANDOM: the legacy per-layer SHA-256 key (``random_hash_v1``).

Plans reuse the immutable ``LayerSelectionPlan`` so the attention path is unchanged.
"""
import numpy as np
import torch
from qwen_vl.research.episodic_contract import VISUAL_TOKENS_PER_GROUP
from qwen_vl.stream.episodic import (
    LayerSelectionPlan, RunningTopB, _readonly, admission_order, as_priorities, plan_digest,
    random_priorities)


def recency_priorities(token_ids):
    """C_RECENCY: exact float64 copies of original token IDs (newest first)."""
    return np.asarray(token_ids, dtype=np.int64).astype(np.float64)


def _frame_priority(value):
    priority = np.float64(value)
    if not np.isfinite(priority):
        raise ValueError('Nonfinite frame priority')
    return priority


class RunningTopFrames:
    """Exact running top-F frames for fixed scalar frame priorities.

    Admitting sources one at a time keeps the same set as a batch top-F over
    all eligible frames.  ``ids``/``sources``/``priorities`` expose the
    retained rows in original-ID order, like ``RunningTopB``.
    """

    def __init__(self, frames, tokens_per_frame=VISUAL_TOKENS_PER_GROUP):
        if frames < 1:
            raise ValueError('Frame selection needs at least one frame')
        self.frames, self.tokens_per_frame = frames, tokens_per_frame
        self.frame_sources = np.empty(0, dtype=np.int64)
        self.frame_starts = np.empty(0, dtype=np.int64)
        self.frame_priorities = np.empty(0, dtype=np.float64)
        self.last_margin = None
        self._expand()

    def _expand(self):
        width = self.tokens_per_frame
        self.ids = (self.frame_starts[:, None] + np.arange(width, dtype=np.int64)).reshape(-1)
        self.sources = np.repeat(self.frame_sources, width)
        self.priorities = np.repeat(self.frame_priorities, width)

    def admit(self, ids, priority, source):
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if ids.size != self.tokens_per_frame or not np.array_equal(
                ids, ids[0] + np.arange(self.tokens_per_frame)):
            raise ValueError('A frame candidate is exactly its contiguous visual rows')
        if self.ids.size and int(ids[0]) <= int(self.ids.max()):
            raise ValueError('Expired frame must follow every resident episodic row')
        if self.frame_sources.size and int(source) <= int(self.frame_sources.max()):
            raise ValueError('Frame sources must be admitted in chronological order')
        starts = np.append(self.frame_starts, ids[0])
        sources = np.append(self.frame_sources, np.int64(source))
        priorities = np.append(self.frame_priorities, _frame_priority(priority))
        order = admission_order(priorities, sources)       # priority desc, source asc
        self.last_margin = None
        if starts.size > self.frames:
            self.last_margin = float(priorities[order[self.frames - 1]] - priorities[order[self.frames]])
        keep = np.sort(order[:self.frames])               # chronological == original-ID order
        self.frame_starts, self.frame_sources = starts[keep], sources[keep]
        self.frame_priorities = priorities[keep]
        self._expand()
        return self.ids


def batch_reference_frames(frames, starts, priorities, upto):
    """Independent (non-incremental) top-F frame rows over sources 0..upto; test oracle."""
    if upto < 0:
        return np.empty(0, np.int64)
    candidates = sorted(range(upto + 1), key=lambda s: (-float(priorities[s]), s))[:frames]
    return np.sort(np.concatenate([np.arange(starts[s], starts[s] + VISUAL_TOKENS_PER_GROUP)
                                   for s in candidates]).astype(np.int64))


def make_running(spec):
    if spec.is_frame:
        return RunningTopFrames(spec.frame_budget)
    return RunningTopB(spec.budget_tokens, spec.score == 'C_RANDOM')


def token_priorities(spec, episode_uid, layer_idx, visual, scores=None):
    """Sortable host priorities for the token arms, [G, 300]."""
    if spec.score == 'C_RANDOM':
        if scores is not None:
            raise ValueError('Random control must not consume write scores')
        return random_priorities(spec.selector_seed, episode_uid, layer_idx,
                                 visual.reshape(-1)).reshape(visual.shape)
    if spec.score == 'C_RECENCY':
        if scores is not None:
            raise ValueError('Recency control must not consume write scores')
        return recency_priorities(visual)
    if scores is None or tuple(scores.shape) != tuple(visual.shape):
        raise ValueError(f'{spec.score} selection needs FP32 [groups, visual] scores')
    return as_priorities(scores)


def build_sweep_layer_plan(spec, episode_uid, layer_idx, layout, scores=None, frame_scores=None):
    """Plan for one attention layer, using only prefix-eligible sources per group.

    ``scores``: FP32 [G, 300] token priorities (token GDN arms).
    ``frame_scores``: float64 [G] fixed frame priorities (S7/S8).
    Rows of groups ``> t - R`` can never affect group ``t``.
    """
    groups, recent = layout.groups, spec.recent_groups
    if layer_idx not in spec.attention_layers:
        raise ValueError(f'Layer {layer_idx} is not an episodic attention layer')
    visual = layout.visual_matrix().numpy()
    eligible = max(0, groups - recent)
    if spec.is_frame:
        if scores is not None or frame_scores is None:
            raise ValueError('Frame selection consumes one priority per source frame')
        frame_scores = np.asarray(frame_scores)
        if frame_scores.dtype != np.float64 or frame_scores.shape != (groups,):
            raise ValueError('Frame priorities must be float64 [groups]')
        if not np.isfinite(frame_scores).all():
            raise ValueError('Nonfinite frame priority')
        priorities = frame_scores
    else:
        if frame_scores is not None:
            raise ValueError('Token selection does not consume frame priorities')
        if spec.needs_write_scores:
            if scores is None or tuple(scores.shape) != (groups, layout.visual_tokens):
                raise ValueError(f'{spec.score} selection needs FP32 [groups, visual] scores')
            priorities = as_priorities(scores)                     # validates every group
        else:
            priorities = token_priorities(spec, episode_uid, layer_idx, visual[:eligible], scores)
    running = make_running(spec)
    offsets, chunks, sources, margins = [0], [], [], []
    for group in range(groups):
        expired = group - recent
        margin = np.nan
        if expired >= 0:
            running.admit(visual[expired], priorities[expired], expired)
            margin = np.nan if running.last_margin is None else running.last_margin
        ids, src = running.ids, running.sources
        if ids.size and int(ids.max()) >= layout.group_spans[max(0, group - recent + 1)][0]:
            raise RuntimeError('Episodic row overlaps the recent window')
        chunks.append(ids.copy())
        sources.append(src.copy())
        margins.append(margin)
        offsets.append(offsets[-1] + ids.size)
    ids = np.concatenate(chunks) if chunks else np.empty(0, np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    return LayerSelectionPlan(
        layer_idx=layer_idx, recent_groups=recent, budget_tokens=spec.budget_tokens,
        group_offsets=_readonly(offsets), token_ids=_readonly(ids.astype(np.int64)),
        token_sources=_readonly(np.concatenate(sources) if sources else np.empty(0, np.int64)),
        cutoff_margins=_readonly(np.asarray(margins, dtype=np.float64)),
        source_digest=plan_digest(layer_idx, recent, spec.budget_tokens, offsets, ids))


def observation_segment_indices(layout):
    """Indices of each group's observation block inside ``layout.score_segments()``."""
    segments = layout.score_segments()
    lookup = {segment: index for index, segment in enumerate(segments)}
    result = []
    for (start, _), observed in zip(layout.group_spans, layout.observation_ends):
        if (start, observed) not in lookup:
            raise ValueError('Observation block is not a score segment')
        result.append(lookup[(start, observed)])
    return tuple(result)


def frame_priorities_from_tokens(visual_scores):
    """S7 frame priorities from FP32 [G, 300] S0 token scores (float64 [G])."""
    from qwen_vl.models.gdn_features import frame_mean_priority
    array = np.asarray(visual_scores)
    return np.asarray([frame_mean_priority(row) for row in array], dtype=np.float64)


@torch.no_grad()
def offline_sweep_plan(spec, episode_uid, attention_layer, decoder_layer, hidden, layout,
                       diagnostics=None):
    """Score one attention layer's sources from its GDN input and build the immutable plan.

    ``hidden`` is the detached [T, width] input of the scoring GDN layer.  Runs
    outside the checkpointed decoder call, so recomputation can never alter a plan.
    """
    from qwen_vl.models.gdn_features import BOUNDARY_FIELDS, GdnSweepScorer, drift_priority
    gdn = spec.scoring_layer_for(attention_layer)
    scorer = GdnSweepScorer(spec, gdn, decoder_layer, hidden.device)
    observations = observation_segment_indices(layout) if spec.score in ('S8N', 'S8W') else ()
    tokens, stats = scorer.score_sequence(hidden, layout.score_segments(), observations)
    del scorer
    record = {} if diagnostics is None else diagnostics
    if spec.score in ('S8N', 'S8W'):
        frame = drift_priority(stats, spec.statistic, spec.floor(gdn)).cpu()
        frames = frame.numpy().astype(np.float64)
        plan = build_sweep_layer_plan(spec, episode_uid, attention_layer, layout, frame_scores=frames)
        if diagnostics is not None:
            record.setdefault('frame_scores', {})[attention_layer] = frames
            record.setdefault('drift_stats', {})[attention_layer] = dict(
                fields=BOUNDARY_FIELDS, values=stats.cpu().numpy(),
                floor_hits=int((stats[:, 2] < spec.floor(gdn)).sum()))
        return plan
    visual_scores = tokens[layout.visual_matrix().to(tokens.device)].cpu()      # [G, 300] FP32
    if spec.is_frame:                                                           # S7
        frames = frame_priorities_from_tokens(visual_scores.numpy())
        plan = build_sweep_layer_plan(spec, episode_uid, attention_layer, layout, frame_scores=frames)
        if diagnostics is not None:
            record.setdefault('frame_scores', {})[attention_layer] = frames
            record.setdefault('visual_scores', {})[attention_layer] = visual_scores
        return plan
    plan = build_sweep_layer_plan(spec, episode_uid, attention_layer, layout, visual_scores)
    if diagnostics is not None:
        record.setdefault('visual_scores', {})[attention_layer] = visual_scores
    return plan


class OfflineRankerFeatures:
    """Per-episode birth-time features and frozen-ranker plans for a learned arm (offline).

    ``observe`` runs ``GdnFeatureExtractor`` (the extractor used by the label
    collector and serving) over the detached input of one window GDN layer
    and keeps the encoded visual-row features; ``plan`` flattens the window
    in GDN-layer, head, channel order, scores 300-row frames with the frozen
    FP64 ranker and builds the same immutable plan as every token arm.
    """

    def __init__(self, spec, layout, device):
        self.spec, self.layout = spec, layout
        self.ranker = spec.ranker()
        self.rows = layout.visual_matrix().reshape(-1).to(device)
        self.segments = layout.score_segments()
        self.encoded = {}

    @torch.no_grad()
    def observe(self, gdn_layer, decoder_layer, hidden):
        from qwen_vl.models.gdn_features import RAW_FEATURE_SCHEMA, GdnFeatureExtractor, encode_features
        extractor = GdnFeatureExtractor(decoder_layer, hidden.device,
                                        with_state=self.spec.feature_schema == 'F1_v2')
        fields = extractor.sequence(hidden, self.segments)
        raw = {name: value[self.rows] for name, value in fields.items()}
        self.encoded[gdn_layer] = encode_features(dict(schema=RAW_FEATURE_SCHEMA, fields=raw),
                                                  self.spec.feature_schema)['values']

    @torch.no_grad()
    def plan(self, attention_layer, episode_uid, diagnostics=None):
        from qwen_vl.research.gdn_ranker import flatten_window
        window = self.spec.feature_layers_for(attention_layer)
        features = flatten_window(self.encoded, window)
        for gdn in window:
            del self.encoded[gdn]
        scores = self.ranker.score(attention_layer, features)
        visual_scores = scores.reshape(self.layout.groups, self.layout.visual_tokens).cpu()
        plan = build_sweep_layer_plan(self.spec, episode_uid, attention_layer, self.layout, visual_scores)
        if diagnostics is not None:
            diagnostics.setdefault('visual_scores', {})[attention_layer] = visual_scores
        return plan
