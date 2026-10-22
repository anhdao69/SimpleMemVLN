"""Session-owned episodic metadata and atomic full-attention KV promotion.

Physical full-attention cache rows are kept as
``[prefix | episodic rows sorted by original ID | contiguous recent groups]``.
Keys already carry their native rotary transform; rows are only selected,
never re-encoded or re-positioned.  No full-history K/V backup exists: an
expired row is either admitted (it is still resident when its group expires)
or discarded forever.  Native GDN/conv states are untouched.
"""
from collections import deque
from dataclasses import dataclass, field
import hashlib
import numpy as np
import torch
from qwen_vl.models.committed_write import CommittedWriteScorer
from qwen_vl.stream.episodic import RunningTopB, as_priorities, random_priorities


@dataclass
class _Group:
    step_id: int
    start: int
    end: int | None = None
    visual_start: int | None = None
    scores: dict = field(default_factory=dict)       # attention layer -> FP32 [300]


class EpisodicSessionState:
    def __init__(self, spec, model, episode_uid, device):
        self.spec, self.episode_uid = spec, episode_uid
        layers = model.backbone.model.language_model.layers
        self.attention_layers = spec.attention_layers
        self.scorers = ({gdn: CommittedWriteScorer(layers[gdn], device) for gdn, _ in spec.scoring_pairs}
                        if spec.needs_write_scores else {})
        self.running = {layer: RunningTopB(spec.budget_tokens, spec.score == 'random_hash_v1')
                        for layer in self.attention_layers}
        self.resident = {layer: np.empty(0, np.int64) for layer in self.attention_layers}
        self.groups = deque()
        self.prefix_length = None
        self.segment_kind = None
        self.visual_offset = None
        self.promotions = []
        # MemFork (stream/retention.py): one frame-level decision per admission, shared by every
        # attention layer, replaces the per-layer running top-B. None keeps plain M5.
        self.retention = None
        self.bank = None

    def set_retention(self, control):
        if self.groups or self.promotions or self.bank is not None:
            raise RuntimeError('Retention is chosen before the first observation')
        if not self.spec.needs_write_scores or self.spec.budget_tokens % 300:
            raise ValueError('Frame retention needs committed-write scores and a whole-frame budget')
        from qwen_vl.stream.retention import FrameBank
        self.retention, self.bank = control, FrameBank(self.spec.budget_tokens // 300)

    def episodic_ids(self, layer):
        return self.bank.ids if self.bank is not None else self.running[layer].ids

    # ---- score routing (called from the project forward) ---------------
    def score_append(self, gdn_layer, attention_layer, decoder_layer, hidden):
        scores = self.scorers[gdn_layer].append(hidden, self.segment_kind)
        if self.segment_kind == 'observation':
            group = self.groups[-1]
            if attention_layer in group.scores:
                raise RuntimeError('Source scores are written exactly once')
            visual = scores[self.visual_offset:self.visual_offset + 300]
            if visual.numel() != 300 or not torch.isfinite(visual).all():
                raise ValueError('Invalid observation committed-write scores')
            group.scores[attention_layer] = visual.cpu().numpy().astype(np.float32)

    def begin_append(self, kind, logical_start, length, visual_offset=None):
        if kind not in ('prefix', 'observation', 'response_token', 'response_close'):
            raise ValueError(kind)
        self.segment_kind, self.visual_offset = kind, visual_offset
        self._pending = (logical_start, length)

    def end_append(self, cache):
        start, length = self._pending
        for layer in self.attention_layers:
            self.resident[layer] = np.concatenate(
                (self.resident[layer], np.arange(start, start + length, dtype=np.int64)))
        if self.segment_kind == 'prefix':
            self.prefix_length = start + length
        self.segment_kind = self.visual_offset = None
        self.assert_invariants(cache)

    def open_group(self, step_id, logical_start, visual_start):
        self.groups.append(_Group(step_id, logical_start, visual_start=visual_start))

    def close_group(self, logical_end):
        group = self.groups[-1]
        if group.end is not None:
            raise RuntimeError('Group closed twice')
        group.end = logical_end
        if self.spec.needs_write_scores and set(group.scores) != set(self.attention_layers):
            raise RuntimeError('Observation scores missing for some attention layer')

    # ---- promotion at a logical group boundary --------------------------
    def promote(self, cache, step_id, generator=None):
        """Admit/evict every group older than the recent window, atomically.

        Idempotent per step. With MemFork retention, ``generator`` (the episode's sampling
        generator) drives the stochastic memory decisions; greedy retention ignores it.
        """
        recent = self.spec.recent_groups
        rebuilt, expired = {}, False
        while self.groups and self.groups[0].step_id <= step_id - recent:
            expired = True
            group = self.groups[0]
            if group.end is None:
                raise RuntimeError('Cannot promote an unclosed group')
            visual = np.arange(group.visual_start, group.visual_start + 300, dtype=np.int64)
            if self.bank is not None:
                from qwen_vl.stream.retention import frame_features
                self.bank.admit(group.step_id, group.visual_start,
                                frame_features(group.scores, self.attention_layers), step_id,
                                self.retention, generator)
                self.groups.popleft()
                continue
            for layer in self.attention_layers:
                if self.spec.budget_tokens:
                    if self.spec.score == 'random_hash_v1':
                        priorities = random_priorities(self.spec.selector_seed, self.episode_uid,
                                                       layer, visual)
                    else:
                        priorities = as_priorities(group.scores[layer])
                    self.running[layer].admit(visual, priorities, group.step_id)
            self.groups.popleft()
        if self.bank is not None and expired:
            self.bank.record_bank(step_id)
        if not expired:
            return
        recent_ids = (np.arange(self.groups[0].start, self.groups[-1].end, dtype=np.int64)
                      if self.groups else np.empty(0, np.int64))
        prefix = np.arange(self.prefix_length, dtype=np.int64)
        for layer in self.attention_layers:
            keep = np.concatenate((prefix, self.episodic_ids(layer), recent_ids))
            resident = self.resident[layer]
            if keep.size == resident.size and np.array_equal(keep, resident):
                continue
            rows = np.searchsorted(resident, keep)
            if rows.max(initial=-1) >= resident.size or not np.array_equal(resident[rows], keep):
                raise RuntimeError('Promotion needs a row that is no longer resident')
            rebuilt[layer] = (torch.from_numpy(rows), keep)
        for layer, (rows, keep) in rebuilt.items():
            entry = cache.layers[layer]
            index = rows.to(entry.keys.device)
            entry.keys = entry.keys.index_select(-2, index)
            entry.values = entry.values.index_select(-2, index)
            self.resident[layer] = keep
        if rebuilt:
            digest = hashlib.sha256()
            for layer in self.attention_layers:
                digest.update(self.episodic_ids(layer).tobytes())
            self.promotions.append(dict(step_id=step_id, digest=digest.hexdigest(),
                                        episodic_rows={l: int(self.episodic_ids(l).size)
                                                       for l in self.attention_layers}))
        self.assert_invariants(cache)

    def assert_invariants(self, cache):
        counts = set()
        for layer in self.attention_layers:
            ids = self.resident[layer]
            entry = cache.layers[layer]
            rows = 0 if getattr(entry, 'keys', None) is None else entry.keys.shape[-2]
            if rows != ids.size:
                raise RuntimeError(f'Layer {layer}: {rows} KV rows but {ids.size} recorded IDs')
            if ids.size and not np.all(np.diff(ids) > 0):
                raise RuntimeError('Resident IDs must be unique and in original order')
            if self.prefix_length is not None:
                if not np.array_equal(ids[:self.prefix_length], np.arange(self.prefix_length)):
                    raise RuntimeError('Prefix rows lost')
                old = self.episodic_ids(layer)
                if old.size:
                    first_recent = self.groups[0].start if self.groups else np.inf
                    if old.max() >= first_recent or old.min() < self.prefix_length:
                        raise RuntimeError('Episodic rows overlap prefix or recent window')
            counts.add(ids.size)
        if len(counts) > 1:
            raise RuntimeError(f'Inconsistent resident row counts {counts}')

    def summary(self):
        if self.bank is not None:
            return {layer: dict(episodic_rows=int(self.bank.ids.size), source_groups=len(self.bank.sources),
                                resident_rows=int(self.resident[layer].size), retained=list(self.bank.sources))
                    for layer in self.attention_layers}
        return {layer: dict(episodic_rows=int(self.running[layer].ids.size),
                            source_groups=int(np.unique(self.running[layer].sources).size),
                            resident_rows=int(self.resident[layer].size))
                for layer in self.attention_layers}
