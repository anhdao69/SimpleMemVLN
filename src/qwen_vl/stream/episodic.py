"""Deterministic episodic admission shared by offline training and serving.

Ordering is priority descending, then original logical token ID ascending.
Priorities are fixed when a source is written: a committed-write score from
the GDN layer immediately below the consuming attention layer, or a per-layer
SHA-256 random key for the matched control.  A source becomes a candidate
exactly when its group leaves the recent window (group ``t - R`` before step
``t``); future groups can never influence an earlier plan.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
import torch
_UINT64_MAX = np.uint64(np.iinfo(np.uint64).max)
RANDOM_KEY_VERSION = 'random_hash_v1'


def random_priorities(seed, episode_uid, layer_idx, token_ids):
    """64-bit big-endian prefix of SHA-256(version|seed|episode|layer|token).

    Independent of packing, ranks and process hash randomization.  Episode UID
    is used only by this control, never as a policy-network input.
    """
    base = hashlib.sha256(f'{RANDOM_KEY_VERSION}|{int(seed)}|{episode_uid}|{int(layer_idx)}|'.encode())
    ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    out = np.empty(ids.shape[0], dtype=np.uint64)
    for index, token in enumerate(ids.tolist()):
        digest = base.copy()
        digest.update(str(token).encode())
        out[index] = int.from_bytes(digest.digest()[:8], 'big')
    return out


def _sort_key(priorities):
    if priorities.dtype == np.uint64:
        return _UINT64_MAX - priorities          # ascending key == descending priority
    if priorities.dtype != np.float64:
        raise TypeError('Score priorities must be exact float64 copies of FP32 scores')
    return -priorities


def admission_order(priorities, ids):
    """Indices sorted by (priority desc, original token ID asc)."""
    return np.lexsort((ids, _sort_key(priorities)))


def as_priorities(scores):
    """Validate scores and convert exactly to sortable host priorities."""
    if isinstance(scores, np.ndarray) and scores.dtype == np.uint64:
        return scores
    if torch.is_tensor(scores):
        if scores.dtype != torch.float32:
            raise TypeError('Committed-write scores must be FP32')
        scores = scores.detach().cpu().numpy()
    scores = np.asarray(scores)
    if scores.dtype != np.float32:
        raise TypeError('Committed-write scores must be FP32')
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite episodic priority')
    return scores.astype(np.float64)


class RunningTopB:
    """Exact running top-B for fixed priorities and no changing constraints.

    Admitting groups one at a time keeps the same set as a batch top-B over all
    eligible rows; reference tests check this equivalence for every prefix.
    """

    def __init__(self, budget, random_keys):
        if budget < 0:
            raise ValueError('Negative episodic budget')
        self.budget = budget
        dtype = np.uint64 if random_keys else np.float64
        self.ids = np.empty(0, dtype=np.int64)
        self.priorities = np.empty(0, dtype=dtype)
        self.sources = np.empty(0, dtype=np.int64)
        self.last_margin = None

    def admit(self, ids, priorities, source):
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if priorities.dtype != self.priorities.dtype or priorities.shape != ids.shape:
            raise ValueError('Candidate priorities/IDs mismatch')
        if self.ids.size and ids.size and int(ids.min()) <= int(self.ids.max()):
            raise ValueError('Expired group must follow every resident episodic row')
        if np.unique(ids).size != ids.size:
            raise ValueError('Duplicate episodic candidate IDs')
        all_ids = np.concatenate((self.ids, ids))
        all_priorities = np.concatenate((self.priorities, priorities))
        all_sources = np.concatenate((self.sources, np.full(ids.size, source, np.int64)))
        order = admission_order(all_priorities, all_ids)
        keep = order[:self.budget]
        self.last_margin = None
        if all_ids.size > self.budget > 0 and self.priorities.dtype == np.float64:
            self.last_margin = float(all_priorities[order[self.budget - 1]]
                                     - all_priorities[order[self.budget]])
        keep = keep[np.argsort(all_ids[keep], kind='stable')]   # serve by original ID
        self.ids, self.priorities, self.sources = all_ids[keep], all_priorities[keep], all_sources[keep]
        return self.ids

    def removed_from(self, before):
        return np.setdiff1d(before, self.ids, assume_unique=True)


def _readonly(array):
    array = np.ascontiguousarray(array)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class LayerSelectionPlan:
    """Immutable per-layer episodic visibility for every group of one episode.

    ``token_ids[group_offsets[t]:group_offsets[t+1]]`` are the old visual rows
    visible to every query of group t, sorted by original token ID.  The same
    object is captured by the forward and by checkpoint recomputation; the
    digest is re-verified before any backward use.
    """
    layer_idx: int
    recent_groups: int
    budget_tokens: int
    group_offsets: np.ndarray
    token_ids: np.ndarray
    token_sources: np.ndarray
    cutoff_margins: np.ndarray
    source_digest: str

    def group_ids(self, group):
        return self.token_ids[self.group_offsets[group]:self.group_offsets[group + 1]]

    @property
    def groups(self):
        return len(self.group_offsets) - 1

    def verify(self):
        if plan_digest(self.layer_idx, self.recent_groups, self.budget_tokens,
                       self.group_offsets, self.token_ids) != self.source_digest:
            raise RuntimeError(f'Episodic plan for layer {self.layer_idx} changed after capture')
        return self


def plan_digest(layer_idx, recent, budget, offsets, ids):
    digest = hashlib.sha256(f'{layer_idx}|{recent}|{budget}|'.encode())
    digest.update(np.ascontiguousarray(offsets, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(ids, dtype=np.int64).tobytes())
    return digest.hexdigest()


def build_layer_plan(spec, episode_uid, layer_idx, layout, scores=None):
    """Plan for one attention layer, using only prefix-eligible sources per group.

    ``scores``: FP32 [G, 300] committed-write scores for the visual rows of
    every group (only rows of groups <= t-R can affect group t).  Random
    controls derive priorities from the hash; B0 needs neither.
    """
    groups, recent, budget = layout.groups, spec.recent_groups, spec.budget_tokens
    if layer_idx not in spec.attention_layers:
        raise ValueError(f'Layer {layer_idx} is not an episodic attention layer')
    visual = layout.visual_matrix().numpy()
    eligible_groups = max(0, groups - recent)
    if budget == 0 or spec.score == 'none':
        if scores is not None:
            raise ValueError('B0 does not consume scores')
        priorities = None
    elif spec.score == 'random_hash_v1':
        if scores is not None:
            raise ValueError('Random control must not consume write scores')
        priorities = random_priorities(spec.selector_seed, episode_uid, layer_idx,
                                       visual[:eligible_groups].reshape(-1)).reshape(
                                           eligible_groups, layout.visual_tokens)
    elif spec.score == 'committed_mean_l2_v1':
        if scores is None or tuple(scores.shape) != (groups, layout.visual_tokens):
            raise ValueError('Committed selection needs FP32 [groups, visual] scores')
        priorities = as_priorities(scores)
    else:
        raise ValueError(f'Unsupported score {spec.score}')
    offsets, chunks, sources, margins = [0], [], [], []
    running = RunningTopB(budget, spec.score == 'random_hash_v1') if priorities is not None else None
    for group in range(groups):
        expired = group - recent
        margin = np.nan
        if running is not None and expired >= 0:
            running.admit(visual[expired], priorities[expired], expired)
            margin = np.nan if running.last_margin is None else running.last_margin
        ids = running.ids if running is not None else np.empty(0, np.int64)
        src = running.sources if running is not None else np.empty(0, np.int64)
        if ids.size and int(ids.max()) >= layout.group_spans[max(0, group - recent + 1)][0]:
            raise RuntimeError('Episodic row overlaps the recent window')
        chunks.append(ids.copy())
        sources.append(src.copy())
        margins.append(margin)
        offsets.append(offsets[-1] + ids.size)
    ids = np.concatenate(chunks) if chunks else np.empty(0, np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    return LayerSelectionPlan(
        layer_idx=layer_idx, recent_groups=recent, budget_tokens=budget,
        group_offsets=_readonly(offsets), token_ids=_readonly(ids.astype(np.int64)),
        token_sources=_readonly(np.concatenate(sources) if sources else np.empty(0, np.int64)),
        cutoff_margins=_readonly(np.asarray(margins, dtype=np.float64)),
        source_digest=plan_digest(layer_idx, recent, budget, offsets, ids))


def batch_reference_selection(budget, ids_by_group, priorities_by_group, upto_group):
    """Independent (non-incremental) top-B over groups 0..upto_group; test reference."""
    if upto_group < 0:
        return np.empty(0, np.int64)
    ids = np.concatenate([np.asarray(x, np.int64) for x in ids_by_group[:upto_group + 1]])
    prios = np.concatenate(priorities_by_group[:upto_group + 1])
    keep = admission_order(prios, ids)[:budget]
    return np.sort(ids[keep])


def plan_summary(plan, layout, early_steps=3):
    """Bounded diagnostics: occupancy, age, early-frame concentration, turnover, margins."""
    final = plan.group_ids(plan.groups - 1)
    sources = plan.token_sources[plan.group_offsets[-2]:plan.group_offsets[-1]]
    turnover = 0
    for t in range(1, plan.groups):
        before, after = plan.group_ids(t - 1), plan.group_ids(t)
        turnover += int(np.setdiff1d(before, after, assume_unique=True).size)
    margins = plan.cutoff_margins[np.isfinite(plan.cutoff_margins)]
    age = (plan.groups - 1 - sources) if sources.size else np.empty(0)
    return dict(
        groups=plan.groups, final_rows=int(final.size),
        distinct_source_groups=int(np.unique(sources).size),
        fraction_from_steps_0_2=float(np.mean(sources < early_steps)) if sources.size else None,
        fraction_first_frame=float(np.mean(sources == 0)) if sources.size else None,
        mean_age_groups=float(age.mean()) if age.size else None,
        evicted_total=turnover,
        margin_quantiles=[float(x) for x in np.quantile(margins, [0, .1, .5])] if margins.size else None,
        digest=plan.source_digest[:16])
