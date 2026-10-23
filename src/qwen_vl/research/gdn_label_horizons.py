"""Offline near/tail query plans and support-weighted utility aggregates (``gdn_utility_near_tail_v2``).

Training-label metadata only: the remaining trajectory length, sampled query
offsets and future utilities never enter a serving feature.

For source group ``s`` first eligible before group ``a = s + R`` in an episode
with ``T`` recorded decisions (every group ends in one decision):

    M = max(T - a, 0),   N = min(16, M),   F = M - N

Near population offsets are ``[0, N)``, tail population offsets ``[N, M)``
(relative to admission).  ``min(4, N)`` near and ``min(4, F)`` tail offsets
are drawn uniformly without replacement *within* each stratum from local
deterministic RNG streams keyed by (protocol, label seed, canonical episode
key, source group, stratum); supports of size <= 4 are exhaustive.  No
utility, score, residency, patch, layer or global RNG state influences a draw.

    near_mean = mean over sampled near queries
    tail_mean = mean over sampled tail queries          (null when F = 0)
    remaining = (N / M) near_mean + (F / M) tail_mean    (raw values, before any log)

A missing requested query makes its stratum null (never zero, never a mean of
survivors); the remaining mean needs every nonempty stratum.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import numpy as np

PROTOCOL = 'gdn_utility_near_tail_v2'
TARGET_NEAR = 'shadow_fixed_query_loo_near16_mean_v2'
TARGET_TAIL = 'shadow_fixed_query_loo_tail_after16_mean_v2'
TARGET_REMAINING = 'shadow_fixed_query_loo_remaining_mean_v2'
TARGET_IDS = (TARGET_NEAR, TARGET_TAIL, TARGET_REMAINING)
NEAR_HORIZON, SAMPLES_PER_STRATUM, RECENT_GROUPS = 16, 4, 4
SAMPLED, IMPORTED, DENSE = 'sampled_v2', 'imported', 'dense_v2'
# Spatial sampling of label rows: two positions per 5x5 tile of the 15x20 grid.
GRID_ROWS, GRID_COLS, TILE, PER_TILE = 15, 20, 5, 2
SHADOW_CONTEXT = 'shadow_ctx_uniform_displace_v1'


def local_rng(*key):
    """A numpy Generator seeded only by SHA-256 of the key (no global RNG is touched)."""
    digest = hashlib.sha256('|'.join(str(part) for part in key).encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], 'big')))


@dataclass(frozen=True)
class SourceHorizonPlan:
    episode_key: str
    source_group: int
    admission_group: int
    episode_steps: int               # OFFLINE LABEL METADATA ONLY
    near_support: int
    tail_support: int
    near_offsets: tuple[int, ...]
    tail_offsets: tuple[int, ...]
    seed: int
    near_origin: str                 # sampled_v2 | imported | dense_v2
    tail_origin: str                 # sampled_v2 | dense_v2
    recent_groups: int = RECENT_GROUPS
    near_horizon: int = NEAR_HORIZON
    samples_per_stratum: int = SAMPLES_PER_STRATUM

    @property
    def remaining_support(self):
        return self.near_support + self.tail_support

    @property
    def offsets(self):
        return self.near_offsets + self.tail_offsets

    def query_steps(self):
        """Absolute decision indices of the requested queries (all ``< episode_steps``)."""
        return tuple(self.admission_group + u for u in self.offsets)

    def stratum(self, offset):
        if offset in self.near_offsets:
            return 'near'
        if offset in self.tail_offsets:
            return 'tail'
        raise KeyError(offset)


@dataclass(frozen=True)
class HorizonAggregate:
    near_mean: float | None
    tail_mean: float | None
    remaining_mean: float | None
    near_status: str                 # complete | missing | not_eligible
    tail_status: str                 # complete | missing | not_applicable
    remaining_status: str            # complete | missing | not_eligible


def _draw(rng, low, high, count):
    population = high - low
    if population <= count:
        return tuple(range(low, high))
    return tuple(sorted(int(x) for x in low + rng.choice(population, size=count, replace=False)))


def make_horizon_plan(episode_key, source_group, episode_steps, seed, recent_groups=RECENT_GROUPS,
                      near_horizon=NEAR_HORIZON, samples_per_stratum=SAMPLES_PER_STRATUM,
                      existing_near_offsets=None, dense=False):
    """Deterministic local near/tail plan for one source; never consumes utilities.

    ``dense`` evaluates every query of both strata (label-quality subset).
    """
    for name, value, low in (('source_group', source_group, 0), ('episode_steps', episode_steps, 1),
                             ('recent_groups', recent_groups, 1), ('near_horizon', near_horizon, 1),
                             ('samples_per_stratum', samples_per_stratum, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < low:
            raise ValueError(f'{name} must be an integer >= {low}')
    if source_group >= episode_steps:
        raise ValueError('Source group must be a recorded decision of the episode')
    admission = source_group + recent_groups
    support = max(episode_steps - admission, 0)
    near = min(near_horizon, support)
    tail = support - near
    if dense:
        if existing_near_offsets is not None:
            raise ValueError('Dense plans evaluate the whole near stratum')
        near_offsets, tail_offsets, near_origin, tail_origin = (
            tuple(range(near)), tuple(range(near, support)), DENSE, DENSE)
    else:
        tail_origin = SAMPLED
        if existing_near_offsets is not None:
            near_offsets = tuple(int(x) for x in existing_near_offsets)
            if (list(near_offsets) != sorted(set(near_offsets)) or len(near_offsets) != min(samples_per_stratum, near)
                    or any(not 0 <= x < near for x in near_offsets)):
                raise ValueError('Imported near offsets must be min(4, N) sorted unique offsets in [0, N)')
            near_origin = IMPORTED
        else:
            near_offsets = _draw(local_rng(PROTOCOL, seed, episode_key, source_group, 'near'), 0, near,
                                 samples_per_stratum)
            near_origin = SAMPLED
        tail_offsets = _draw(local_rng(PROTOCOL, seed, episode_key, source_group, 'tail'), near, support,
                             samples_per_stratum)
    return SourceHorizonPlan(
        episode_key=str(episode_key), source_group=source_group, admission_group=admission,
        episode_steps=episode_steps, near_support=near, tail_support=tail, near_offsets=near_offsets,
        tail_offsets=tail_offsets, seed=int(seed), near_origin=near_origin, tail_origin=tail_origin,
        recent_groups=recent_groups, near_horizon=near_horizon, samples_per_stratum=samples_per_stratum)


def plan_sha256(plan):
    text = json.dumps(asdict(plan), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()


def remaining_query_weights(plan):
    """offset -> (stratum, inclusion probability, stratum-mean weight, remaining-mean weight)."""
    weights = {}
    n_near, n_tail = len(plan.near_offsets), len(plan.tail_offsets)
    support = plan.remaining_support
    for offset in plan.near_offsets:
        weights[offset] = ('near', n_near / plan.near_support, 1 / n_near, plan.near_support / (support * n_near))
    for offset in plan.tail_offsets:
        weights[offset] = ('tail', n_tail / plan.tail_support, 1 / n_tail, plan.tail_support / (support * n_tail))
    return weights


def _stratum_mean(offsets, utilities):
    values = [utilities.get(offset) for offset in offsets]
    if any(v is None for v in values):
        return None
    for v in values:
        if not (isinstance(v, (int, float)) and math.isfinite(v) and v >= 0):
            raise ValueError(f'Invalid raw utility {v!r}')
    return math.fsum(values) / len(values)


def aggregate_horizon_labels(plan, utilities_by_query):
    """Strict requested-query coverage on raw utilities; missing is never zero.

    ``utilities_by_query``: offset relative to admission -> raw utility or None.
    """
    requested = set(plan.offsets)
    unexpected = set(utilities_by_query) - requested
    if unexpected:
        raise ValueError(f'Utilities for queries that were not planned: {sorted(unexpected)}')
    if plan.remaining_support == 0:
        return HorizonAggregate(None, None, None, 'not_eligible', 'not_applicable', 'not_eligible')
    near = _stratum_mean(plan.near_offsets, utilities_by_query)
    near_status = 'complete' if near is not None else 'missing'
    if plan.tail_support == 0:
        tail, tail_status = None, 'not_applicable'
    else:
        tail = _stratum_mean(plan.tail_offsets, utilities_by_query)
        tail_status = 'complete' if tail is not None else 'missing'
    if near is None or (plan.tail_support and tail is None):
        remaining, remaining_status = None, 'missing'
    else:
        support = plan.remaining_support
        remaining = (plan.near_support / support) * near + (plan.tail_support / support) * (tail or 0.0)
        remaining_status = 'complete'
    return HorizonAggregate(near, tail, remaining, near_status, tail_status, remaining_status)


def sample_patch_positions(seed, episode_key, source_group, rows=GRID_ROWS, cols=GRID_COLS, tile=TILE,
                           per_tile=PER_TILE):
    """Raster positions (row * cols + col) of the sampled label rows of one source frame.

    ``per_tile`` positions from every ``tile x tile`` tile, chosen before any
    utility or score exists and independently of residency.
    """
    if rows % tile or cols % tile:
        raise ValueError('Tiles must partition the grid')
    rng = local_rng(PROTOCOL, 'patches', seed, episode_key, source_group)
    positions = []
    for tile_row in range(rows // tile):
        for tile_col in range(cols // tile):
            cells = rng.choice(tile * tile, size=per_tile, replace=False)
            for cell in sorted(int(c) for c in cells):
                r, c = tile_row * tile + cell // tile, tile_col * tile + cell % tile
                positions.append(r * cols + c)
    return tuple(sorted(positions))


def displaced_row(seed, episode_key, source_group, query_step, attention_layer, bank_ids):
    """Old-bank row replaced by a non-resident candidate in the shadow context.

    Keyed by source/query/layer/context protocol only (never by target ID,
    candidate, utility or score), so every candidate of a source shares the
    same background context at a query, and near/tail reuse it.
    """
    bank = np.asarray(bank_ids, dtype=np.int64)
    if bank.size == 0 or np.any(np.diff(bank) <= 0):
        raise ValueError('Shadow displacement needs a nonempty bank sorted by original ID')
    rng = local_rng(PROTOCOL, SHADOW_CONTEXT, seed, episode_key, source_group, query_step, attention_layer)
    return int(bank[int(rng.integers(bank.size))])
