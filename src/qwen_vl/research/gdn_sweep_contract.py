"""Versioned contract for the M5 GDN-only S sweep (no model/CUDA imports).

Mode ``episodic_window_gdn_sweep_v1`` keeps M5's visibility exactly: in every
full-attention layer the instruction prefix, the ``R`` most recent complete
observation/action groups (including the current one) and at most ``B`` older
*visual* native K/V rows at their original logical positions.  Only the fixed
birth-time priority of old visual rows changes.  Native GDN/conv state still
processes every original token and is never evicted.

Every arm is an explicit registry entry.  A config must restate the entry's
selection unit, head reduction, scorer backend and numerical constants, so a
silently coupled or mistyped arm fails closed.  Calibration artifacts (S6
slow-head mask, S8 drift-denominator floors) are embedded with their SHA-256,
so a checkpoint's navigation contract is self-contained; a missing or altered
artifact raises, it never falls back to S0 or random.  The legacy
``episodic_window_v1`` contract (M5 = S0, M5c) is not modified.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from qwen_vl.research.episodic_contract import (
    ATTENTION_BACKEND, GDN_SWEEP_MODE, QWEN35_4B_PAIRS, SCORE_SEGMENTATION,
    VISUAL_TOKENS_PER_GROUP, derived_kv_guard, scoring_pairs_from_layer_types)

# Scorer backends.  ``qualified_m5_legacy_v1`` is the unchanged
# ``CommittedWriteScorer`` (fp32_torch_wy_block64_v1): the only S0/S7 path.
LEGACY_BACKEND = 'qualified_m5_legacy_v1'
# Same WY recurrence and block tiling as the legacy scan, keeping value-head
# norms (S6) or observation-boundary state statistics (S8).
HEADWISE_BACKEND = 'fp32_torch_wy_block64_headwise_v1'
BOUNDARY_BACKEND = 'fp32_torch_wy_block64_boundary_v1'
# Unscaled residual block solve (beta in column j, never divided out).
RESIDUAL_BACKEND = 'fp32_torch_unscaled_residual_block64_v1'
# Native prepared beta/values only; no recurrent state is read.
PREPARE_BACKEND = 'fp32_native_prepare_only_v1'
NO_BACKEND = 'none'

TOKEN, FRAME = 'token', 'frame300'
SLOW_HEADS_PER_LAYER = 8          # 25% of 32 value heads
MIN_DRIFT_FLOOR = 1e-6

# score_id -> (selection unit, head reduction, scorer backend, statistic).
ARM_REGISTRY = {
    'S0': (TOKEN, 'mean', LEGACY_BACKEND, 'committed_norm'),
    'S1': (TOKEN, 'mean', RESIDUAL_BACKEND, 'error_norm'),
    'S2': (TOKEN, 'mean', PREPARE_BACKEND, 'gated_value_norm'),
    'S3': (TOKEN, 'mean', PREPARE_BACKEND, 'value_norm'),
    'S4': (TOKEN, 'min', RESIDUAL_BACKEND, 'cosine_error'),
    'S5': (TOKEN, 'mean', RESIDUAL_BACKEND, 'cosine_error'),
    'S6': (TOKEN, 'slow_head_mean', HEADWISE_BACKEND, 'committed_norm'),
    'S7': (FRAME, 'visual_token_mean', LEGACY_BACKEND, 'committed_norm'),
    'S8N': (FRAME, 'full_state_frobenius', BOUNDARY_BACKEND, 'net_state_drift'),
    'S8W': (FRAME, 'full_state_frobenius', BOUNDARY_BACKEND, 'transported_write'),
    'S9': (TOKEN, 'mean', RESIDUAL_BACKEND, 'gate_complement_residual'),
    'C_RECENCY': (TOKEN, 'none', NO_BACKEND, 'latest_token_id'),
    'C_RANDOM': (TOKEN, 'none', NO_BACKEND, 'random_hash_v1'),
}
SLOW_HEAD_ARMS = ('S6',)
DRIFT_ARMS = ('S8N', 'S8W')
CONTROL_ARMS = ('C_RECENCY', 'C_RANDOM')

# Numerical constants are part of the scorer artifact; never tuned on navigation.
FIXED = dict(
    priority_timing='birth_fixed',
    score_dtype='float32',
    key_normalization='native_additive_eps_1e-6',
    cosine_epsilon=1e-6,
    zero_value_threshold=1e-6,
    state_layout='value_heads_key_by_value',
    attention_backend=ATTENTION_BACKEND,
    logical_position_policy='preserve',
    candidate_policy='resident_plus_newly_eligible',
    tie_break='priority_desc_original_id_asc',
)
REQUIRED_KEYS = frozenset((
    'mode', 'recent_groups_including_current', 'episodic_budget_tokens', 'selection_unit',
    'score_id', 'head_reduction', 'scorer_backend', 'score_segmentation', 'selector_seed',
    'evict_before_new_step', 'kv_guard_tokens', *FIXED))
OPTIONAL_KEYS = frozenset(('artifacts',))


def canonical_sha256(value):
    """SHA-256 of canonical JSON (sorted keys, no whitespace, float repr)."""
    text = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class GdnSweepSpec:
    """Attributes shared with ``EpisodicSpec`` keep the attention path unchanged."""
    version: str
    recent_groups: int
    budget_tokens: int
    granularity: str                  # selection unit: token | frame300
    score: str                        # registry score_id
    scoring_pairs: tuple[tuple[int, int], ...]
    selector_seed: int
    per_source_cap: None
    score_backend: str
    score_segmentation: str
    attention_backend: str
    head_reduction: str
    statistic: str
    priority_timing: str
    score_dtype: str
    key_normalization: str
    cosine_epsilon: float
    zero_value_threshold: float
    state_layout: str
    logical_position_policy: str
    candidate_policy: str
    tie_break: str
    slow_head_mask: tuple[tuple[int, tuple[int, ...]], ...] | None = None
    slow_head_mask_sha256: str | None = None
    drift_floor: tuple[tuple[int, float], ...] | None = None
    drift_floor_sha256: str | None = None
    calibration_sha256: str | None = None

    @property
    def attention_layers(self):
        return tuple(attention for _, attention in self.scoring_pairs)

    @property
    def needs_write_scores(self):
        """True when priorities come from a native-GDN scoring side pass."""
        return self.score not in CONTROL_ARMS

    @property
    def is_frame(self):
        return self.granularity == FRAME

    @property
    def frame_budget(self):
        return self.budget_tokens // VISUAL_TOKENS_PER_GROUP

    def scoring_layer_for(self, attention_layer):
        for gdn, attention in self.scoring_pairs:
            if attention == attention_layer:
                return gdn
        raise KeyError(attention_layer)

    def slow_heads(self, gdn_layer):
        if self.slow_head_mask is None:
            raise ValueError(f'{self.score} has no slow-head mask')
        return dict(self.slow_head_mask)[gdn_layer]

    def floor(self, gdn_layer):
        if self.drift_floor is None:
            raise ValueError(f'{self.score} has no drift denominator floor')
        return dict(self.drift_floor)[gdn_layer]

    def to_dict(self):
        result = asdict(self)
        result['scoring_pairs'] = [list(pair) for pair in self.scoring_pairs]
        if self.slow_head_mask is not None:
            result['slow_head_mask'] = [[layer, list(heads)] for layer, heads in self.slow_head_mask]
        if self.drift_floor is not None:
            result['drift_floor'] = [[layer, value] for layer, value in self.drift_floor]
        return result


def slow_head_mask_record(mask):
    """Canonical {"gdn_layer": [sorted heads]} record for hashing/embedding."""
    return {str(int(layer)): sorted(int(h) for h in heads) for layer, heads in dict(mask).items()}


def drift_floor_record(floors):
    return {str(int(layer)): float(value) for layer, value in dict(floors).items()}


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _parse_mask(raw, gdn_layers, value_heads):
    if not isinstance(raw, dict) or set(raw) != {str(layer) for layer in gdn_layers}:
        raise ValueError('slow_head_mask must name exactly the scoring GDN layers')
    mask = []
    for layer in gdn_layers:
        heads = raw[str(layer)]
        if (not isinstance(heads, list) or len(heads) != SLOW_HEADS_PER_LAYER
                or any(isinstance(h, bool) or not isinstance(h, int) for h in heads)
                or heads != sorted(set(heads)) or not all(0 <= h < value_heads for h in heads)):
            raise ValueError(f'slow_head_mask[{layer}] must be {SLOW_HEADS_PER_LAYER} sorted '
                             f'unique value heads in [0, {value_heads})')
        mask.append((layer, tuple(heads)))
    return tuple(mask)


def _parse_floor(raw, gdn_layers):
    if not isinstance(raw, dict) or set(raw) != {str(layer) for layer in gdn_layers}:
        raise ValueError('drift_denominator_floor must name exactly the scoring GDN layers')
    floors = []
    for layer in gdn_layers:
        value = raw[str(layer)]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < MIN_DRIFT_FLOOR):
            raise ValueError(f'drift floor for layer {layer} must be finite and >= {MIN_DRIFT_FLOOR}')
        floors.append((layer, float(value)))
    return tuple(floors)


def parse_gdn_sweep_spec(config, text_config=None):
    """Strict parser for ``episodic_window_gdn_sweep_v1``."""
    memory = config.get('memory', {})
    if memory.get('mode') != GDN_SWEEP_MODE:
        raise ValueError(f'Not a {GDN_SWEEP_MODE} memory config')
    unknown = set(memory) - REQUIRED_KEYS - OPTIONAL_KEYS
    missing = REQUIRED_KEYS - set(memory)
    if unknown or missing:
        raise ValueError(f'GDN sweep memory keys: unknown {sorted(unknown)}, missing {sorted(missing)}')
    if config['model']['output_mode'] != 'qwen_text':
        raise ValueError('The GDN sweep is qualified only for canonical text actions')
    if config['model'].get('step_lane', {}).get('enabled', False):
        raise ValueError('The GDN sweep disables the step lane')
    if config.get('research'):
        raise ValueError('The core GDN sweep has no teacher/auxiliary research objective')
    if config['runtime']['text_attention'] != 'simplememvln_step_attention':
        raise ValueError('The GDN sweep requires the project step attention')
    if memory['evict_before_new_step'] is not True:
        raise ValueError('Episodic window promotes/evicts before each new observation')
    score = memory['score_id']
    if score not in ARM_REGISTRY:
        raise ValueError(f'Unknown GDN sweep score_id {score!r}')
    unit, reduction, backend, statistic = ARM_REGISTRY[score]
    declared = (memory['selection_unit'], memory['head_reduction'], memory['scorer_backend'])
    if declared != (unit, reduction, backend):
        raise ValueError(f'{score} must declare unit/reduction/backend {(unit, reduction, backend)}, '
                         f'got {declared}')
    segmentation = NO_BACKEND if backend == NO_BACKEND else SCORE_SEGMENTATION
    if memory['score_segmentation'] != segmentation:
        raise ValueError(f'{score} score_segmentation must be {segmentation!r}')
    for key, expected in FIXED.items():
        value = memory[key]
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f'{key}={value!r} differs from the fixed sweep convention {expected!r}')
    recent = _integer(memory['recent_groups_including_current'], 'recent groups', 1)
    budget = _integer(memory['episodic_budget_tokens'], 'episodic budget', 1)
    if unit == FRAME and budget % VISUAL_TOKENS_PER_GROUP:
        raise ValueError('frame300 budget must be a whole number of 300-row frames')
    seed = _integer(memory['selector_seed'], 'selector seed', 0)
    pairs = QWEN35_4B_PAIRS
    value_heads = 32
    if text_config is not None:
        pairs = scoring_pairs_from_layer_types(list(text_config.layer_types))
        value_heads = int(text_config.linear_num_value_heads)
    gdn_layers = tuple(gdn for gdn, _ in pairs)
    artifacts = memory.get('artifacts') or {}
    if not isinstance(artifacts, dict):
        raise ValueError('artifacts must be a mapping')
    allowed = set()
    mask = mask_sha = floor = floor_sha = None
    if score in SLOW_HEAD_ARMS:
        allowed = {'slow_head_mask', 'slow_head_mask_sha256', 'calibration_sha256'}
        if not {'slow_head_mask', 'slow_head_mask_sha256'} <= set(artifacts):
            raise ValueError(f'{score} requires an embedded frozen slow-head mask and its SHA-256')
        mask = _parse_mask(artifacts['slow_head_mask'], gdn_layers, value_heads)
        mask_sha = artifacts['slow_head_mask_sha256']
        if canonical_sha256(slow_head_mask_record(mask)) != mask_sha:
            raise ValueError('slow_head_mask SHA-256 mismatch')
    elif score in DRIFT_ARMS:
        allowed = {'drift_denominator_floor', 'drift_denominator_floor_sha256', 'calibration_sha256'}
        if not {'drift_denominator_floor', 'drift_denominator_floor_sha256'} <= set(artifacts):
            raise ValueError(f'{score} requires embedded frozen drift floors and their SHA-256')
        floor = _parse_floor(artifacts['drift_denominator_floor'], gdn_layers)
        floor_sha = artifacts['drift_denominator_floor_sha256']
        if canonical_sha256(drift_floor_record(floor)) != floor_sha:
            raise ValueError('drift_denominator_floor SHA-256 mismatch')
    if set(artifacts) - allowed:
        raise ValueError(f'{score} does not consume artifacts {sorted(set(artifacts) - allowed)}')
    calibration = artifacts.get('calibration_sha256')
    if calibration is not None and (not isinstance(calibration, str) or len(calibration) != 64):
        raise ValueError('calibration_sha256 must be a SHA-256 hex digest')
    spec = GdnSweepSpec(
        version=GDN_SWEEP_MODE, recent_groups=recent, budget_tokens=budget, granularity=unit,
        score=score, scoring_pairs=pairs, selector_seed=seed, per_source_cap=None,
        score_backend=backend, score_segmentation=segmentation,
        head_reduction=reduction, statistic=statistic,
        slow_head_mask=mask, slow_head_mask_sha256=mask_sha,
        drift_floor=floor, drift_floor_sha256=floor_sha, calibration_sha256=calibration,
        **{key: value for key, value in FIXED.items()})
    guard = derived_kv_guard(config, spec)
    if memory['kv_guard_tokens'] != guard:
        raise ValueError(f'kv_guard_tokens must equal the derived structural guard {guard}')
    return spec


def arm_memory_config(score, *, recent=4, budget=1200, selector_seed=429, artifacts=None,
                      max_prefix_tokens=2048, max_step_group_tokens=384):
    """The complete memory section for one registry arm (configs are generated, not hand-edited)."""
    unit, reduction, backend, _ = ARM_REGISTRY[score]
    memory = dict(
        mode=GDN_SWEEP_MODE, recent_groups_including_current=recent,
        episodic_budget_tokens=budget, selection_unit=unit, score_id=score,
        head_reduction=reduction, scorer_backend=backend,
        score_segmentation=NO_BACKEND if backend == NO_BACKEND else SCORE_SEGMENTATION,
        selector_seed=selector_seed, evict_before_new_step=True,
        kv_guard_tokens=max_prefix_tokens + recent * max_step_group_tokens + budget, **FIXED)
    if artifacts:
        memory['artifacts'] = artifacts
    return dict(memory=memory)
