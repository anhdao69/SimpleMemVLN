"""Immutable episodic-window contract (no model/CUDA imports).

Memory mode ``episodic_window_v1`` keeps, in every full-attention layer, the
instruction prefix, the most recent ``R`` complete observation/action groups
(including the current one) and at most ``B`` older *visual* K/V rows chosen by
a fixed write-time priority.  Native GDN/conv state still processes every
original token and is never evicted.  Every value that changes visibility,
score, cap or backend is part of the saved navigation contract, so strict
reload/resume rejects any mismatch.
"""
from dataclasses import asdict, dataclass

EPISODIC_MODE = 'episodic_window_v1'
SCORES = ('none', 'committed_mean_l2_v1', 'random_hash_v1')
GRANULARITIES = ('token', 'visual_group')
# FP32 PyTorch WY-block committed-write scan; see models/committed_write.py.
SCORE_BACKEND = 'fp32_torch_wy_block64_v1'
# Score-helper segmentation shared by training and serving: the prefix, each
# observation block, and each group's complete response (ordinary response
# tokens + EOS + separator) are separate segments; each segment is split into
# internal blocks of at most 64 tokens from its own start.  The carried helper
# state is never reset inside an episode.
SCORE_SEGMENTATION = 'prefix_obs_response_block64_v1'
# Index-gathered FA2 varlen windows, bounded batches, FP32 K/V gradient scatter.
ATTENTION_BACKEND = 'episodic_fa2_varlen_gather_fp32acc_v1'
VISUAL_TOKENS_PER_GROUP = 300
SCORE_BLOCK_TOKENS = 64


@dataclass(frozen=True)
class EpisodicSpec:
    version: str
    recent_groups: int
    budget_tokens: int
    granularity: str
    score: str
    scoring_pairs: tuple[tuple[int, int], ...]
    selector_seed: int
    per_source_cap: int | None
    score_backend: str
    score_segmentation: str
    attention_backend: str

    @property
    def attention_layers(self):
        return tuple(attention for _, attention in self.scoring_pairs)

    @property
    def needs_write_scores(self):
        return self.budget_tokens > 0 and self.score == 'committed_mean_l2_v1'

    def scoring_layer_for(self, attention_layer):
        for gdn, attention in self.scoring_pairs:
            if attention == attention_layer:
                return gdn
        raise KeyError(attention_layer)

    def to_dict(self):
        result = asdict(self)
        result['scoring_pairs'] = [list(pair) for pair in self.scoring_pairs]
        return result


def scoring_pairs_from_layer_types(layer_types):
    """Each full-attention layer is scored only by the GDN layer immediately below it."""
    pairs = []
    for index, kind in enumerate(layer_types):
        if kind == 'full_attention':
            if index == 0 or layer_types[index - 1] != 'linear_attention':
                raise ValueError(f'Attention layer {index} has no preceding GDN scorer')
            pairs.append((index - 1, index))
        elif kind != 'linear_attention':
            raise ValueError(f'Unsupported layer type {kind!r}')
    if not pairs:
        raise ValueError('Episodic memory requires full-attention layers')
    return tuple(pairs)


# Pinned Qwen3.5-4B layout; validated against the real text config at load.
QWEN35_4B_PAIRS = tuple((layer - 1, layer) for layer in range(3, 32, 4))


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def derived_kv_guard(config, spec):
    """Structural bound: prefix cap + R complete groups + B old visual rows."""
    obs = config['observations']
    return (obs['max_prefix_tokens'] + spec.recent_groups * obs['max_step_group_tokens']
            + spec.budget_tokens)


def parse_episodic_spec(config, text_config=None):
    """Return the spec for ``episodic_window_v1``; ``None`` for legacy memory modes."""
    memory = config['memory']
    if memory.get('mode') != EPISODIC_MODE:
        for key in memory:
            if key.startswith('episodic_') or key == 'recent_groups_including_current':
                raise ValueError(f'{key} is only valid for {EPISODIC_MODE}')
        return None
    if config['model']['output_mode'] != 'qwen_text':
        raise ValueError('Episodic memory v1 is qualified only for canonical text actions')
    if config['model'].get('step_lane', {}).get('enabled', False):
        raise ValueError('Episodic campaign disables the step lane')
    recent = _integer(memory.get('recent_groups_including_current'), 'recent groups', 1)
    budget = _integer(memory.get('episodic_budget_tokens'), 'episodic budget', 0)
    granularity = memory.get('episodic_granularity', 'token')
    score = memory.get('episodic_score')
    if granularity not in GRANULARITIES:
        raise ValueError(f'Unknown episodic granularity {granularity!r}')
    if score not in SCORES:
        raise ValueError(f'Unknown episodic score {score!r}')
    if (budget == 0) != (score == 'none'):
        raise ValueError('B0 requires score none; a nonzero budget requires a real score')
    if granularity == 'visual_group':
        if budget % VISUAL_TOKENS_PER_GROUP:
            raise ValueError('visual_group budget must be a whole number of 300-token groups')
        # M3 is a documented Phase-2 granularity; its plan builder is not part of v1.
        raise ValueError('visual_group granularity is reserved for Phase 2 and not implemented')
    cap = memory.get('episodic_per_source_cap')
    if cap is not None:
        raise ValueError('Per-source caps are a separately named variant, not part of v1')
    seed = _integer(memory.get('episodic_selector_seed', 0), 'selector seed', 0)
    if not memory.get('evict_before_new_step', False):
        raise ValueError('Episodic window promotes/evicts before each new observation')
    if config['runtime']['text_attention'] != 'simplememvln_step_attention':
        raise ValueError('Episodic window requires the project step attention')
    pairs = QWEN35_4B_PAIRS
    if text_config is not None:
        pairs = scoring_pairs_from_layer_types(list(text_config.layer_types))
    spec = EpisodicSpec(
        version=EPISODIC_MODE, recent_groups=recent, budget_tokens=budget,
        granularity=granularity, score=score, scoring_pairs=pairs,
        selector_seed=seed, per_source_cap=None,
        score_backend=SCORE_BACKEND if score == 'committed_mean_l2_v1' else 'none',
        score_segmentation=SCORE_SEGMENTATION if score == 'committed_mean_l2_v1' else 'none',
        attention_backend=ATTENTION_BACKEND,
    )
    for key, expected in (('episodic_score_backend', spec.score_backend),
                          ('episodic_score_segmentation', spec.score_segmentation),
                          ('episodic_attention_backend', spec.attention_backend)):
        if key in memory and memory[key] != expected:
            raise ValueError(f'{key}={memory[key]!r} differs from implemented {expected!r}')
    guard = derived_kv_guard(config, spec)
    if memory.get('kv_guard_tokens') != guard:
        raise ValueError(f'kv_guard_tokens must equal the derived structural guard {guard}')
    return spec


def spec_from_dict(raw):
    raw = dict(raw)
    raw['scoring_pairs'] = tuple(tuple(pair) for pair in raw['scoring_pairs'])
    return EpisodicSpec(**raw)
