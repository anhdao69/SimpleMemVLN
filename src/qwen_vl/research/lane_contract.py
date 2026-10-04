"""Immutable, JSON-compatible configuration for the experimental parallel lane."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class StepLaneSpec:
    version: str = 'parallel_step_lane_v1'
    layers: tuple[int, ...] = (16, 20, 24, 28)
    num_key_heads: int = 2
    num_value_heads: int = 4
    key_dim: int = 128
    value_dim: int = 128
    clock_mode: str = 'step_end'
    prefix_mode: str = 'read_only_zero'
    half_lives_steps: tuple[float, ...] = (8., 24., 64., 192.)
    beta_init: float = 0.1
    token_rate_reference: float | None = None
    projection_init_std: float = 0.02
    eps: float = 1e-6

    def to_dict(self):
        result = asdict(self)
        result['layers'] = list(self.layers)
        result['half_lives_steps'] = list(self.half_lives_steps)
        return result


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')


def parse_step_lane_spec(config: dict, text_config=None) -> StepLaneSpec | None:
    """Parse whole navigation config; disabled configurations retain their legacy contract.

    Text config is optional for serializer/config validation. Model installation must
    provide it to validate the real decoder layout before creating parameters.
    """
    raw = config['model'].get('step_lane', {})
    if not isinstance(raw, dict):
        raise ValueError('model.step_lane must be a mapping')
    if raw.get('enabled', False) is False:
        return None
    if raw.get('enabled') is not True:
        raise ValueError('model.step_lane.enabled must be boolean')
    if config['model']['output_mode'] == 'classification':
        raise ValueError('classification has no post-decision closing block in lane v1')
    if config['model']['output_mode'] not in ('qwen_text', 'candidate_logits'):
        raise ValueError('Unsupported lane output mode')
    unknown = set(raw) - {'enabled', 'init_seed'} - StepLaneSpec.__dataclass_fields__.keys()
    if unknown:
        raise ValueError(f'Unknown step_lane fields: {sorted(unknown)}')
    values = {key: value for key, value in raw.items() if key not in ('enabled', 'init_seed')}
    for key in ('layers', 'half_lives_steps'):
        if key in values:
            try:
                values[key] = tuple(values[key])
            except TypeError as exc:
                raise ValueError(f'{key} must be a sequence') from exc
    spec = StepLaneSpec(**values)
    if spec.version != 'parallel_step_lane_v1':
        raise ValueError('Unsupported lane version')
    if spec.clock_mode not in ('step_end', 'token_rate_calibrated') or spec.prefix_mode != 'read_only_zero':
        raise ValueError('Unsupported lane clock/prefix mode')
    if not spec.layers or any(type(i) is not int or i < 0 for i in spec.layers) or tuple(sorted(set(spec.layers))) != spec.layers:
        raise ValueError('Lane layers must be sorted, unique nonnegative indices')
    if any(type(value) is not int for value in (spec.num_key_heads, spec.num_value_heads, spec.key_dim, spec.value_dim)) or (spec.num_key_heads, spec.num_value_heads, spec.key_dim, spec.value_dim) != (2, 4, 128, 128):
        raise ValueError('Lane v1 requires 2 key heads, 4 value heads, and dimensions 128')
    if len(spec.half_lives_steps) != spec.num_value_heads:
        raise ValueError('Each value head requires a half-life')
    for value in spec.half_lives_steps:
        _positive(value, 'half_lives_steps')
    _positive(spec.beta_init, 'beta_init')
    if spec.beta_init >= 1:
        raise ValueError('beta_init must be less than one')
    _positive(spec.projection_init_std, 'projection_init_std')
    _positive(spec.eps, 'eps')
    if spec.clock_mode == 'token_rate_calibrated':
        _positive(spec.token_rate_reference, 'token_rate_reference')
    elif spec.token_rate_reference is not None:
        raise ValueError('token_rate_reference is only valid for calibrated control')
    if 'init_seed' in raw and (type(raw['init_seed']) is not int or raw['init_seed'] < 0):
        raise ValueError('init_seed must be a nonnegative integer')
    if text_config is not None:
        def field(name):
            return text_config.get(name) if isinstance(text_config, dict) else getattr(text_config, name, None)
        kinds = field('layer_types')
        if field('hidden_size') != 2560 or kinds is None:
            raise ValueError('Lane v1 requires the pinned 2560-wide decoder with explicit layer_types')
        for index in spec.layers:
            if index >= len(kinds) or kinds[index] != 'linear_attention':
                raise ValueError(f'Lane layer {index} must be an existing linear_attention layer')
    return spec
