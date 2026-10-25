"""Training-only matrix optimizer contract (no torch imports): absent means AdamW everywhere.

``training.matrix_optimizer`` selects ZeRO-offload Muon (``qwen_vl.train.zero_muon``)
for the language model's 2D hidden-layer weights.  Every field is explicit and
exact, so a typo fails closed instead of silently training with AdamW.
"""
import math

MATRIX_OPTIMIZER = 'muon_zero_offload_v1'
SCOPE = 'language_model_layers_2d'
SPEC_KEYS = {'name', 'scope', 'momentum', 'nesterov', 'ns_steps', 'ns_coefficients', 'eps', 'adjust_lr_fn',
             'group_numel'}


def parse_matrix_optimizer(training):
    """The validated ``training.matrix_optimizer`` mapping, or None when absent (AdamW everywhere)."""
    spec = training.get('matrix_optimizer')
    if spec is None:
        return None
    if not isinstance(spec, dict) or set(spec) != SPEC_KEYS:
        raise ValueError(f'matrix_optimizer must declare exactly {sorted(SPEC_KEYS)}')
    if spec['name'] != MATRIX_OPTIMIZER or spec['scope'] != SCOPE:
        raise ValueError(f'Unsupported matrix optimizer {spec["name"]!r}/{spec["scope"]!r}')
    momentum, coefficients = spec['momentum'], spec['ns_coefficients']
    if type(momentum) is not float or not 0.0 <= momentum < 1.0:
        raise ValueError('Muon momentum must be a float in [0, 1)')
    if type(spec['nesterov']) is not bool:
        raise ValueError('Muon nesterov must be a boolean')
    if type(spec['ns_steps']) is not int or not 1 <= spec['ns_steps'] < 100:
        raise ValueError('Muon ns_steps must be an integer in [1, 99]')
    if (not isinstance(coefficients, list) or len(coefficients) != 3
            or any(type(c) is not float or not math.isfinite(c) for c in coefficients)):
        raise ValueError('Muon ns_coefficients must be three finite floats')
    if type(spec['eps']) is not float or not spec['eps'] > 0:
        raise ValueError('Muon eps must be a positive float')
    if spec['adjust_lr_fn'] not in ('original', 'match_rms_adamw'):
        raise ValueError('Muon adjust_lr_fn must be original or match_rms_adamw')
    if type(spec['group_numel']) is not int or spec['group_numel'] < 1:
        raise ValueError('Muon group_numel must be a positive integer')
    if training.get('cpu_optimizer_impl') != 'deepspeed_cpu_adam':
        raise ValueError('Muon requires the DeepSpeedCPUAdam ZeRO-offload path for the other parameters')
    return spec
