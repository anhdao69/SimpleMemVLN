"""Per-instance native-forward composition, preserving pretrained namespaces."""
from types import MethodType
import torch
from qwen_vl.models.parallel_step_lane import ParallelStepLane

_LANE_KEYS = ('step_lane_roles', 'step_lane_cache', 'step_lane_logical_start')


def _strip_forward(self, *args, **kwargs):
    for name in _LANE_KEYS:
        kwargs.pop(name, None)
    return self._step_lane_passthrough_forward(self, *args, **kwargs)


def _lane_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    roles = kwargs.pop('step_lane_roles', None)
    cache = kwargs.pop('step_lane_cache', None)
    start = kwargs.pop('step_lane_logical_start', None)
    if roles is None:
        raise ValueError('Enabled step lane requires explicit roles')
    if cache is not None:
        if self.training or torch.is_grad_enabled() or cache_params is None:
            raise ValueError('Lane sidecar requires eval, no gradients and native cache')
        if start is None:
            raise ValueError('Lane sidecar requires logical append start')
        initial = cache.initial_for(self._step_lane_layer_idx, start)
    else:
        if cache_params is not None or start is not None:
            raise ValueError('Cached native execution requires a matching lane sidecar')
        initial = None
    native = self._step_lane_native_forward(
        self, hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs
    )
    lane, final = self.step_lane(hidden_states, roles, initial, return_final_state=cache is not None)
    if cache is not None:
        cache.commit(self._step_lane_layer_idx, start, hidden_states.shape[1], final)
    return native + lane


def install_step_lanes(model, spec, *, init_seed):
    """Install before optimizer construction; same-spec calls preserve learned weights."""
    if spec is None:
        return
    previous = getattr(model, 'step_lane_spec', None)
    if previous is not None:
        if previous != spec:
            raise ValueError('Incompatible repeated step lane installation')
        return
    layers = model.backbone.model.language_model.layers
    for index in spec.layers:
        if index >= len(layers) or not hasattr(layers[index], 'linear_attn'):
            raise ValueError(f'Step lane layer {index} is not native linear attention')
        native = layers[index].linear_attn
        if hasattr(native, 'step_lane') or 'forward' in native.__dict__:
            raise ValueError(f'Layer {index} already has an incompatible forward adapter')
    devices = sorted({next(layers[index].linear_attn.parameters()).device.index
                      for index in spec.layers
                      if next(layers[index].linear_attn.parameters()).is_cuda})
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(init_seed)
        for device in devices:
            torch.cuda.default_generators[device].manual_seed(init_seed)
        # Construct all lanes before mutating any native forward.
        created = {}
        for index in spec.layers:
            native = layers[index].linear_attn
            parameter = next(native.parameters())
            created[index] = ParallelStepLane(
                model.config.text_config.hidden_size, spec,
                device=parameter.device, dtype=parameter.dtype,
                chunk_kernel=native.chunk_gated_delta_rule,
                recurrent_kernel=native.recurrent_gated_delta_rule,
                lane_seed=init_seed + index,
            )
    for index, layer in enumerate(layers):
        if index in created:
            native = layer.linear_attn
            object.__setattr__(native, '_step_lane_native_forward', native.forward.__func__)
            native._step_lane_layer_idx = index
            native.step_lane = created[index]
            native.forward = MethodType(_lane_forward, native)
        else:
            native = layer.linear_attn if hasattr(layer, 'linear_attn') else layer.self_attn
            object.__setattr__(native, '_step_lane_passthrough_forward', native.forward.__func__)
            native.forward = MethodType(_strip_forward, native)
    model.step_lane_spec = spec
