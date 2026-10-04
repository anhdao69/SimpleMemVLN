"""Inference-only sidecar: lane matrices and logical append cursors per session."""
import torch
from qwen_vl.research.lane_contract import StepLaneSpec


class StepLaneCache:
    def __init__(self, spec: StepLaneSpec, episode_uid: str, *, device):
        if not spec.layers or len(set(spec.layers)) != len(spec.layers):
            raise ValueError('Lane cache requires unique selected layers')
        self.spec = spec
        self.device = torch.device(device)
        self._shape = (1, spec.num_value_heads, spec.key_dim, spec.value_dim)
        self.reset(episode_uid)

    def reset(self, episode_uid: str) -> None:
        if not isinstance(episode_uid, str) or not episode_uid:
            raise ValueError('Lane cache requires a nonempty episode_uid')
        self.episode_uid = episode_uid
        self._states = {index: torch.zeros(self._shape, device=self.device, dtype=torch.float32)
                        for index in self.spec.layers}
        self._processed_tokens = {index: 0 for index in self.spec.layers}

    @staticmethod
    def _inference_only():
        if torch.is_grad_enabled():
            raise RuntimeError('Lane cache access is forbidden while grad is enabled')

    def _check_start(self, layer_idx, logical_start):
        if type(layer_idx) is not int or layer_idx not in self._states:
            raise ValueError('Unknown lane layer index')
        if type(logical_start) is not int or logical_start < 0 or logical_start != self._processed_tokens[layer_idx]:
            raise ValueError('Lane append must start at its current logical-token cursor')

    def initial_for(self, layer_idx: int, logical_start: int) -> torch.Tensor:
        self._inference_only()
        self._check_start(layer_idx, logical_start)
        # Serving code receives a copy: even a mutating backend cannot corrupt the
        # last committed state before validation of the complete append.
        return self._states[layer_idx].clone()

    def commit(self, layer_idx: int, logical_start: int, appended_tokens: int,
               final_state: torch.Tensor) -> None:
        self._inference_only()
        self._check_start(layer_idx, logical_start)
        if type(appended_tokens) is not int or appended_tokens <= 0:
            raise ValueError('Lane commit requires a positive actual token count')
        if not isinstance(final_state, torch.Tensor) or final_state.shape != self._shape:
            raise ValueError('Invalid lane state shape')
        if final_state.dtype != torch.float32 or final_state.device != self._states[layer_idx].device:
            raise ValueError('Lane state must be FP32 on cache device')
        if final_state.requires_grad or final_state.grad_fn is not None:
            raise ValueError('Lane cache cannot retain a gradient graph')
        if not torch.isfinite(final_state).all():
            raise ValueError('Lane state must be finite')
        # Validate everything before replacing either state or accounting. Clone
        # isolates the store from aliases held by the kernel or its caller.
        stored = final_state.detach().clone()
        self._states[layer_idx] = stored
        self._processed_tokens[layer_idx] = logical_start + appended_tokens

    def assert_complete_append(self, logical_end: int) -> None:
        if type(logical_end) is not int or logical_end < 0:
            raise ValueError('Invalid logical append end')
        if any(count != logical_end for count in self._processed_tokens.values()):
            raise ValueError('Selected lane layers did not all complete the logical append')
