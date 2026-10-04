"""Sequential FP32 delta scan, deliberately confined to test code."""
import torch
from qwen_vl.research.lane_contract import StepLaneSpec


def tiny_spec(**overrides):
    defaults = dict(layers=(0, 1), num_key_heads=1, num_value_heads=2, key_dim=3,
                    value_dim=5, half_lives_steps=(8., 24.))
    defaults.update(overrides)
    return StepLaneSpec(**defaults)


def reference_kernel(q, k, v, g, beta, *, initial_state=None,
                     output_final_state=False, scale=None, use_qk_l2norm_in_kernel=False):
    dtype = v.dtype
    q, k, v = q.float(), k.float(), v.float()
    if use_qk_l2norm_in_kernel:
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    if scale is None:
        scale = q.shape[-1] ** -0.5
    batch, length, heads, key_dim = q.shape
    state = (torch.zeros(batch, heads, key_dim, v.shape[-1], device=q.device)
             if initial_state is None else initial_state)
    out = []
    for i in range(length):
        decayed = state * g[:, i].float().exp()[..., None, None]
        predicted = torch.einsum('bhkv,bhk->bhv', decayed, k[:, i])
        delta = beta[:, i].float()[..., None] * (v[:, i] - predicted)
        state = decayed + k[:, i, :, :, None] * delta[:, :, None, :]
        out.append(torch.einsum('bhkv,bhk->bhv', state, q[:, i] * scale))
    return torch.stack(out, 1).to(dtype), state if output_final_state else None
