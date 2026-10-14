"""Version-pinned, per-instance GeoAddress integration for all native cache modes.

Only addition to the native equations is on raw Q/K after conv and before head
expansion. Optional relation scalar is computed here, inside layer checkpointing.
No graph hooks or mutable current-episode metadata.
"""
from types import MethodType
import torch
from torch.nn import functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states
from qwen_vl.models.geoaddress import GeoAddress


def geoaddress_gdn(gdn, hidden_states, image_mask, cache_params=None, attention_mask=None,
                   relation=None, **kwargs):
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch, length, _ = hidden_states.shape
    previous = cache_params is not None and cache_params.has_previous_state(gdn.layer_idx)
    if previous:
        conv_state = cache_params.layers[gdn.layer_idx].conv_states
        recurrent_state = cache_params.layers[gdn.layer_idx].recurrent_states
    mixed = gdn.in_proj_qkv(hidden_states).transpose(1, 2)
    z = gdn.in_proj_z(hidden_states).reshape(batch, length, -1, gdn.head_v_dim)
    b, a = gdn.in_proj_b(hidden_states), gdn.in_proj_a(hidden_states)
    if previous and length == 1:
        mixed = gdn.causal_conv1d_update(mixed, conv_state, gdn.conv1d.weight.squeeze(1),
                                       gdn.conv1d.bias, gdn.activation)
    else:
        if previous:
            mixed = torch.cat([conv_state, mixed], dim=-1)
        if cache_params is not None:
            new_state = F.pad(mixed, (gdn.conv_kernel_size - mixed.shape[-1], 0))
            cache_params.update_conv_state(new_state, gdn.layer_idx)
        if gdn.causal_conv1d_fn is not None:
            mixed = gdn.causal_conv1d_fn(x=mixed, weight=gdn.conv1d.weight.squeeze(1),
                bias=gdn.conv1d.bias, activation=gdn.activation, seq_idx=kwargs.get('seq_idx'))
        else:
            mixed = F.silu(gdn.conv1d(mixed)[:, :, :mixed.shape[-1]])
        if previous:
            mixed = mixed[:, :, -length:]
    q, k, v = torch.split(mixed.transpose(1, 2), [gdn.key_dim, gdn.key_dim, gdn.value_dim], dim=-1)
    q = q.reshape(batch, length, -1, gdn.head_k_dim)
    k = k.reshape(batch, length, -1, gdn.head_k_dim)
    v = v.reshape(batch, length, -1, gdn.head_v_dim)
    q, k = gdn.geoaddress(hidden_states, q, k, image_mask)
    loss = hidden_states.new_zeros((), dtype=torch.float32)
    if relation is not None:
        if cache_params is not None:
            raise ValueError('Relation objective is uncached training only')
        loss = relation(q, k)
    beta = b.sigmoid()
    g = -gdn.A_log.float().exp() * F.softplus(a.float() + gdn.dt_bias)
    repeat = gdn.num_v_heads // gdn.num_k_heads
    if repeat > 1:
        q, k = q.repeat_interleave(repeat, dim=2), k.repeat_interleave(repeat, dim=2)
    if previous and length == 1:
        out, final = gdn.recurrent_gated_delta_rule(q, k, v, g=g, beta=beta,
            initial_state=recurrent_state, output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True)
    else:
        out, final = gdn.chunk_gated_delta_rule(q, k, v, g=g, beta=beta,
            initial_state=recurrent_state if previous else None,
            output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True,
            cu_seqlens=kwargs.get('cu_seq_lens_q'))
    if cache_params is not None:
        cache_params.update_recurrent_state(final, gdn.layer_idx)
    out = gdn.norm(out.reshape(-1, gdn.head_v_dim), z.reshape(-1, gdn.head_v_dim))
    return gdn.out_proj(out.reshape(batch, length, -1)), loss


def _forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    mask = kwargs.pop('geoaddress_image_mask', None)
    if mask is None:
        raise ValueError('GeoAddress runtime requires explicit image mask')
    return geoaddress_gdn(self, hidden_states, mask, cache_params, attention_mask, **kwargs)[0]


def install_geoaddress(text_model, *, layer_index=24, hidden_size=2560, head_dim=128, seed=429):
    from qwen_vl.models.episodic_forward import check_native_source, NATIVE_FORWARD_SHA256
    check_native_source(NATIVE_FORWARD_SHA256)
    spec = (layer_index, hidden_size, head_dim, seed)
    previous = getattr(text_model, '_geoaddress_spec', None)
    if previous is not None:
        if previous != spec:
            raise ValueError('Incompatible repeated GeoAddress installation')
        return
    if layer_index >= len(text_model.layers) or not hasattr(text_model.layers[layer_index], 'linear_attn'):
        raise ValueError('GeoAddress selected decoder must be native GDN')
    gdn = text_model.layers[layer_index].linear_attn
    if 'forward' in gdn.__dict__ or hasattr(gdn, 'geoaddress'):
        raise ValueError('GeoAddress cannot compose with another installed mixer adapter')
    if (gdn.in_proj_qkv.in_features != hidden_size or gdn.head_k_dim != head_dim or gdn.num_k_heads < 2):
        raise ValueError('GeoAddress native dimensions differ')
    parameter = next(gdn.parameters())
    adapter = GeoAddress(hidden_size, head_dim, seed=seed).to(device=parameter.device, dtype=parameter.dtype)
    object.__setattr__(gdn, '_geoaddress_native_forward', gdn.forward.__func__)
    gdn.geoaddress = adapter
    gdn.forward = MethodType(_forward, gdn)
    text_model._geoaddress_spec = spec


def geoaddress_decoder(layer, hidden, image_mask, relation=None):
    residual = hidden
    output, loss = geoaddress_gdn(layer.linear_attn, layer.input_layernorm(hidden), image_mask, relation=relation)
    hidden = residual + output
    return hidden + layer.mlp(layer.post_attention_layernorm(hidden)), loss
