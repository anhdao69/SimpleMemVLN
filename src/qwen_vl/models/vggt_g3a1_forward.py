"""G3+A1 native GDN layer24: GeoAddress Q/K adapters, A1 relation KL and G3 state reads.

One uncached training forward composing, in native order, exactly the two
qualified single-arm layer24 paths:

  native projections/convolution -> raw Q/K/V split
  -> GeoAddress image-token residual on raw heads 0/1 (serving-persistent, as A0/A1)
  -> A1 relation KL on the ADAPTED raw heads 0/1 (as ``geoaddress_gdn``)
  -> native head repetition
  -> G3 read-only probe rows after the anchors (as ``native_state_gdn``)
  -> ONE pinned FLA chunk call
  -> real rows restored before native gated norm/z, out_proj, residual and MLP.

The G3 query context is the detached native query at the anchor AFTER the
adapter, i.e. the query the deployed model actually uses there (anchors are
text/cue rows, where the image-only adapter adds nothing directly). G3 probes
read a recurrent state written with adapted image keys, so the state loss also
trains the adapters. Returns the two objective sums ``[G3, A1]`` separately.
"""
import torch
from torch.nn import functional as F
from qwen_vl.research.vggt_g3 import insert_probes


def address_state_gdn(gdn, hidden, head, anchors, lags, targets, image_mask, relation):
    batch, length, _ = hidden.shape
    if batch != 1:
        raise ValueError('G3A1 supports one complete unpadded episode')
    if getattr(gdn, 'geoaddress', None) is None:
        raise ValueError('G3A1 requires the installed GeoAddress adapter at the selected GDN layer')
    if relation is None or head is None:
        raise ValueError('G3A1 requires both the A1 relation and the G3 state head')
    kernel = gdn.chunk_gated_delta_rule
    if not (getattr(kernel, '__module__', '').startswith('fla.ops.gated_delta_rule.')
            and getattr(kernel, '__name__', '') == 'chunk_gated_delta_rule'):
        raise RuntimeError('G3A1 requires the qualified native FLA chunk kernel')
    mixed = gdn.in_proj_qkv(hidden).transpose(1, 2)
    z = gdn.in_proj_z(hidden).reshape(batch, length, -1, gdn.head_v_dim)
    b = gdn.in_proj_b(hidden)
    a = gdn.in_proj_a(hidden)
    if gdn.causal_conv1d_fn is not None:
        mixed = gdn.causal_conv1d_fn(x=mixed, weight=gdn.conv1d.weight.squeeze(1),
                                     bias=gdn.conv1d.bias, activation=gdn.activation, seq_idx=None)
    else:
        mixed = F.silu(gdn.conv1d(mixed)[:, :, :length])
    q, k, v = torch.split(mixed.transpose(1, 2), [gdn.key_dim, gdn.key_dim, gdn.value_dim], -1)
    q = q.reshape(batch, length, -1, gdn.head_k_dim)
    k = k.reshape(batch, length, -1, gdn.head_k_dim)
    v = v.reshape(batch, length, -1, gdn.head_v_dim)
    # A1: identical adapter call and relation as geoaddress_gdn (uncached path).
    q, k = gdn.geoaddress(hidden, q, k, image_mask)
    relation_loss = relation(q, k)
    beta = b.sigmoid()
    g = -gdn.A_log.float().exp() * F.softplus(a.float() + gdn.dt_bias)
    repeat = gdn.num_v_heads // gdn.num_k_heads
    if repeat > 1:
        q = q.repeat_interleave(repeat, dim=2)
        k = k.repeat_interleave(repeat, dim=2)
    # G3: identical probe insertion, single kernel call and head loss as native_state_gdn.
    if len(anchors):
        queries = head.queries(q[0, anchors], lags)
        prepared, real, probes = insert_probes((q, k, v, g, beta), anchors.repeat_interleave(5), queries)
        pq, pk, pv, pg, pb = prepared
        out, _ = kernel(pq, pk, pv, g=pg, beta=pb, initial_state=None, output_final_state=False,
                        use_qk_l2norm_in_kernel=True, cu_seqlens=None)
        state_loss = head.loss(out[0, probes].reshape(len(anchors), 5, gdn.num_v_heads, gdn.head_v_dim), targets)
        out = out[:, real]
    else:
        out, _ = kernel(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False,
                        use_qk_l2norm_in_kernel=True, cu_seqlens=None)
        state_loss = head.empty_loss(hidden)
    out = gdn.norm(out.reshape(-1, gdn.head_v_dim), z.reshape(-1, gdn.head_v_dim))
    losses = torch.stack((state_loss.float().reshape(()), relation_loss.float().reshape(())))
    return gdn.out_proj(out.reshape(batch, length, -1)), losses


def address_state_decoder(layer, hidden, head, anchors, lags, targets, image_mask, relation):
    if layer.layer_type != 'linear_attention':
        raise ValueError('G3A1 selected layer must be GDN')
    residual = hidden
    output, losses = address_state_gdn(layer.linear_attn, layer.input_layernorm(hidden), head, anchors, lags,
                                       targets, image_mask, relation)
    hidden = residual + output
    return hidden + layer.mlp(layer.post_attention_layernorm(hidden)), losses
