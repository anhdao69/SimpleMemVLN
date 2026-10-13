"""G3 native GDN layer24: pure uncached forward with training-only state reads.

Mirrors the pinned native projection/convolution/gates. Exactly one FLA call;
only its sequence is expanded. Original rows are restored before native norm,
z gating, output projection and decoder residual/MLP. No state snapshots/hooks.
"""
import torch
from torch.nn import functional as F
from qwen_vl.research.vggt_g3 import insert_probes


def native_state_gdn(gdn,hidden,head,anchors,lags,targets):
    batch,length,_=hidden.shape
    if batch!=1:raise ValueError('G3 supports one complete unpadded episode')
    kernel=gdn.chunk_gated_delta_rule
    if not (getattr(kernel,'__module__','').startswith('fla.ops.gated_delta_rule.')
            and getattr(kernel,'__name__','')=='chunk_gated_delta_rule'):
        raise RuntimeError('G3 requires the qualified native FLA chunk kernel')
    mixed=gdn.in_proj_qkv(hidden).transpose(1,2)
    z=gdn.in_proj_z(hidden).reshape(batch,length,-1,gdn.head_v_dim)
    b=gdn.in_proj_b(hidden);a=gdn.in_proj_a(hidden)
    if gdn.causal_conv1d_fn is not None:
        mixed=gdn.causal_conv1d_fn(x=mixed,weight=gdn.conv1d.weight.squeeze(1),
            bias=gdn.conv1d.bias,activation=gdn.activation,seq_idx=None)
    else:
        mixed=F.silu(gdn.conv1d(mixed)[:,:,:length])
    q,k,v=torch.split(mixed.transpose(1,2),[gdn.key_dim,gdn.key_dim,gdn.value_dim],-1)
    q=q.reshape(batch,length,-1,gdn.head_k_dim)
    k=k.reshape(batch,length,-1,gdn.head_k_dim)
    v=v.reshape(batch,length,-1,gdn.head_v_dim)
    beta=b.sigmoid();g=-gdn.A_log.float().exp()*F.softplus(a.float()+gdn.dt_bias)
    repeat=gdn.num_v_heads//gdn.num_k_heads
    if repeat>1:q=q.repeat_interleave(repeat,dim=2);k=k.repeat_interleave(repeat,dim=2)
    if len(anchors):
        queries=head.queries(q[0,anchors],lags)
        prepared,real,probes=insert_probes((q,k,v,g,beta),anchors.repeat_interleave(5),queries)
        pq,pk,pv,pg,pb=prepared
        out,_=kernel(pq,pk,pv,g=pg,beta=pb,initial_state=None,output_final_state=False,
                     use_qk_l2norm_in_kernel=True,cu_seqlens=None)
        loss=head.loss(out[0,probes].reshape(len(anchors),5,gdn.num_v_heads,gdn.head_v_dim),targets)
        out=out[:,real]
    else:
        out,_=kernel(q,k,v,g=g,beta=beta,initial_state=None,output_final_state=False,
                     use_qk_l2norm_in_kernel=True,cu_seqlens=None)
        loss=head.empty_loss(hidden)
    out=gdn.norm(out.reshape(-1,gdn.head_v_dim),z.reshape(-1,gdn.head_v_dim))
    return gdn.out_proj(out.reshape(batch,length,-1)),loss


def native_state_decoder(layer,hidden,head,anchors,lags,targets):
    if layer.layer_type!='linear_attention':raise ValueError('G3 selected layer must be GDN')
    residual=hidden
    output,loss=native_state_gdn(layer.linear_attn,layer.input_layernorm(hidden),head,anchors,lags,targets)
    hidden=residual+output
    return hidden+layer.mlp(layer.post_attention_layernorm(hidden)),loss
