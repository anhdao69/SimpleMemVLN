"""R1 uncached training: native GDN once, original lane once, lane-only probes."""
import torch
from torch.nn import functional as F
from qwen_vl.research.vggt_g3 import insert_probes
from qwen_vl.models.parallel_step_lane import _finite


def lane_forward_with_queries(lane,x,roles,head,context,anchors,lags,targets):
    spec=lane.spec
    if x.ndim!=3 or x.shape[0]!=1 or roles.shape!=x.shape[:2] or roles.dtype!=torch.uint8:
        raise ValueError('R1 requires original B1 lane roles')
    if roles.device!=x.device or roles.gt(3).any() or spec.clock_mode!='step_end':
        raise ValueError('R1 requires valid step-end lane roles')
    if len(anchors) and (anchors.min()<0 or anchors.max()>=x.shape[1] or roles[0,anchors].eq(3).any()):
        raise ValueError('R1 requires pre-action anchors, before STEP_END')
    if not len(anchors):return lane(x,roles)[0],head.empty_loss(x)
    _finite(x,'R1 input')
    batch,length,_=x.shape
    kw=spec.num_key_heads*spec.key_dim;vw=spec.num_value_heads*spec.value_dim
    q,k,v=F.silu(lane.in_proj_qkv(x)).split((kw,kw,vw),dim=-1)
    repeat=spec.num_value_heads//spec.num_key_heads
    q=q.reshape(batch,length,spec.num_key_heads,spec.key_dim).repeat_interleave(repeat,dim=2)
    k=k.reshape(batch,length,spec.num_key_heads,spec.key_dim).repeat_interleave(repeat,dim=2)
    v=v.reshape(batch,length,spec.num_value_heads,spec.value_dim)
    z=lane.in_proj_z(x).reshape(batch,length,spec.num_value_heads,spec.value_dim)
    g_raw=-lane.A_log.float().exp()*F.softplus(lane.in_proj_a(x).float()+lane.dt_bias.float())
    beta_raw=lane.in_proj_b(x).sigmoid()
    write=roles.eq(3)[...,None]
    g=torch.where(write,g_raw,0.);beta=torch.where(write,beta_raw,0.)
    queries=head.queries(context,lags)
    prepared,real,probes=insert_probes((q,k,v,g,beta),anchors.repeat_interleave(5),queries)
    pq,pk,pv,pg,pb=prepared
    read,_=lane._chunk_kernel(q=pq,k=pk,v=pv,g=pg,beta=pb,scale=spec.key_dim**-.5,
        initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=True)
    loss=head.loss(read[0,probes].reshape(len(anchors),5,spec.num_value_heads,spec.value_dim),targets)
    read=read[:,real].float()
    normalized=read*torch.rsqrt(read.square().mean(-1,keepdim=True)+spec.eps)
    gated=normalized*lane.norm.weight.float()*F.silu(z.float())
    output=lane.out_proj(gated.to(v.dtype).reshape(batch,length,vw))
    _finite(output,'R1 output');_finite(loss,'R1 auxiliary loss')
    return output,loss


def native_output_and_context(gdn,hidden,anchors):
    """Pinned native uncached preparation; never invoke its installed lane wrapper."""
    batch,length,_=hidden.shape
    kernel=gdn.chunk_gated_delta_rule
    if not (getattr(kernel,'__module__','').startswith('fla.ops.gated_delta_rule.')
            and getattr(kernel,'__name__','')=='chunk_gated_delta_rule'):
        raise RuntimeError('R1 requires the qualified native FLA kernel')
    mixed=gdn.in_proj_qkv(hidden).transpose(1,2)
    z=gdn.in_proj_z(hidden).reshape(batch,length,-1,gdn.head_v_dim)
    b=gdn.in_proj_b(hidden);a=gdn.in_proj_a(hidden)
    if gdn.causal_conv1d_fn is not None:
        mixed=gdn.causal_conv1d_fn(x=mixed,weight=gdn.conv1d.weight.squeeze(1),
            bias=gdn.conv1d.bias,activation=gdn.activation,seq_idx=None)
    else:mixed=F.silu(gdn.conv1d(mixed)[:,:,:length])
    q,k,v=torch.split(mixed.transpose(1,2),[gdn.key_dim,gdn.key_dim,gdn.value_dim],-1)
    q=q.reshape(batch,length,-1,gdn.head_k_dim);k=k.reshape(batch,length,-1,gdn.head_k_dim)
    v=v.reshape(batch,length,-1,gdn.head_v_dim)
    beta=b.sigmoid();g=-gdn.A_log.float().exp()*F.softplus(a.float()+gdn.dt_bias)
    repeat=gdn.num_v_heads//gdn.num_k_heads
    if repeat>1:q=q.repeat_interleave(repeat,dim=2);k=k.repeat_interleave(repeat,dim=2)
    context=q[0,anchors].detach()
    out,_=kernel(q,k,v,g=g,beta=beta,initial_state=None,output_final_state=False,
                 use_qk_l2norm_in_kernel=True,cu_seqlens=None)
    out=gdn.norm(out.reshape(-1,gdn.head_v_dim),z.reshape(-1,gdn.head_v_dim))
    return gdn.out_proj(out.reshape(batch,length,-1)),context


def lane_state_decoder(layer,hidden,roles,head,anchors,lags,targets):
    if layer.layer_type!='linear_attention' or not hasattr(layer.linear_attn,'step_lane'):
        raise ValueError('R1 requires an installed native lane at decoder24')
    x=layer.input_layernorm(hidden)
    native,context=native_output_and_context(layer.linear_attn,x,anchors)
    lane,loss=lane_forward_with_queries(layer.linear_attn.step_lane,x,roles,head,context,anchors,lags,targets)
    hidden=hidden+(native+lane)
    return hidden+layer.mlp(layer.post_attention_layernorm(hidden)),loss
