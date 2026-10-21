"""Transparent layer-input capture, block64 GDN reconstruction, and gate G-3.

State orientation is [value_head, key_dim, value_dim]. Native keys have key
heads, repeated with repeat_interleave only inside the recurrence. No policy
forward, tensor, module, parameter, cache, or scientific source is modified.
"""
import torch
from qwen_vl.models.committed_write import CommittedWriteScorer, _scan_blocks, internal_blocks
from qwen_vl.data.episodic_metadata import build_episodic_layout
from qwen_vl.research.vggt_g3 import observation_anchors

LAYERS=(8,16,22,24,26)


class LayerInputCapture:
    """One offline forward; CPU clones of [T,width] decoder inputs.

    Usage: ``with LayerInputCapture(model.backbone.model.language_model.layers)
    as capture: output=model(**encoded)``. Hooks return None. Repeated calls to
    a selected layer are rejected to avoid accidentally mixing streaming or
    recomputation traces. Use a fresh context for each episode.
    """
    def __init__(self,layers,indices=LAYERS):
        self.layers=layers; self.indices=tuple(indices); self.hidden={}; self.handles=[]
        if len(set(self.indices))!=len(self.indices): raise ValueError('Duplicate capture layer')

    def __enter__(self):
        if self.handles or self.hidden: raise RuntimeError('Capture contexts are single-use')
        try:
            for index in self.indices:
                def capture(module,args,kwargs,index=index):
                    value=args[0] if args else kwargs.get('hidden_states')
                    if value is None or value.ndim!=3 or value.shape[0]!=1:
                        raise ValueError('Expected one complete [1,tokens,width] layer input')
                    if index in self.hidden: raise RuntimeError(f'Layer {index} called more than once')
                    self.hidden[index]=value[0].detach().to('cpu',copy=True)
                self.handles.append(self.layers[index].register_forward_pre_hook(capture,with_kwargs=True))
        except BaseException:
            for handle in self.handles: handle.remove()
            self.handles=[]
            raise
        return self

    def __exit__(self,exc_type,exc,tb):
        for handle in self.handles: handle.remove()
        self.handles=[]
        if exc_type is None and set(self.hidden)!=set(self.indices):
            raise RuntimeError('Not all selected layers executed')


def layout_from_encoded(data,image_token_id):
    """Derive identical token geometry for M5 or Window8, without re-encoding."""
    prefix,spans=data['step_plan']
    anchors=observation_anchors(data['response_target_positions'],data['response_action_index'],data['num_actions'])
    obs=tuple((int(start),int(anchor)+1) for (start,_),anchor in zip(spans,anchors,strict=True))
    derived=build_episodic_layout(data['input_ids'],image_token_id,int(prefix),obs,spans)
    existing=data.get('episodic_layout')
    if existing is not None and existing.to_dict()!=derived.to_dict():
        raise ValueError('Existing episodic layout differs from action-target-derived layout')
    return derived


def _validate_prepared(prepared):
    key,value,g,beta=prepared
    if key.ndim!=3 or value.ndim!=3 or key.shape[0]!=value.shape[0] or not key.shape[0]:
        raise ValueError('Expected nonempty keys [T,Hk,Dk] and values [T,Hv,Dv]')
    if g.shape!=value.shape[:2] or beta.shape!=g.shape or value.shape[1]%key.shape[1]:
        raise ValueError('Gate shapes or key/value head ratio differ')
    for tensor in prepared:
        if tensor.dtype!=torch.float32 or not bool(torch.isfinite(tensor).all()):
            raise ValueError('Prepared recurrence tensors must be finite FP32')
    return value.shape[1]//key.shape[1]


@torch.no_grad()
def scan_prepared(prepared,segments,initial_state=None):
    """Synthetic/diagnostic entrypoint to the unchanged native block64 helper."""
    ratio=_validate_prepared(prepared)
    key,value,_,_=prepared
    shape=(value.shape[1],key.shape[2],value.shape[2])
    if initial_state is None: initial_state=key.new_zeros(shape)
    if tuple(initial_state.shape)!=shape or initial_state.dtype!=torch.float32:
        raise ValueError('State must be FP32 [Hv,Dk,Dv]')
    _,state=_scan_blocks(*prepared,initial_state,internal_blocks(segments),ratio)
    return state


def relative_state_error(reconstructed,native):
    """Relative Frobenius error, no denominator floor or tolerance relaxation."""
    if reconstructed.shape!=native.shape:
        raise ValueError('Reconstructed/native state shapes differ')
    a=reconstructed.detach().double(); b=native.detach().to(a.device).double()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError('Nonfinite state')
    denominator=float(b.norm()); numerator=float((a-b).norm())
    if denominator==0:
        if numerator==0: return 0.
        raise ValueError('Nonzero reconstruction against zero native state; relative error undefined')
    return numerator/denominator


@torch.no_grad()
def native_fla_state(decoder_layer,prepared):
    """Independent native FLA recurrence on the exact prepared prefix.

    The already-normalized FP32 keys/values/g/beta are passed unchanged, with
    normalization disabled. Queries equal repeated keys: queries affect outputs
    but never the final recurrent state. This checks recurrence, not independent
    fused-convolution preparation. The policy's actual FLA function is required.
    """
    ratio=_validate_prepared(prepared)
    key,value,g,beta=prepared
    fn=decoder_layer.linear_attn.chunk_gated_delta_rule
    if not getattr(fn,'__module__','').startswith('fla.'):
        raise RuntimeError('Gate G-3 requires native FLA, never a torch fallback')
    if key.device.type!='cuda': raise ValueError('Native FLA gate requires CUDA')
    repeated=key.repeat_interleave(ratio,dim=1)[None].contiguous()
    _,state=fn(repeated,repeated,value[None].contiguous(),g=g[None].contiguous(),
               beta=beta[None].contiguous(),initial_state=None,output_final_state=True,
               use_qk_l2norm_in_kernel=False)
    if state is None: raise RuntimeError('Native FLA returned no final state')
    expected=(1,value.shape[1],key.shape[2],value.shape[2])
    if tuple(state.shape)!=expected: raise ValueError('Native FLA state orientation differs')
    return state[0].float()


def _validate_layout(layout,length):
    if layout.length!=length or layout.prefix<=0: raise ValueError('Layout length/prefix mismatch')
    segments=layout.score_segments()
    cursor=0
    for start,end in segments:
        if start!=cursor or end<=start: raise ValueError('Segments must tile the full stream')
        cursor=end
    if len(segments)!=1+2*layout.groups or cursor!=length:
        raise ValueError('Expected prefix then observation/response segments')
    visual=layout.visual_matrix()
    for t,((start,end),obs) in enumerate(zip(layout.group_spans,layout.observation_ends,strict=True)):
        if not start<obs<end or not bool(((visual[t]>=start)&(visual[t]<obs)).all()):
            raise ValueError('Visual positions must lie inside observation segment')
    return segments,visual


@torch.no_grad()
def reconstruct(decoder_layer,hidden,layout,*,visual_dtype=torch.float16,check_fla=False):
    """Reconstruct one layer from captured CPU/GPU hidden [T,width].

    Returns CPU states [F,Hv,Dk,Dv] FP32 and visual_keys [F,V,Hk,Dk],
    visual_values [F,V,Hv,Dv], visual_beta [F,V,Hv] (FP16 by default), plus
    original visual IDs, anchor IDs, segmentation metadata and gate records.
    `check_fla=True` compares every observation anchor with a separate FLA call
    over the same entire prepared prefix. The gate fails at any error >.01.
    A failed gate is returned with all computed values; the caller MUST stop.
    Only G-3's two episodes should request the quadratic full-prefix checks.
    """
    if hidden.ndim!=2: raise ValueError('Captured hidden must be [tokens,width]')
    if visual_dtype not in (torch.float16,torch.float32): raise ValueError('Unsupported visual storage dtype')
    segments,visual_ids=_validate_layout(layout,len(hidden))
    parameter=next(decoder_layer.parameters())
    scorer=CommittedWriteScorer(decoder_layer,parameter.device)
    states=[]; visual_keys=[]; visual_values=[]; visual_beta=[]; gate=[]; prefix_parts=[]
    for number,(start,end) in enumerate(segments):
        segment=hidden[start:end].to(device=parameter.device,dtype=parameter.dtype)
        prepared=scorer._prepare(segment)
        _validate_prepared(prepared)
        scorer._scan(prepared,[(0,end-start)])
        if check_fla: prefix_parts.append(prepared)
        # Segment zero is prefix; odd segments are observations.
        if number%2==1:
            step=(number-1)//2
            states.append(scorer.state.detach().float().cpu().clone())
            ids=(visual_ids[step]-start).to(parameter.device)
            for destination,tensor in zip((visual_keys,visual_values,visual_beta),
                                           (prepared[0],prepared[1],prepared[3]),strict=True):
                destination.append(tensor[ids].to(device='cpu',dtype=visual_dtype))
            if check_fla:
                full=tuple(torch.cat([part[i] for part in prefix_parts],dim=0) for i in range(4))
                native=native_fla_state(decoder_layer,full)
                error=relative_state_error(scorer.state,native)
                gate.append(dict(step=step,anchor_token_id=end-1,prefix_tokens=end,
                                 relative_frobenius_error=error,limit=.01,passed=error<=.01))
                del full,native
    result=dict(states=torch.stack(states),visual_keys=torch.stack(visual_keys),
                visual_values=torch.stack(visual_values),visual_beta=torch.stack(visual_beta),
                visual_token_ids=visual_ids.cpu(),anchor_token_ids=[x-1 for x in layout.observation_ends],
                segments=[list(x) for x in segments],block_tokens=64,state_orientation='Hv,Dk,Dv',
                key_head_repetition='repeat_interleave',gate_g3=dict(
                    status=('PASS' if all(x['passed'] for x in gate) else 'FAIL') if check_fla else 'NOT_RUN',
                    limit=.01,anchors=gate,
                    reference='native FLA full-prefix recurrence on identical prepared FP32 tensors',
                    excludes='independent fused-convolution preparation parity'))
    for name in ('states','visual_keys','visual_values','visual_beta'):
        if not bool(torch.isfinite(result[name]).all()): raise ValueError(f'Nonfinite retained {name}')
    return result
