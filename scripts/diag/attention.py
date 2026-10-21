"""D-c: actual post-RoPE attention inputs, exact visibility, and FP32 mass.

Uses the supported AttentionInterface registration API to wrap only the
project's registered step attention function. Native Transformers modules and
policy configuration are untouched. The native output is returned unchanged.
Mass and probability-times-V parity are measured BEFORE Qwen's output gate.
"""
from bisect import bisect_right
import hashlib
import math
import numpy as np
import torch

LAYERS=(3,7,11,15,19,23,27,31)
CATEGORIES=('prefix','recent_lag_0','recent_lag_1','recent_lag_2','recent_lag_3',
            'bank_lag_4_7','bank_lag_8_15','bank_lag_16_32','bank_lag_33_plus',
            'window8_exact_lag_4_7')
ATTENTION_NAME='simplememvln_step_attention'


def sample_queries(layout,episode_uid,seed=20261019):
    """One anchor +32 visual draws without replacement; same across all layers.

    Each episode/step gets SHA256(seed|episode_uid|step), then NumPy PCG64.
    Anchor and visual roles remain separate even if their token IDs coincide.
    """
    if layout.visual_tokens<32: raise ValueError('D-c requires at least 32 visual tokens')
    result=[]
    for step,end in enumerate(layout.observation_ends):
        text=f'{int(seed)}|{episode_uid}|{step}'
        sub_seed=int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'big')
        offsets=np.random.default_rng(sub_seed).choice(layout.visual_tokens,size=32,replace=False)
        result.append(dict(step=step,ids=[end-1]+[int(layout.visual_starts[step]+x) for x in offsets],
                           roles=['anchor']+['visual']*32,seed=seed,sub_seed=sub_seed))
    return result


def visible_key_ids(layout,step,recent=None,plan=None):
    """Exact original IDs packed by _key_index/_pack_windows before causality.

    This copies their prefix + bank + recent construction; the current group's
    future keys remain in the packed set and are causally masked below.
    `recent=None` denotes Window8, which has no episodic bank.
    """
    if not 0<=step<layout.groups: raise ValueError('Step outside layout')
    width=8 if recent is None else recent
    if type(width) is not int or width<1: raise ValueError('Invalid recent window')
    if recent is None and plan is not None: raise ValueError('Window8 cannot have a bank plan')
    first=layout.group_spans[max(0,step-width+1)][0]
    old=np.empty(0,dtype=np.int64)
    if plan is not None:
        plan.verify()
        if plan.groups!=layout.groups or plan.recent_groups!=recent:
            raise ValueError('Plan/layout window mismatch')
        old=np.asarray(plan.group_ids(step),dtype=np.int64)
        if old.ndim!=1 or (len(old)>1 and not bool(np.all(old[1:]>old[:-1]))):
            raise ValueError('Bank IDs must be sorted unique')
        if len(old) and (old[0]<layout.prefix or old[-1]>=first):
            raise ValueError('Bank IDs overlap prefix or recent window')
        # Banks contain visual rows only; reject corrupted plans before labeling.
        visuals=set(layout.visual_matrix().reshape(-1).tolist())
        if not set(old.tolist())<=visuals: raise ValueError('Nonvisual row in bank')
    end=layout.group_spans[step][1]
    ids=np.concatenate((np.arange(layout.prefix,dtype=np.int64),old,np.arange(first,end,dtype=np.int64)))
    return torch.from_numpy(ids)


def key_categories(layout,step,key_ids,recent=None,plan=None):
    starts=[x[0] for x in layout.group_spans]
    bank=set() if plan is None else set(plan.group_ids(step).tolist())
    categories=[]
    for token in key_ids.tolist():
        if token<layout.prefix:
            categories.append('prefix');continue
        source=bisect_right(starts,token)-1
        if source<0 or token>=layout.group_spans[source][1]: raise ValueError('Key outside source groups')
        lag=step-source
        if lag<0: raise ValueError('Future source group visible')
        if token in bank:
            if lag<4: raise ValueError('M5 bank contains lag below 4')
            categories.append('bank_lag_4_7' if lag<=7 else 'bank_lag_8_15' if lag<=15
                              else 'bank_lag_16_32' if lag<=32 else 'bank_lag_33_plus')
        elif lag<=3:
            categories.append(f'recent_lag_{lag}')
        elif recent is None and lag<=7:
            categories.append('window8_exact_lag_4_7')
        else:
            raise ValueError('Unclassified exact key: only Window8 or M5 R4 supported')
    return categories


@torch.no_grad()
def probabilities_and_output(query,key,value,query_ids,key_ids,scaling):
    """Inputs Q[Hq,Q,Dk], K[Hkv,K,Dk], V[Hkv,K,Dv], logical token IDs.

    Returns probabilities [Q,Hq,K] and pre-output-gate output [Q,Hq,Dv].
    Consecutive query heads share one KV head (native repeat_interleave GQA).
    Bottom-right packed causality is exactly original key_id <= query_id.
    """
    if query.ndim!=3 or key.ndim!=3 or value.ndim!=3:
        raise ValueError('Expected head-first Q/K/V rank3')
    if (key.shape[:2]!=value.shape[:2] or query.shape[2]!=key.shape[2]
            or query.shape[0]%key.shape[0] or len(query_ids)!=query.shape[1] or len(key_ids)!=key.shape[1]):
        raise ValueError('Invalid GQA dimensions or token IDs')
    if not math.isfinite(float(scaling)): raise ValueError('Nonfinite scaling')
    if any(not bool(torch.isfinite(x).all()) for x in (query,key,value)):
        raise ValueError('Nonfinite Q/K/V')
    factor=query.shape[0]//key.shape[0]
    keys=key.float().repeat_interleave(factor,dim=0)
    values=value.float().repeat_interleave(factor,dim=0)
    mask=key_ids.to(query.device)[None,:]<=query_ids.to(query.device)[:,None]
    if not bool(mask.any(-1).all()): raise ValueError('Query has no causal key')
    # Enforce FP32 dot products rather than an ambient TF32 policy. Restore the
    # previous backend setting before returning to any later native layer.
    old_tf32=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        scores=torch.matmul(query.float(),keys.transpose(-1,-2))*float(scaling)
        scores=scores.masked_fill(~mask[None],-torch.inf)
        probs=torch.softmax(scores,dim=-1,dtype=torch.float32)
        output=torch.matmul(probs,values)
    finally:
        torch.backends.cuda.matmul.allow_tf32=old_tf32
    if not bool(torch.isfinite(output).all()): raise ValueError('Nonfinite reconstructed output')
    return probs.transpose(0,1).contiguous(),output.transpose(0,1).contiguous()


def category_mass(probabilities,categories,query_ids,key_ids):
    """Per-query/per-head total mass and mass per actually visible category key.

    Empty categories have mass0 and per-token mass None, not invented zeros.
    `total_bank` is an additional aggregate, not a disjoint category.
    """
    if probabilities.ndim!=3 or probabilities.shape[2]!=len(categories):
        raise ValueError('Category/probability shape mismatch')
    if not set(categories)<=set(CATEGORIES): raise ValueError('Unknown category')
    visible=key_ids.to(probabilities.device)[None,:]<=query_ids.to(probabilities.device)[:,None]
    result={}
    for category in CATEGORIES+('total_bank',):
        selected=torch.tensor([c.startswith('bank_') if category=='total_bank' else c==category
                               for c in categories],device=probabilities.device)
        counts=(visible&selected[None]).sum(-1)
        mass=probabilities[:,:,selected].sum(-1)
        result[category]=dict(visible_tokens=counts.cpu().tolist(),mass=mass.cpu().tolist(),
            mass_per_token=[(mass[i]/count).cpu().tolist() if count else None for i,count in enumerate(counts.tolist())])
    return result


def output_gate(reconstructed,native):
    """Relative Frobenius error on each step's captured query block (<=.01).

    Per-query relative errors are also reported descriptively. No denominator
    floor is used; a nonzero estimate against an exactly zero reference fails.
    """
    if reconstructed.shape!=native.shape or reconstructed.ndim!=3:
        raise ValueError('Expected matching [queries,heads,value_dim] outputs')
    a=reconstructed.double();b=native.to(a.device).double()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()): raise ValueError('Nonfinite attention output')
    def relative(x,y):
        denominator=float(y.norm());numerator=float((x-y).norm())
        if denominator==0:
            if numerator==0:return 0.
            raise ValueError('Undefined relative output error against zero kernel output')
        return numerator/denominator
    error=relative(a,b)
    per_query=[None if float(y.norm())==0 and float((x-y).norm())!=0 else relative(x,y)
               for x,y in zip(a,b,strict=True)]
    return dict(passed=error<=.01,relative_frobenius_error=error,limit=.01,
                per_query_relative_error=per_query,
                max_query_relative_error=max((x for x in per_query if x is not None),default=None),
                undefined_query_relative_errors=sum(x is None for x in per_query),
                queries=len(a),heads=a.shape[1],value_dim=a.shape[2],
                gate_unit='step captured query block before output gate')


class AttentionCapture:
    """Single offline episode with the project's attention function wrapped.

    Enter AFTER native load_checkpoint registers the project attention function.
    Combine this context with other hooks and demand bitwise G-1 logits before
    using D-c. No concurrent model forwards or nested attention contexts.
    `raw_sink(layer,step,record)` can save actual Q/K/V and outputs to disk; it
    receives independent CPU tensors and must not retain unbounded GPU state.
    """
    def __init__(self,layout,episode_uid,*,seed=20261019,layers=LAYERS,raw_sink=None):
        self.layout=layout;self.episode_uid=episode_uid;self.layers=tuple(layers)
        self.queries=sample_queries(layout,episode_uid,seed);self.raw_sink=raw_sink
        self.records={};self.original=None;self.interface=None

    def __enter__(self):
        from transformers import AttentionInterface
        from qwen_vl.stream.window_attention import step_flash_attention
        if self.records or self.original is not None: raise RuntimeError('Capture contexts are single-use')
        original=AttentionInterface()[ATTENTION_NAME]
        if original is not step_flash_attention:
            raise RuntimeError('Expected unwrapped native project attention registration')
        self.original=original;self.interface=AttentionInterface
        AttentionInterface.register(ATTENTION_NAME,self.wrapped)
        return self

    def __exit__(self,exc_type,exc,tb):
        self.interface.register(ATTENTION_NAME,self.original)
        self.original=None
        if exc_type is None and set(self.records)!=set(self.layers):
            raise RuntimeError('Not all selected attention layers were captured')

    @torch.no_grad()
    def wrapped(self,module,query,key,value,attention_mask,**kwargs):
        if self.original is None: raise RuntimeError('Capture context is not active')
        layer=module.layer_idx
        if layer not in self.layers:
            return self.original(module,query,key,value,attention_mask,**kwargs)
        if layer in self.records: raise RuntimeError('Attention layer called more than once')
        if kwargs.get('stream_append') or attention_mask is not None or query.shape[0]!=1:
            raise ValueError('D-c capture requires unpadded offline B1')
        prefix,spans=kwargs['step_plan']
        if prefix!=self.layout.prefix or tuple(spans)!=self.layout.group_spans:
            raise ValueError('Forward token layout differs from capture layout')
        # Exactly one original call. Return its original tuple, no output clone
        # or replacement and no change to the gate applied by the native module.
        result=self.original(module,query,key,value,attention_mask,**kwargs)
        kernel=result[0]
        recent=kwargs.get('episodic_recent_groups');plan=kwargs.get('episodic_plan')
        scaling=kwargs.get('scaling',module.scaling)
        records=[];self.records[layer]=records
        for selection in self.queries:
            step=selection['step']
            ids=visible_key_ids(self.layout,step,recent,plan)
            qids=torch.tensor(selection['ids'],device=query.device)
            kids=ids.to(query.device)
            q=query[0].index_select(1,qids)
            k=key[0].index_select(1,kids);v=value[0].index_select(1,kids)
            probs,estimated=probabilities_and_output(q,k,v,qids,kids,scaling)
            actual=kernel[0].index_select(0,qids)
            gate=output_gate(estimated,actual)
            categories=key_categories(self.layout,step,ids,recent,plan)
            records.append(dict(step=step,query_ids=selection['ids'],query_roles=selection['roles'],
                sampling_seed=selection['seed'],sampling_sub_seed=selection['sub_seed'],
                visible_key_ids=ids.tolist(),key_categories=categories,
                mode='window8' if recent is None else 'episodic',gate=gate,
                categories=category_mass(probs,categories,qids,kids)))
            if self.raw_sink is not None:
                self.raw_sink(layer,step,dict(query=q.detach().cpu().clone(),key=k.detach().cpu().clone(),
                    value=v.detach().cpu().clone(),query_ids=qids.cpu(),key_ids=ids.clone(),
                    native_output=actual.detach().cpu().clone(),reconstructed_output=estimated.cpu(),
                    scaling=float(scaling),post_rope=True,before_output_gate=True))
        return result

    def gate_summary(self):
        gates=[row['gate'] for rows in self.records.values() for row in rows]
        complete=set(self.records)==set(self.layers) and all(len(rows)==self.layout.groups for rows in self.records.values())
        passed=bool(gates) and complete and all(g['passed'] for g in gates)
        return dict(passed=passed,complete=complete,episode_uid=self.episode_uid,
                    layers=len(self.records),step_blocks=len(gates),queries=sum(g['queries'] for g in gates),
                    max_relative_frobenius_error=max((g['relative_frobenius_error'] for g in gates),default=None),
                    limit=.01,episodes=1,required_gate_episodes=2,
                    coverage='single episode only; caller must validate two panel episodes and G-1')
