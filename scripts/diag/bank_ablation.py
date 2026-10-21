"""D-b immutable bank interventions and an explicit copied forward adapter.

No native monkeypatching. Targeted conditions freeze ALL baseline layer plans;
only the specified query group's lower banks change. Global bank-off recomputes
scores under changed hidden states, sets lower plans to B0, and leaves R4 intact.

Matching is fixed before outcomes: sample up to four eligible anchors, then up
to two C-lag resident frames per anchor with RNG seed20261019. A control must be
a DIFFERENT C-lag frame with enough tokens at EACH lower layer to match removals.
Choose the frame with minimum sum of absolute per-layer count differences, ties
by source index. Randomly subsample control IDs per layer to exact target counts.
Thus share matching is nearest feasible, not an unreported arbitrary threshold.
Insufficient anchors/sources/matches are recorded as unavailable, never replaced
using downstream measurements. Inputs are a single episode; callers should pass
an episode-specific deterministic seed or reuse the documented fixed seed.

Native generative exports yield vocabulary distributions, not four semantic
action probabilities. logit_kl is explicitly token-distribution KL at teacher-
forced response decision positions. Semantic action-distribution KL is unavailable.
"""
from dataclasses import replace
import math
import random
import numpy as np

LOWER=(3,7,11,15,19,23)


def rewrite_plan(plan,groups,budget=None):
    from qwen_vl.stream.episodic import plan_digest
    plan.verify()
    if len(groups)!=plan.groups:raise ValueError('Query-group count changed')
    old={int(i):int(s) for i,s in zip(plan.token_ids,plan.token_sources)}
    chunks=[]
    for t,ids in enumerate(groups):
        ids=np.asarray(ids,dtype=np.int64)
        if len(set(map(int,ids)))!=len(ids) or not set(map(int,ids))<=set(map(int,plan.group_ids(t))):
            raise ValueError('Intervention may only remove original bank IDs')
        chunks.append(np.sort(ids))
    offsets=np.cumsum([0]+list(map(len,chunks))).astype(np.int64)
    ids=np.concatenate(chunks) if chunks else np.empty(0,np.int64)
    sources=np.array([old[int(i)] for i in ids],dtype=np.int64)
    margins=plan.cutoff_margins.copy()
    for t in range(plan.groups):
        if not np.array_equal(chunks[t],plan.group_ids(t)):margins[t]=np.nan
    for a in (offsets,ids,sources,margins):a.setflags(write=False)
    budget=plan.budget_tokens if budget is None else budget
    return replace(plan,group_offsets=offsets,token_ids=ids,token_sources=sources,
                   cutoff_margins=margins,budget_tokens=budget,
                   source_digest=plan_digest(plan.layer_idx,plan.recent_groups,budget,offsets,ids)).verify()


def remove_bank(plan):
    return rewrite_plan(plan,[[] for _ in range(plan.groups)],budget=0)


def global_bank_off(plans):
    return {l:remove_bank(p) if l in LOWER else p for l,p in plans.items()}


def apply_removals(plans,anchor,removals):
    result=dict(plans)
    for layer,ids in removals.items():
        layer=int(layer)
        if layer not in LOWER:raise ValueError('Only layers below 24 may be ablated')
        p=plans[layer];p.verify()
        if not 0<=anchor<p.groups:raise ValueError('Anchor outside plan')
        remove=set(map(int,ids))
        if not remove<=set(map(int,p.group_ids(anchor))):raise ValueError('Removal IDs not resident')
        groups=[p.group_ids(t) for t in range(p.groups)]
        groups[anchor]=[i for i in groups[anchor] if int(i) not in remove]
        result[layer]=rewrite_plan(p,groups)
    return result


def residents(plans,anchor):
    frames={}
    for layer,p in sorted(plans.items()):
        if layer not in LOWER:continue
        p.verify()
        if not 0<=anchor<p.groups:raise ValueError('Anchor outside plan')
        lo,hi=p.group_offsets[anchor:anchor+2]
        for token,source in zip(p.token_ids[lo:hi],p.token_sources[lo:hi]):
            source=int(source)
            if 8<=anchor-source<=32:frames.setdefault(source,{}).setdefault(layer,[]).append(int(token))
    return frames


def match_target_control(plans,anchor,source,seed=20261019):
    frames=residents(plans,anchor);target=frames.get(source)
    out={'anchor':anchor,'source':source,'available':False}
    if not target:return dict(out,reason='target_not_resident_in_lag_C')
    candidates=[]
    for other,counts in frames.items():
        if other==source:continue
        if all(len(counts.get(l,[]))>=len(v) for l,v in target.items()):
            distance=sum(abs(len(target.get(l,[]))-len(counts.get(l,[]))) for l in LOWER)
            candidates.append((distance,other))
    if not candidates:return dict(out,reason='no_different_C_source_with_per_layer_capacity')
    distance,other=min(candidates);rng=random.Random(seed)
    control={l:sorted(rng.sample(frames[other][l],len(ids))) for l,ids in sorted(target.items())}
    return dict(out,available=True,control_source=other,target_ids=target,control_ids=control,
                share_count_l1_distance=distance,removed_tokens=sum(map(len,target.values())),
                match_definition='nearest feasible per-layer count L1; exact per-layer removal counts')


def choose_pairs(plans,seed=20261019):
    if not plans:raise ValueError('Baseline plans required')
    groups={p.groups for p in plans.values()}
    if len(groups)!=1:raise ValueError('Plans have unequal episode lengths')
    rng=random.Random(seed)
    anchors=[t for t in range(next(iter(groups))) if residents(plans,t)]
    rng.shuffle(anchors);anchors=sorted(anchors[:4]);pairs=[];unavailable=[]
    if len(anchors)<4:unavailable.append({'reason':'fewer_than_four_eligible_anchors','available':len(anchors)})
    for t in anchors:
        sources=sorted(residents(plans,t));rng.shuffle(sources)
        if len(sources)<2:unavailable.append({'anchor':t,'reason':'fewer_than_two_C_sources','available':len(sources)})
        for source in sources[:2]:pairs.append(match_target_control(plans,t,source,rng.randrange(2**63)))
    return {'seed':seed,'anchors':anchors,'pairs':pairs,'unavailable':unavailable,
            'selection_uses_only_baseline_plans':True}


def logit_kl(original,ablated):
    a=np.asarray(original,dtype=np.float64);b=np.asarray(ablated,dtype=np.float64)
    if a.shape!=b.shape or a.ndim!=2 or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Finite aligned decision logits required')
    def logs(x):
        x=x-x.max(axis=-1,keepdims=True)
        return x-np.log(np.exp(x).sum(axis=-1,keepdims=True))
    la,lb=logs(a),logs(b);values=np.sum(np.exp(la)*(la-lb),axis=-1)
    return {'mean':float(values.mean()),'per_decision':values.tolist(),'n_decisions':len(values),
            'definition':'KL(original vocabulary distribution || ablated vocabulary distribution) at teacher-forced response token decision positions',
            'semantic_action_distribution_kl':None,'units':'nats'}


def validate_retained(plans,spec,layout):
    if set(plans)!=set(spec.attention_layers):raise ValueError('Retain every baseline attention plan')
    for l,p in plans.items():
        p.verify()
        if p.layer_idx!=l or p.groups!=layout.groups or p.recent_groups!=spec.recent_groups:
            raise ValueError('Retained plan layout/spec mismatch')


def prefix_plan(plan,groups):
    """Keep the first complete query groups with unchanged logical token IDs."""
    from qwen_vl.stream.episodic import plan_digest
    plan.verify()
    if type(groups) is not int or not 1<=groups<=plan.groups:raise ValueError('Invalid prefix group count')
    end=int(plan.group_offsets[groups])
    offsets=plan.group_offsets[:groups+1].copy();ids=plan.token_ids[:end].copy()
    sources=plan.token_sources[:end].copy();margins=plan.cutoff_margins[:groups].copy()
    for array in (offsets,ids,sources,margins):array.setflags(write=False)
    return replace(plan,group_offsets=offsets,token_ids=ids,token_sources=sources,cutoff_margins=margins,
        source_digest=plan_digest(plan.layer_idx,plan.recent_groups,plan.budget_tokens,offsets,ids)).verify()


def prefix_layout(layout,anchor):
    """Include the complete anchor group; state snapshot still precedes action.

    Keeping its response suffix is computational only and preserves the native
    complete-group layout contract. No response token can affect anchor_t.
    """
    if type(anchor) is not int or not 0<=anchor<layout.groups:raise ValueError('Anchor outside layout')
    count=anchor+1
    return replace(layout,group_spans=layout.group_spans[:count],
                   observation_ends=layout.observation_ends[:count],visual_starts=layout.visual_starts[:count])


def prepare_model_inputs(model,data):
    """Cache native embeddings and original logical/mRoPE positions once/episode."""
    import torch
    from qwen_vl.stream.positions import PositionLedger
    if model.training or model.episodic_spec is None or getattr(model,'step_lane_spec',None) is not None:
        raise ValueError('Eval-mode episodic model without step lane required')
    with torch.inference_mode():
        positions=PositionLedger().append(model.backbone.model,data['input_ids'],data['mm_token_type_ids'],data['image_grid_thw'])
        embeddings=model.embed(data['input_ids'],data['pixel_values'],data['image_grid_thw'])
    return dict(embeddings=embeddings,positions=positions,episode_uid=data['episode_uid'])


def prefix_layer24_input(model,prepared,layout,plans,anchor):
    """Run unchanged decoder layers0..23, returning their unnormalized output.

    Original positions/IDs and every causal visible key are preserved. This is
    the input to GDN24, sufficient for exact recurrence reconstruction; no
    logits or layer24+ hidden outputs are claimed by this shortened forward.
    """
    import torch
    shortened=prefix_layout(layout,anchor)
    retained={layer:prefix_plan(plan,shortened.groups) for layer,plan in plans.items()}
    end=shortened.length;capture={};diagnostics={}
    with torch.inference_mode():
        hidden=episodic_text_forward(model.backbone.model.language_model,
            prepared['embeddings'][:,:end],prepared['positions'][...,:end],model.episodic_spec,
            step_plan=(shortened.prefix,shortened.group_spans),layout=shortened,
            episode_uid=prepared['episode_uid'],retained_plans=retained,capture=capture,
            diagnostics=diagnostics,stop_before_layer=24,
            window_attention_batch_steps=model.navigation_config['runtime'].get('window_attention_batch_steps',1))
    return dict(hidden=hidden.detach().cpu()[0].clone(),layout=shortened,diagnostics=diagnostics,
                prefix_tokens=end,anchor_token_id=shortened.observation_ends[-1]-1,
                includes_anchor_group_response_for_layout_only=True)


def forward_model(model,data,*,retained_plans=None,bank_off_layers=(),capture_layer24=True,prepared=None):
    """Effective adapter through native embeddings/head and copied text forward.

    Full episode retained for action KL; no misleading prefix-only model logits.
    Layer24 input is CPU detached for native recurrence reconstruction downstream.
    Requires external G-1/G-2/G-3 gates; this helper does not assert they passed.
    """
    import torch
    from qwen_vl.stream.positions import PositionLedger
    if model.training or model.episodic_spec is None or getattr(model,'step_lane_spec',None) is not None:
        raise ValueError('Eval-mode episodic model without step lane required')
    with torch.inference_mode():
        prepared=prepare_model_inputs(model,data) if prepared is None else prepared
        if prepared['episode_uid']!=data['episode_uid']:raise ValueError('Prepared episode identity mismatch')
        pos,embeds=prepared['positions'],prepared['embeddings']
        diagnostics={};capture={} if capture_layer24 else None
        hidden=episodic_text_forward(model.backbone.model.language_model,embeds,pos,model.episodic_spec,
            step_plan=data['step_plan'],layout=data['episodic_layout'],episode_uid=data['episode_uid'],
            diagnostics=diagnostics,retained_plans=retained_plans,bank_off_layers=bank_off_layers,capture=capture,
            window_attention_batch_steps=model.navigation_config['runtime'].get('window_attention_batch_steps',1))
        if model.output_mode=='candidate_logits':logits=model.action_logits(hidden[0,data['read_positions']])
        elif model.output_mode=='classification':logits=model.classifier(hidden[0,data['read_positions']])
        else:logits=model.backbone.lm_head(hidden[0,data['response_target_positions']-1])
    return {'logits':logits.detach().float().cpu().numpy(),'diagnostics':diagnostics,'capture':capture,
            'output_mode':model.output_mode,'effective_forward':'scripts.diag.bank_ablation.episodic_text_forward'}


def episodic_text_forward(text_model, inputs_embeds, position_ids, spec, *, step_plan=None,
                          layout=None, episode_uid=None, past_key_values=None, use_cache=False,
                          stream_append=False, serving=None, window_attention_batch_steps=1,
                          diagnostics=None, state_head=None, anchors=None, lags=None,
                          targets=None, state_layer=24, state_heads=None,
                          retained_plans=None, bank_off_layers=(), capture=None, stop_before_layer=None):
    """Return last hidden state [1, T, width] under the episodic visibility rule."""
    # Copied native project forward; additive plan injection/capture only.
    import torch
    from transformers.masking_utils import create_causal_mask
    from qwen_vl.models.committed_write import CommittedWriteScorer
    from qwen_vl.stream.episodic import build_layer_plan
    from qwen_vl.models.episodic_forward import _layer_kwargs, attention_index
    if stream_append or use_cache or past_key_values is not None:
        raise ValueError('Diagnostic copy is offline only')
    if retained_plans is not None:
        validate_retained(retained_plans, spec, layout)
    if not set(bank_off_layers) <= set(spec.attention_layers):
        raise ValueError('Unknown bank-off layers')
    if stop_before_layer not in (None,24):raise ValueError('Only input24 truncation is supported')
    if stop_before_layer is not None and any(x is not None for x in (state_head,state_heads)):
        raise ValueError('Prefix adapter supports serving forwards only')
    config = text_model.config
    if inputs_embeds.shape[0] != 1:
        raise ValueError('Episodic forward requires one unpadded episode')
    if position_ids.ndim != 3 or position_ids.shape[0] != 4:
        raise ValueError('Expected logical + 3-axis multimodal positions')
    text_position_ids, mrope_position_ids = position_ids[0], position_ids[1:]
    offline = not stream_append
    if state_heads is not None and state_head is not None:
        raise ValueError('Specify one head API only')
    heads=({str(state_layer):state_head} if state_head is not None else state_heads)
    if heads is not None:
        if not heads or any(not isinstance(i,str) or not i.isdigit() or str(int(i))!=i for i in heads):
            raise ValueError('State heads require canonical integer layer keys')
        if not offline or use_cache or past_key_values is not None:
            raise ValueError('State supervision requires uncached episodic training')
        if anchors is None or lags is None or targets is None or len(anchors)!=len(lags) or len(anchors)!=len(targets):
            raise ValueError('State supervision requires matching anchors, lags and targets')
        if any(not 0<=int(i)<config.num_hidden_layers or config.layer_types[int(i)]!='linear_attention' for i in heads):
            raise ValueError('State supervision requires a native GDN layer')
    elif any(x is not None for x in (anchors,lags,targets)):
        raise ValueError('State metadata requires a state head')
    if offline:
        if past_key_values is not None or use_cache or layout is None or step_plan is None:
            raise ValueError('Offline episodic forward is uncached and needs a layout')
        if (layout.prefix, layout.group_spans) != (step_plan[0], tuple(step_plan[1])):
            raise ValueError('Layout and step plan disagree')
    elif serving is None or past_key_values is None:
        raise ValueError('Serving append needs session episodic state and cache')

    causal_mask = create_causal_mask(
        config=config, inputs_embeds=inputs_embeds, attention_mask=None,
        past_key_values=past_key_values, position_ids=text_position_ids)
    linear_mask = text_model._update_linear_attn_mask(None, past_key_values)
    hidden = inputs_embeds
    aux = hidden.new_zeros((),dtype=torch.float32)
    position_embeddings = text_model.rotary_emb(hidden, mrope_position_ids)

    scorer_layers = {gdn: attention for gdn, attention in spec.scoring_pairs}
    plans = {} if retained_plans is None else dict(retained_plans)
    if retained_plans is None and offline and spec.budget_tokens and not spec.needs_write_scores:
        for attention in spec.attention_layers:
            plans[attention] = build_layer_plan(spec, episode_uid, attention, layout)
    visual = layout.visual_matrix().to(hidden.device) if offline and spec.needs_write_scores else None
    segments = layout.score_segments() if visual is not None else None
    for index, decoder_layer in enumerate(text_model.layers[:config.num_hidden_layers]):
        if index == 24 and capture is not None:
            capture['layer24_input'] = hidden.detach().cpu().clone()
        if index == stop_before_layer:
            if diagnostics is not None:diagnostics['plans']=plans
            return hidden
        kind = config.layer_types[index]
        if retained_plans is None and index in scorer_layers and spec.needs_write_scores:
            attention = scorer_layers[index]
            if offline:
                scorer = CommittedWriteScorer(decoder_layer, hidden.device)
                scores = scorer.score_sequence(hidden.detach()[0], segments)
                visual_scores = scores[visual]                        # [G, 300] FP32
                plans[attention] = build_layer_plan(spec, episode_uid, attention, layout,
                                                    visual_scores.cpu())
                if diagnostics is not None:
                    diagnostics.setdefault('visual_scores', {})[attention] = visual_scores.cpu()
                del scorer, scores
            else:
                serving.score_append(index, attention, decoder_layer, hidden.detach()[0])
        layer_kwargs = {}
        if kind == 'full_attention' and offline:
            if index in bank_off_layers:
                plans[index] = remove_bank(plans[index])
            layer_kwargs = _layer_kwargs(spec, plans.get(attention_index(index, spec)))
        if heads is not None and str(index) in heads:
            from qwen_vl.models.vggt_g3_forward import native_state_decoder
            # Bound layer default is essential: checkpoint recomputes after
            # the loop has advanced. Scoring/plans remain outside checkpoint.
            def run_state(x,layer=decoder_layer,head=heads[str(index)]):
                return native_state_decoder(layer,x,head,anchors,lags,targets,uniform_empty=True)
            if text_model.gradient_checkpointing and text_model.training:
                hidden,layer_aux=decoder_layer._gradient_checkpointing_func(run_state,hidden)
            else:hidden,layer_aux=run_state(hidden)
            aux=aux+layer_aux/len(heads)
            continue
        hidden = decoder_layer(
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=linear_mask if kind == 'linear_attention' else causal_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            step_plan=step_plan,
            stream_append=stream_append,
            window_attention_batch_steps=window_attention_batch_steps,
            **layer_kwargs,
        )
    if diagnostics is not None:
        diagnostics['plans'] = plans
    hidden=text_model.norm(hidden)
    return (hidden,aux) if heads is not None else hidden
