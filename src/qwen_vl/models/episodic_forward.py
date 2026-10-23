"""Project language forward for ``episodic_window_v1`` (same native parameters).

It reproduces ``Qwen3_5TextModel.forward`` of the pinned Transformers 5.11.0
for an unpadded single episode: identical embeddings, 3-axis MRoPE handling,
causal/linear masks, decoder-layer calls (including native gradient
checkpointing and activation offload) and final norm.  The only additions:

* before each scoring GDN layer ``l-1`` (offline, write-score arms), a pure
  ``torch.no_grad`` side pass computes committed-write scores from that
  layer's input; this is outside the checkpointed call, so recomputation can
  never produce a different plan;
* for the learned GDN-sweep arms, every GDN layer of an attention layer's
  feature window gets the same kind of side pass, and the frozen ranker
  scores the cached birth-time features before that window's last layer;
* an immutable ``LayerSelectionPlan`` is passed explicitly to its consuming
  attention layer ``l`` (no global/hook state; the same object is captured by
  checkpoint recomputation).

Serving appends route visual scores into a session-owned state object.
``SOURCE_FINGERPRINT`` pins the native forward this mirrors.
"""
import hashlib
import inspect
import torch
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_5 import modeling_qwen3_5 as native
from qwen_vl.models.committed_write import CommittedWriteScorer
from qwen_vl.research.episodic_contract import GDN_SWEEP_MODE
from qwen_vl.stream.episodic import build_layer_plan
from qwen_vl.stream.gdn_selectors import OfflineRankerFeatures, build_sweep_layer_plan, offline_sweep_plan

NATIVE_FORWARD_SHA256 = 'e56d7184954e7b1fc7f3bcc0120b38ea684c620e0b568fd687f41e1bcf9816db'


def native_forward_fingerprint():
    source = inspect.getsource(native.Qwen3_5TextModel.forward)
    source += inspect.getsource(native.Qwen3_5DecoderLayer.forward)
    source += inspect.getsource(native.Qwen3_5GatedDeltaNet.forward)
    return hashlib.sha256(source.encode()).hexdigest()


def check_native_source(expected=None):
    actual = native_forward_fingerprint()
    if expected is not None and actual != expected:
        raise RuntimeError(f'Native Qwen3.5 forward changed: {actual} != {expected}')
    return actual


def _layer_kwargs(spec, plan):
    return dict(episodic_recent_groups=spec.recent_groups, episodic_plan=plan)


def episodic_text_forward(text_model, inputs_embeds, position_ids, spec, *, step_plan=None,
                          layout=None, episode_uid=None, past_key_values=None, use_cache=False,
                          stream_append=False, serving=None, window_attention_batch_steps=1,
                          diagnostics=None, state_head=None, anchors=None, lags=None,
                          targets=None, state_layer=24, state_heads=None):
    """Return last hidden state [1, T, width] under the episodic visibility rule."""
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
    plans = {}
    # The separately versioned GDN sweep only swaps the plan builder/scorer;
    # visibility, attention and every native parameter computation are shared.
    sweep = spec.version == GDN_SWEEP_MODE
    if offline and spec.budget_tokens and not spec.needs_write_scores:
        for attention in spec.attention_layers:
            plans[attention] = (build_sweep_layer_plan if sweep else build_layer_plan)(
                spec, episode_uid, attention, layout)
    visual = layout.visual_matrix().to(hidden.device) if offline and spec.needs_write_scores else None
    segments = layout.score_segments() if visual is not None else None
    # Phase-B learned arms read birth-time features of every GDN layer in each
    # attention layer's window (no_grad side passes outside the checkpointed
    # call) and score with the frozen ranker; S arms never enter this branch.
    feature_layers = spec.feature_pairs if sweep and spec.is_learned else {}
    ranker_features = (OfflineRankerFeatures(spec, layout, hidden.device)
                       if feature_layers and offline else None)
    for index, decoder_layer in enumerate(text_model.layers[:config.num_hidden_layers]):
        kind = config.layer_types[index]
        if index in feature_layers:
            attention = feature_layers[index]
            if offline:
                ranker_features.observe(index, decoder_layer, hidden.detach()[0])
                if index == spec.scoring_layer_for(attention):
                    plans[attention] = ranker_features.plan(attention, episode_uid, diagnostics)
            else:
                serving.feature_append(index, attention, decoder_layer, hidden.detach()[0])
        elif index in scorer_layers and spec.needs_write_scores:
            attention = scorer_layers[index]
            if offline and sweep:
                plans[attention] = offline_sweep_plan(spec, episode_uid, attention, decoder_layer,
                                                      hidden.detach()[0], layout, diagnostics)
            elif offline:
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


def attention_index(index, spec):
    if index not in spec.attention_layers:
        raise ValueError(f'Layer {index} is not a declared episodic attention layer')
    return index
