"""Project language forward for ``episodic_window_v1`` (same native parameters).

It reproduces ``Qwen3_5TextModel.forward`` of the pinned Transformers 5.11.0
for an unpadded single episode: identical embeddings, 3-axis MRoPE handling,
causal/linear masks, decoder-layer calls (including native gradient
checkpointing and activation offload) and final norm.  The only additions:

* before each scoring GDN layer ``l-1`` (offline, write-score arms), a pure
  ``torch.no_grad`` side pass computes committed-write scores from that
  layer's input; this is outside the checkpointed call, so recomputation can
  never produce a different plan;
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
from qwen_vl.stream.episodic import build_layer_plan

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
                          diagnostics=None):
    """Return last hidden state [1, T, width] under the episodic visibility rule."""
    config = text_model.config
    if inputs_embeds.shape[0] != 1:
        raise ValueError('Episodic forward requires one unpadded episode')
    if position_ids.ndim != 3 or position_ids.shape[0] != 4:
        raise ValueError('Expected logical + 3-axis multimodal positions')
    text_position_ids, mrope_position_ids = position_ids[0], position_ids[1:]
    offline = not stream_append
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
    position_embeddings = text_model.rotary_emb(hidden, mrope_position_ids)

    scorer_layers = {gdn: attention for gdn, attention in spec.scoring_pairs}
    plans = {}
    if offline and spec.budget_tokens and not spec.needs_write_scores:
        for attention in spec.attention_layers:
            plans[attention] = build_layer_plan(spec, episode_uid, attention, layout)
    visual = layout.visual_matrix().to(hidden.device) if offline and spec.needs_write_scores else None
    segments = layout.score_segments() if visual is not None else None
    for index, decoder_layer in enumerate(text_model.layers[:config.num_hidden_layers]):
        kind = config.layer_types[index]
        if index in scorer_layers and spec.needs_write_scores:
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
            layer_kwargs = _layer_kwargs(spec, plans.get(attention_index(index, spec)))
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
    return text_model.norm(hidden)


def attention_index(index, spec):
    if index not in spec.attention_layers:
        raise ValueError(f'Layer {index} is not a declared episodic attention layer')
    return index
