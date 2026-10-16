"""Functional native Window8 decoder loop with one checkpoint-local G1 loss.

Same native modules/parameters. No persistent hooks, hidden taps, cached graph or
auxiliary token rows. Only the hidden tensor is a checkpoint/offload input.
"""
import torch
from transformers.masking_utils import create_causal_mask


def research_text_forward(text_model, embeddings, position_ids, step_plan, *,
                          projector=None, visual_positions=None, targets=None,
                          alignment_layer=23, window_attention_batch_steps=1,
                          state_head=None,anchors=None,lags=None, image_mask=None,address_relation=None):
    if embeddings.shape[0] != 1 or position_ids.shape[0] != 4:
        raise ValueError('Research forward needs one unpadded episode with 4-axis positions')
    if projector is not None and not 0 <= alignment_layer < len(text_model.layers):
        raise ValueError('Alignment layer outside decoder')
    logical, mrope = position_ids[0], position_ids[1:]
    causal = create_causal_mask(config=text_model.config, inputs_embeds=embeddings,
        attention_mask=None,past_key_values=None,position_ids=logical)
    linear = text_model._update_linear_attn_mask(None,None)
    rope = text_model.rotary_emb(embeddings,mrope)
    if state_head is not None and image_mask is not None and text_model._geoaddress_spec[0] != 24:
        raise ValueError('G3A1 requires GeoAddress and the G3 state read at the same decoder24')
    hidden = embeddings
    aux = hidden.new_zeros((),dtype=torch.float32)
    for index, layer in enumerate(text_model.layers[:text_model.config.num_hidden_layers]):
        kwargs = dict(position_embeddings=rope,
            attention_mask=linear if text_model.config.layer_types[index]=='linear_attention' else causal,
            position_ids=logical,past_key_values=None,use_cache=False,step_plan=step_plan,
            stream_append=False,window_attention_batch_steps=window_attention_batch_steps)
        selected = projector is not None and index == alignment_layer
        selected_state = state_head is not None and index==24
        selected_address = image_mask is not None and index == text_model._geoaddress_spec[0]
        def run(x, layer=layer, kwargs=kwargs, selected=selected, selected_state=selected_state, selected_address=selected_address):
            if selected_address and selected_state:
                # G3A1: one fused layer24 forward; returns objective sums [G3, A1].
                from qwen_vl.models.vggt_g3a1_forward import address_state_decoder
                return address_state_decoder(layer,x,state_head,anchors,lags,targets,image_mask,address_relation)
            if selected_address:
                from qwen_vl.models.geoaddress_runtime import geoaddress_decoder
                return geoaddress_decoder(layer,x,image_mask,address_relation)
            if selected_state:
                from qwen_vl.models.vggt_g3_forward import native_state_decoder
                return native_state_decoder(layer,x,state_head,anchors,lags,targets)
            out = layer.forward(x,**kwargs)  # bypass __call__'s automatic checkpoint wrapper
            loss = projector(out[0,visual_positions],targets) if selected else out.new_zeros((),dtype=torch.float32)
            return out, loss
        if text_model.gradient_checkpointing and text_model.training:
            hidden, loss = layer._gradient_checkpointing_func(run,hidden)
        else:
            hidden, loss = run(hidden)
        aux = aux + loss
    return text_model.norm(hidden), aux
