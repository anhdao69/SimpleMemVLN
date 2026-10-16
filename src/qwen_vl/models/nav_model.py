"""Thin native Qwen bridge: frozen vision, full text graph and sparse readouts."""

import torch
from torch import nn
from qwen_vl.stream.positions import PositionLedger
from qwen_vl.stream.timing import timed


def per_action_token_loss(logits, targets, action_index, num_actions):
    losses = torch.nn.functional.cross_entropy(
        logits.float(), targets, reduction="none"
    )
    counts = torch.bincount(action_index, minlength=num_actions)
    if (counts == 0).any():
        raise ValueError("Every action must have supervised response tokens")
    return losses.new_zeros(num_actions).scatter_add(0, action_index, losses) / counts


class SimpleMemVLNForNavigation(nn.Module):
    def _apply(self, fn, recurse=True):
        # DeepSpeed calls module.bfloat16(), including non-persistent buffers.
        # Native loading recreates rotary frequencies in FP32, so rounding them
        # here silently changes training positions and breaks checkpoint reload.
        preserved = [
            (module, name, value)
            for module in self.modules()
            for name, value in module._buffers.items()
            if name in ("inv_freq", "original_inv_freq") and value is not None
        ]
        result = super()._apply(fn, recurse=recurse)
        for module, name, value in preserved:
            module._buffers[name] = value.to(
                device=module._buffers[name].device, dtype=torch.float32
            )
        return result

    def __init__(self, backbone, navigation_config, candidate_token_ids=None):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        self.navigation_config = navigation_config
        self.output_mode = navigation_config["model"]["output_mode"]
        self.backbone.model.visual.requires_grad_(False)
        self.backbone.model.visual.eval()
        self.classifier = None
        if self.output_mode == "candidate_logits":
            if (
                candidate_token_ids is None
                or len(candidate_token_ids) != 4
                or len(set(candidate_token_ids)) != 4
            ):
                raise ValueError("Four verified candidate token IDs required")
            if not all(
                0 <= i < backbone.lm_head.weight.shape[0] for i in candidate_token_ids
            ):
                raise ValueError("Candidate token outside LM vocabulary")
            self.register_buffer(
                "candidate_token_ids",
                torch.tensor(
                    candidate_token_ids,
                    dtype=torch.long,
                    device=backbone.lm_head.weight.device,
                ),
                persistent=False,
            )
            mode = navigation_config["model"].get(
                "action_head_mode", "lm_rows_trainable"
            )
            if mode == "lm_rows_frozen":
                # This also freezes input embeddings when the parameters are tied.
                backbone.lm_head.requires_grad_(False)
            elif mode == "copied_linear":
                head = backbone.lm_head
                self.classifier = nn.Linear(
                    head.weight.shape[1],
                    4,
                    bias=head.bias is not None,
                    device=head.weight.device,
                    dtype=head.weight.dtype,
                )
                with torch.no_grad():
                    self.classifier.weight.copy_(
                        head.weight.index_select(0, self.candidate_token_ids)
                    )
                    if head.bias is not None:
                        self.classifier.bias.copy_(
                            head.bias.index_select(0, self.candidate_token_ids)
                        )
                if head.weight is not backbone.model.language_model.embed_tokens.weight:
                    head.requires_grad_(False)
            elif mode != "lm_rows_trainable":
                raise ValueError("Unsupported action head mode")
        if self.output_mode == "classification":
            width = backbone.config.text_config.hidden_size
            if width != 2560:
                raise ValueError(
                    f"Expected pinned Qwen3.5-4B hidden width, got {width}"
                )
            self.classifier = nn.Linear(width, 4, bias=True).to(
                device=next(backbone.parameters()).device,
                dtype=next(backbone.parameters()).dtype,
            )
            nn.init.normal_(self.classifier.weight, std=0.02)
            nn.init.zeros_(self.classifier.bias)
            # An untied unused LM projection must not enter optimizer/DDP groups.
            if (
                backbone.lm_head.weight
                is not backbone.model.language_model.embed_tokens.weight
            ):
                backbone.lm_head.requires_grad_(False)

        from qwen_vl.research.episodic_contract import parse_episodic_spec
        self.episodic_spec = parse_episodic_spec(navigation_config, backbone.config.text_config)
        if self.episodic_spec is not None:
            from qwen_vl.models.episodic_forward import NATIVE_FORWARD_SHA256, check_native_source
            check_native_source(NATIVE_FORWARD_SHA256)
        from qwen_vl.research.lane_contract import parse_step_lane_spec
        spec = parse_step_lane_spec(navigation_config, backbone.config.text_config)
        if spec is not None:
            from qwen_vl.models.install_step_lane import install_step_lanes
            install_step_lanes(
                self, spec,
                init_seed=navigation_config["model"]["step_lane"].get("init_seed", 429),
            )

        from qwen_vl.research.vggt_g1 import parse_g1
        self.g1_spec = parse_g1(navigation_config)
        if self.g1_spec is not None:
            from qwen_vl.models.vggt_aux_heads import CurrentSpatialProjector
            from qwen_vl.models.episodic_forward import NATIVE_FORWARD_SHA256, check_native_source
            check_native_source(NATIVE_FORWARD_SHA256)
            if self.g1_spec['run_id'] in ('G3','G3_M5'):
                from qwen_vl.models.vggt_g3_heads import NativeStateHead
                text=backbone.config.text_config
                if (text.linear_num_value_heads,text.linear_key_head_dim,text.linear_value_head_dim)!=(32,128,128):
                    raise ValueError('G3 requires the qualified native 32x128 state heads')
                self.research_aux=nn.ModuleDict({'state_head':NativeStateHead()})
            else:
                self.research_aux = nn.ModuleDict({"current_projector": CurrentSpatialProjector()})
            parameter = next(backbone.parameters())
            self.research_aux.to(device=parameter.device, dtype=parameter.dtype)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.model.visual.eval()
        return self

    def action_logits(self, hidden):
        """Project only four pretrained rows; never construct vocabulary logits."""
        if self.output_mode != "candidate_logits":
            raise ValueError("Candidate readout requested for another output mode")
        if self.classifier is not None:
            return self.classifier(hidden)
        head = self.backbone.lm_head
        bias = (
            None
            if head.bias is None
            else head.bias.index_select(0, self.candidate_token_ids)
        )
        return torch.nn.functional.linear(
            hidden, head.weight.index_select(0, self.candidate_token_ids), bias
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.backbone.model.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            or {"use_reentrant": False}
        )
        threshold = self.navigation_config["training"].get(
            "activation_offload_min_tokens"
        )
        if threshold is not None:
            from functools import partial
            from qwen_vl.models.activation_offload import checkpoint_with_cpu_offload

            self.backbone.model.language_model._set_gradient_checkpointing(
                enable=True,
                gradient_checkpointing_func=partial(
                    checkpoint_with_cpu_offload,
                    min_tokens=threshold,
                    backend=self.navigation_config["training"].get(
                        "activation_offload_backend", "save_on_cpu_pinned"),
                    **(gradient_checkpointing_kwargs or {"use_reentrant": False}),
                ),
            )

    def embed(self, input_ids, pixel_values=None, image_grid_thw=None):
        embeddings = self.backbone.model.language_model.embed_tokens(input_ids)
        if pixel_values is not None:
            features = []
            offset = 0
            count = self.navigation_config["runtime"]["vision_microbatch_images"]
            with torch.no_grad(), timed(self, "vision_encoder"):
                for start in range(0, len(image_grid_thw), count):
                    grid = image_grid_thw[start : start + count]
                    patches = int(grid.prod(-1).sum())
                    output = self.backbone.model.get_image_features(
                        pixel_values[offset : offset + patches], grid
                    )
                    features.extend(output.pooler_output)
                    offset += patches
            features = torch.cat(features, dim=0).to(embeddings)
            mask = (
                input_ids.eq(self.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(embeddings)
            )
            if int(mask.sum()) != features.numel():
                raise ValueError("Image feature/token count mismatch")
            embeddings = embeddings.masked_scatter(mask, features)
        return embeddings

    def hidden(
        self,
        input_ids,
        mm_token_type_ids,
        pixel_values=None,
        image_grid_thw=None,
        step_plan=None,
        position_ids=None,
        cache=None,
        stream_append=False,
        step_lane_roles=None,
        step_lane_cache=None,
        step_lane_logical_start=None,
        episodic_layout=None,
        episode_uid=None,
        episodic_serving=None,
        episodic_diagnostics=None,
        aux_visual_positions=None,
        aux_targets=None,
        aux_anchors=None,
        aux_lags=None,
    ):
        lane_kwargs = {}
        if getattr(self, "step_lane_spec", None) is not None:
            if step_lane_roles is None:
                raise ValueError("Enabled step lane requires explicit roles")
            if cache is not None and (self.training or torch.is_grad_enabled()):
                raise ValueError("Step lane cached forward is inference-only")
            if cache is None and (step_lane_cache is not None or step_lane_logical_start is not None):
                raise ValueError("Uncached lane forward cannot use serving state")
            lane_kwargs = dict(step_lane_roles=step_lane_roles, step_lane_cache=step_lane_cache,
                               step_lane_logical_start=step_lane_logical_start)
        elif any(x is not None for x in (step_lane_roles, step_lane_cache, step_lane_logical_start)):
            raise ValueError("Lane metadata supplied to a disabled model")
        if position_ids is None:
            position_ids = PositionLedger().append(
                self.backbone.model, input_ids, mm_token_type_ids, image_grid_thw
            )
        embeddings = self.embed(input_ids, pixel_values, image_grid_thw)
        if aux_anchors is not None:
            if self.g1_spec is None or self.g1_spec['run_id'] not in ('G3','G3_M5') or cache is not None or stream_append:
                raise ValueError('G3 metadata requires uncached G3 research forward')
            if self.g1_spec['run_id']=='G3_M5':
                from qwen_vl.models.episodic_forward import episodic_text_forward
                return episodic_text_forward(self.backbone.model.language_model,embeddings,position_ids,
                    self.episodic_spec,step_plan=step_plan,layout=episodic_layout,episode_uid=episode_uid,
                    state_head=self.research_aux.state_head,anchors=aux_anchors,lags=aux_lags,targets=aux_targets,
                    diagnostics=episodic_diagnostics,
                    window_attention_batch_steps=self.navigation_config['runtime'].get('window_attention_batch_steps',1))
            from qwen_vl.models.research_forward import research_text_forward
            return research_text_forward(self.backbone.model.language_model,embeddings,position_ids,step_plan,
                state_head=self.research_aux.state_head,anchors=aux_anchors,lags=aux_lags,targets=aux_targets,
                window_attention_batch_steps=self.navigation_config['runtime'].get('window_attention_batch_steps',1))
        if aux_visual_positions is not None:
            if self.g1_spec is None or cache is not None or stream_append:
                raise ValueError("G1 metadata is only valid in uncached research forward")
            from qwen_vl.models.research_forward import research_text_forward
            return research_text_forward(
                self.backbone.model.language_model, embeddings, position_ids, step_plan,
                projector=self.research_aux.current_projector,
                visual_positions=aux_visual_positions, targets=aux_targets,
                window_attention_batch_steps=self.navigation_config['runtime'].get('window_attention_batch_steps',1))
        stage = (
            "streaming_language_model_append"
            if pixel_values is not None
            else "feedback_language_model_append"
        )
        if self.episodic_spec is not None:
            from qwen_vl.models.episodic_forward import episodic_text_forward

            with timed(self, stage):
                return episodic_text_forward(
                    self.backbone.model.language_model, embeddings, position_ids,
                    self.episodic_spec, step_plan=step_plan, layout=episodic_layout,
                    episode_uid=episode_uid, past_key_values=cache, use_cache=cache is not None,
                    stream_append=stream_append, serving=episodic_serving,
                    window_attention_batch_steps=self.navigation_config['runtime'].get(
                        'window_attention_batch_steps', 1),
                    diagnostics=episodic_diagnostics)
        if any(x is not None for x in (episodic_layout, episodic_serving, episodic_diagnostics)):
            raise ValueError("Episodic metadata supplied to a non-episodic model")
        with timed(self, stage):
            return self.backbone.model.language_model(
                inputs_embeds=embeddings,
                position_ids=position_ids,
                attention_mask=None,
                past_key_values=cache,
                use_cache=cache is not None,
                step_plan=step_plan,
                stream_append=stream_append,
                window_attention_batch_steps=self.navigation_config['runtime'].get('window_attention_batch_steps', 1),
                output_hidden_states=False,
                **lane_kwargs,
            ).last_hidden_state

    def forward(
        self,
        input_ids,
        mm_token_type_ids,
        pixel_values,
        image_grid_thw,
        step_plan,
        action_class_ids,
        read_positions,
        response_target_positions,
        response_action_index,
        num_actions,
        episode_uid=None,
        step_lane_roles=None,
        episodic_layout=None,
        episodic_diagnostics=None,
        aux_visual_positions=None,
        aux_targets=None,
        aux_anchors=None,
        aux_lags=None,
        aux_frame_count=None,
        aux_pair_count=None,
    ):
        episodic = {}
        if self.episodic_spec is not None:
            if episodic_layout is None or episode_uid is None:
                raise ValueError("Episodic training needs the serialized layout and episode UID")
            episodic = dict(episodic_layout=episodic_layout, episode_uid=episode_uid,
                            episodic_diagnostics=episodic_diagnostics)
        elif episodic_layout is not None:
            raise ValueError("Episodic layout supplied to a non-episodic model")
        if aux_anchors is not None:
            if aux_pair_count!=len(aux_anchors) or aux_lags is None or len(aux_lags)!=len(aux_anchors):
                raise ValueError('Auxiliary pair support mismatch')
            episodic.update(aux_anchors=aux_anchors,aux_lags=aux_lags,aux_targets=aux_targets)
        elif aux_visual_positions is not None:
            if aux_frame_count != len(aux_visual_positions):
                raise ValueError("Auxiliary frame support mismatch")
            episodic.update(aux_visual_positions=aux_visual_positions, aux_targets=aux_targets)
        elif self.training and self.g1_spec is not None:
            raise ValueError("G1 training requires explicit detached teacher targets")
        hidden = self.hidden(
            input_ids, mm_token_type_ids, pixel_values, image_grid_thw, step_plan,
            step_lane_roles=step_lane_roles, **episodic,
        )
        aux_sum = None
        if aux_visual_positions is not None or aux_anchors is not None:
            hidden, aux_sum = hidden
        if self.output_mode == "candidate_logits":
            logits = self.action_logits(hidden[0, read_positions])
            from qwen_vl.models.action_loss import action_losses

            training = self.navigation_config["training"]
            weights = training.get("action_class_weights")
            if weights is None:
                if training.get("class_weighting", "none") != "none":
                    raise ValueError(
                        "Class weights must be resolved from training data before use"
                    )
                weights = [1.0] * 4
            unweighted, losses = action_losses(logits, action_class_ids, weights)
            predictions = logits.argmax(-1)
        elif self.output_mode == "classification":
            logits = self.classifier(hidden[0, read_positions])
            losses = torch.nn.functional.cross_entropy(
                logits.float(), action_class_ids, reduction="none"
            )
            predictions = logits.argmax(-1)
        else:
            logits = self.backbone.lm_head(hidden[0, response_target_positions - 1])
            losses = per_action_token_loss(
                logits,
                input_ids[0, response_target_positions],
                response_action_index,
                num_actions,
            )
            predictions = None
        result = dict(
            loss_sum=losses.sum(),
            num_actions=num_actions,
            logits=logits,
            predictions=predictions,
            action_losses=losses.detach(),
        )
        if aux_sum is not None:
            result["aux_loss_sum"] = aux_sum
            result["aux_pair_count" if aux_anchors is not None else "aux_frame_count"] = aux_pair_count if aux_anchors is not None else aux_frame_count
        if self.output_mode == "candidate_logits":
            result["unweighted_loss_sum"] = unweighted.detach().sum()
        return result
