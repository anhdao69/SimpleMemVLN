"""Thin native Qwen bridge: frozen vision, full text graph and sparse readouts."""
import torch
from torch import nn
from qwen_vl.stream.positions import PositionLedger


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

    def __init__(self, backbone, navigation_config):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        self.navigation_config = navigation_config
        self.output_mode = navigation_config["model"]["output_mode"]
        self.backbone.model.visual.requires_grad_(False)
        self.backbone.model.visual.eval()
        self.classifier = None
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

    def train(self, mode=True):
        super().train(mode)
        self.backbone.model.visual.eval()
        return self

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.backbone.model.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            or {"use_reentrant": False}
        )
        threshold = self.navigation_config['training'].get('activation_offload_min_tokens')
        if threshold is not None:
            from functools import partial
            from qwen_vl.models.activation_offload import checkpoint_with_cpu_offload
            self.backbone.model.language_model._set_gradient_checkpointing(
                enable=True,
                gradient_checkpointing_func=partial(
                    checkpoint_with_cpu_offload, min_tokens=threshold,
                    **(gradient_checkpointing_kwargs or {'use_reentrant': False}),
                ),
            )

    def embed(self, input_ids, pixel_values=None, image_grid_thw=None):
        embeddings = self.backbone.model.language_model.embed_tokens(input_ids)
        if pixel_values is not None:
            features = []
            offset = 0
            count = self.navigation_config["runtime"]["vision_microbatch_images"]
            with torch.no_grad():
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
    ):
        if position_ids is None:
            position_ids = PositionLedger().append(
                self.backbone.model, input_ids, mm_token_type_ids, image_grid_thw
            )
        return self.backbone.model.language_model(
            inputs_embeds=self.embed(input_ids, pixel_values, image_grid_thw),
            position_ids=position_ids,
            attention_mask=None,
            past_key_values=cache,
            use_cache=cache is not None,
            step_plan=step_plan,
            stream_append=stream_append,
            output_hidden_states=False,
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
    ):
        hidden = self.hidden(
            input_ids, mm_token_type_ids, pixel_values, image_grid_thw, step_plan
        )
        if self.output_mode == "classification":
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
        return dict(
            loss_sum=losses.sum(),
            num_actions=num_actions,
            logits=logits,
            predictions=predictions,
            action_losses=losses.detach(),
        )
