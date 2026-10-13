"""Trainer support needed by plain Qwen3.5 supervised fine-tuning."""

import torch
from transformers import Trainer
from qwen_vl.train.sampler import _get_train_sampler


class QwenSFTTrainer(Trainer):
    """Give the trainable vision merger its configured learning rate."""

    _get_train_sampler = _get_train_sampler

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We normalize over action states in the entire accumulation window.
        # Trainer must not divide again; DeepSpeed scale_wrt_gas is also False.
        self.model_accepts_loss_kwargs = True

    def _get_num_items_in_batch(self, batch_samples, device):
        count = torch.tensor(sum(batch.get('num_actions', batch.get('labels',torch.empty(0)).shape[0]) for batch in batch_samples), device=device)
        if any('aux_frame_count' in batch or 'aux_pair_count' in batch for batch in batch_samples):
            frames = torch.tensor(sum(batch.get('aux_pair_count',batch.get('aux_frame_count',0)) for batch in batch_samples),device=device)
            global_frames = int(self.accelerator.gather(frames).sum())
            for batch in batch_samples:
                batch['update_aux_count'] = global_frames
        return self.accelerator.gather(count).sum()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        if 'num_actions' in inputs:
            if model.training and (num_items_in_batch is None or num_items_in_batch <= 0):
                raise ValueError('Training requires a positive global update action count')
            aux_count = inputs.pop('update_aux_count', None)
            outputs = model(**inputs)
            loss = outputs['loss_sum'] / inputs['num_actions'] if num_items_in_batch is None else (
                outputs['loss_sum'] * self.accelerator.num_processes / num_items_in_batch)
            if 'aux_loss_sum' in outputs:
                from qwen_vl.research.vggt_g1 import auxiliary_weight, normalized_aux_loss
                spec = self.model.navigation_config['research']
                weight = auxiliary_weight(self.state.global_step,spec['ramp_updates'],spec['lambda_max'])
                if aux_count is None:
                    if model.training: raise ValueError('G1 requires global update frame count')
                    aux_count = inputs.get('aux_pair_count',inputs.get('aux_frame_count'))
                world = self.accelerator.num_processes if num_items_in_batch is not None else 1
                loss = loss + normalized_aux_loss(outputs['aux_loss_sum'],aux_count,world,weight)
                outputs['aux_weight'] = weight
                outputs['update_aux_count'] = aux_count
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite episode loss; abort distributed update')
            return (loss, outputs) if return_outputs else loss
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"]
        # The collator aligns selected prediction logits with labels shifted
        # once (dummy leading label/trailing logit). Dense labels use the same
        # one-shift rule. Average tokens within each action state.
        targets = labels[:, 1:]
        valid = targets.ne(-100)
        counts = valid.sum(-1)
        if (counts == 0).any():
            raise ValueError("Every action state must have supervised targets")
        token_losses = torch.nn.functional.cross_entropy(
            logits[:, :-1][valid].float(), targets[valid], reduction="none"
        )
        rows = valid.nonzero()[:, 0]
        sums = token_losses.new_zeros(labels.shape[0]).scatter_add(0, rows, token_losses)
        per_state = sums / counts
        loss = per_state.mean() if num_items_in_batch is None else (
            per_state.sum() * self.accelerator.num_processes / num_items_in_batch
        )
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        # Exclude custom RMSNorm weights as well as standard LayerNorm/bias.
        decay_names = {
            name for name, parameter in self.model.named_parameters()
            if parameter.ndim > 1 and not name.endswith("bias")
            and "norm" not in name.lower()
        }
        merger_names = {
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and (".visual.merger." in name or name.startswith('classifier.'))
        }

        lane_names = {
            name for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and ".step_lane." in name
        }
        lane_lr = getattr(self.model, "navigation_config", {}).get("training", {}).get(
            "step_lane_lr", 1e-4
        )
        if lane_names and (not __import__("math").isfinite(lane_lr) or lane_lr <= 0):
            raise ValueError("step_lane_lr must be finite and positive")
        aux_names = {name for name,p in self.model.named_parameters()
                     if p.requires_grad and name.startswith('research_aux.')}
        aux_lr = getattr(self.model,'navigation_config',{}).get('research',{}).get('head_lr',1e-4)
        grouped_parameters = []
        for kind, learning_rate in (
            ("backbone", self.args.learning_rate),
            ("merger", self.args.mm_projector_lr),
            ("lane", lane_lr),
            ("aux", aux_lr),
        ):
            for use_decay in (True, False):
                parameters = [
                    parameter
                    for name, parameter in self.model.named_parameters()
                    if parameter.requires_grad
                    and (
                        "aux" if name in aux_names else
                        "lane" if name in lane_names else
                        "merger" if name in merger_names else "backbone"
                    ) == kind
                    and (name in decay_names) is use_decay
                ]
                if parameters:
                    grouped_parameters.append(
                        {
                            "params": parameters,
                            "weight_decay": (
                                self.args.weight_decay if use_decay else 0.0
                            ),
                            "lr": learning_rate,
                        }
                    )

        chunks = getattr(self.model, "navigation_config", {}).get("training", {}).get(
            "optimizer_group_chunks", 1)
        if chunks > 1:
            # ZeRO-1/2 stages every param group through host RAM (a CPU copy plus a
            # flat CPU buffer) at init. Identical-hyperparameter chunks bound that
            # transient; AdamW is elementwise and clipping uses the global norm.
            total = sum(p.numel() for group in grouped_parameters for p in group["params"])
            target, split = total / chunks, []
            for group in grouped_parameters:
                current = []
                for parameter in group["params"]:
                    current.append(parameter)
                    if sum(p.numel() for p in current) >= target:
                        split.append(dict(group, params=current))
                        current = []
                if current:
                    split.append(dict(group, params=current))
            grouped_parameters = split
        assigned = [
            id(parameter)
            for group in grouped_parameters
            for parameter in group["params"]
        ]
        expected = {
            id(parameter)
            for parameter in self.model.parameters()
            if parameter.requires_grad
        }
        if len(assigned) != len(set(assigned)) or set(assigned) != expected:
            raise RuntimeError("Optimizer groups do not exactly cover trainable parameters")

        implementation = getattr(self.model, "navigation_config", {}).get(
            "training", {}).get("cpu_optimizer_impl", "torch")
        if implementation not in ("torch", "deepspeed_cpu_adam"):
            raise ValueError("Unsupported CPU optimizer implementation")
        if implementation == "deepspeed_cpu_adam":
            import json
            from pathlib import Path
            ds = self.args.deepspeed
            if isinstance(ds, str):
                ds = json.loads(Path(ds).read_text())
            if not isinstance(ds, dict) or ds.get("zero_optimization", {}).get(
                    "offload_optimizer", {}).get("device") != "cpu":
                raise ValueError("DeepSpeedCPUAdam requires explicit optimizer CPU offload")
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            self.optimizer = DeepSpeedCPUAdam(grouped_parameters,
                lr=self.args.learning_rate, betas=(self.args.adam_beta1,self.args.adam_beta2),
                eps=self.args.adam_epsilon, adamw_mode=True, fp32_optimizer_states=True)
            return self.optimizer

        optimizer_class, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
            self.args
        )
        self.optimizer = optimizer_class(grouped_parameters, **optimizer_kwargs)
        return self.optimizer
