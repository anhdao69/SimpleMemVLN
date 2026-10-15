"""Episode path through the existing HF Trainer entry point."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import torch
import transformers
from qwen_vl.contracts import validate_config, validate_resume_contract
from qwen_vl.data.episode_dataset import EpisodeDataset, episode_collator
from qwen_vl.train.argument import TrainingArguments
from qwen_vl.train.trainer import QwenSFTTrainer
from qwen_vl.train.vln_runtime import resolve_config, load_model
from qwen_vl.train.campaign import ScheduleGuard, CampaignReports, checkpoint_policy


def host_memory_record():
    """Process RSS plus job-cgroup usage (v1 or v2) for per-update resource logs."""
    from qwen_vl.train.host_memory import cgroup_sample, process_rss_gib
    from qwen_vl.models.activation_offload import activation_arena

    record = dict(process_rss_gib=process_rss_gib())
    try:
        record.update(cgroup_sample())
    except Exception as exc:  # resource logging must never stop training
        record["cgroup_error"] = repr(exc)[:200]
    arena = activation_arena()
    if arena is not None:
        record.update(arena.stats())
    return record


class InstallActivationArena(transformers.TrainerCallback):
    def __init__(self, capacity):
        self.capacity = capacity

    def on_train_begin(self, args, state, control, **kwargs):
        from qwen_vl.models.activation_offload import install_activation_arena

        arena = install_activation_arena(self.capacity)
        print("ACTIVATION_ARENA", json.dumps(dict(rank=args.process_index,
                                                  capacity_bytes=arena.capacity)), flush=True)


class StopAfterEpochs(transformers.TrainerCallback):
    """Stop at an epoch boundary without shortening the declared LR schedule."""

    def __init__(self, epochs):
        if epochs < 1:
            raise ValueError("stop-after-epochs must be positive")
        self.epochs = epochs

    def on_epoch_end(self, args, state, control, **kwargs):
        if round(state.epoch) >= self.epochs:
            control.should_save = True        # the epoch export/recovery state is written first
            control.should_training_stop = True
            print(f"STOP_AFTER_EPOCH {self.epochs} at update {state.global_step}", flush=True)
        return control


class CheckpointRequest(transformers.TrainerCallback):
    """Slurm pre-termination: save a complete recovery checkpoint at the next update boundary."""

    def __init__(self, out):
        self.flag = Path(out) / "REQUEST_CHECKPOINT_AND_STOP"

    def on_step_end(self, args, state, control, **kwargs):
        import torch.distributed as dist

        job = os.environ.get("SLURM_JOB_ID", "local")
        mine = self.flag.exists() and self.flag.read_text().strip() in (job, "")
        requested = torch.tensor([int(mine)], device=args.device)
        if dist.is_initialized():
            dist.all_reduce(requested)
        if int(requested):
            control.should_save = True
            control.should_training_stop = True
            print(f"CHECKPOINT_REQUEST honored at update {state.global_step}", flush=True)
        return control


class MeasurementCallback(transformers.TrainerCallback):
    def __init__(self, out, malloc_trim=False):
        self.out = Path(out)
        self.started = None
        self.malloc_trim = malloc_trim

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.started = time.perf_counter()
        self.unix_started = time.time()
        torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        record = dict(
            update=state.global_step,
            seconds=time.perf_counter() - self.started,
            unix_start=self.unix_started,
            unix_end=time.time(),
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            device_total_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
            cuda_ooms=torch.cuda.memory_stats().get("num_ooms", 0),
        )
        if self.malloc_trim:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)   # return freed staging heap to the OS
        record.update(host_memory_record())
        with (self.out / f"profile_rank{args.process_index}.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        if args.process_index == 0:
            print("UPDATE_PROFILE", json.dumps(record), flush=True)


class EpisodeTrainer(QwenSFTTrainer):
    def __init__(
        self, *args, serializer, data_contract, recovery_save_steps=0,
        episodic_diagnostics=False, **kwargs
    ):
        self.serializer = serializer
        self.data_contract = data_contract
        self.recovery_save_steps = recovery_save_steps
        self.episodic_diagnostics = episodic_diagnostics
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self):
        # Trainer fetches all GAS slots before forward. Keep those batches on
        # CPU; _prepare_inputs transfers/casts only the current episode.
        placement = self.accelerator.device_placement
        self.accelerator.device_placement = False
        try:
            return super().get_train_dataloader()
        finally:
            self.accelerator.device_placement = placement

    def _prepare_inputs(self, inputs):
        # Consume raw teacher pixels before Trainer/DeepSpeed can BF16-cast them.
        images=inputs.get('teacher_images')
        targets=None
        self.online_teacher_seconds=0.
        if images is not None:
            if not hasattr(self,'online_teacher'):
                from qwen_vl.research.vggt_omega import OnlineOmegaTeacher
                spec=self.model.navigation_config['research']
                if spec['run_id']=='R1':
                    from qwen_vl.research.vggt_r1_teacher import OnlineR1Teacher
                    self.online_teacher=OnlineR1Teacher(spec,self.args.device)
                elif spec['run_id']=='G3':
                    from qwen_vl.research.vggt_g3_teacher import OnlineG3Teacher
                    self.online_teacher=OnlineG3Teacher(spec,self.args.device)
                elif spec['run_id']=='A1':
                    from qwen_vl.research.geoaddress_teacher import OnlineAddressTeacher
                    self.online_teacher=OnlineAddressTeacher(spec,self.args.device)
                else:self.online_teacher=OnlineOmegaTeacher(spec,self.args.device)
            targets=self.online_teacher(images)
            self.online_teacher_seconds=self.online_teacher.last_seconds
        prepared=super()._prepare_inputs({k:v for k,v in inputs.items() if k not in ('teacher_images','aux_targets','aux_source_indices','aux_teacher_sources','aux_teacher_reads','aux_query_cells')})
        if targets is not None and 'aux_teacher_sources' in inputs:
            from qwen_vl.models.geoaddress_loss import teacher_relations
            source=targets[inputs['aux_teacher_sources'].to(targets.device)]
            current=targets[inputs['aux_teacher_reads'].to(targets.device)]
            cells=inputs['aux_query_cells'].to(targets.device)
            current=current.gather(1,cells[:,:,None].expand(-1,-1,targets.shape[-1]))
            prepared['aux_targets']=teacher_relations(current,source)
        elif targets is not None:
            prepared['aux_targets']=targets[inputs['aux_source_indices'].to(targets.device)] if 'aux_source_indices' in inputs else targets
        elif 'aux_targets' in inputs:
            prepared['aux_targets']=inputs['aux_targets'].to(self.args.device,dtype=torch.float32)
        return prepared

    def _get_train_sampler(self, train_dataset=None):
        sampler = super()._get_train_sampler(train_dataset)
        self._episode_sampler = sampler
        return sampler

    def _run_epoch(self, model, epoch, *args, **kwargs):
        # Accelerate's resume SkipBatchSampler adds a third wrapper. Its
        # set_epoch traversal does not reach our underlying episode sampler.
        # Set the canonical Trainer epoch directly, including resumed epochs.
        sampler = getattr(self, "_episode_sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        return super()._run_epoch(model, epoch, *args, **kwargs)

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        if not self.recovery_save_steps:
            return
        from qwen_vl.train.recovery import mark_complete, prune_recovery
        import torch.distributed as dist

        self.accelerator.wait_for_everyone()
        error = [None]
        if self.args.process_index == 0:
            try:
                path = (
                    Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
                )
                epochs = int(self.args.num_train_epochs)
                boundary = (
                    epochs > 0
                    and self.state.max_steps % epochs == 0
                    and self.state.global_step % (self.state.max_steps // epochs) == 0
                )
                mark_complete(path, self.args.world_size, boundary)
                prune_recovery(self.args.output_dir, keep=2)
            except Exception as exc:
                error[0] = repr(exc)
        if dist.is_initialized():
            dist.broadcast_object_list(error, src=0)
        if error[0]:
            raise RuntimeError(f"Checkpoint completion failed: {error[0]}")

    def training_step(self, model, inputs, num_items_in_batch=None):
        torch.cuda.synchronize()
        started = time.perf_counter()
        self.last_forward_seconds = None
        result = super().training_step(model, inputs, num_items_in_batch)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if self.last_forward_seconds is not None:
            record = dict(
                update=self.state.global_step + 1,
                episode_uid=inputs["episode_uid"],
                forward_seconds=self.last_forward_seconds,
                online_teacher_seconds=getattr(self,"online_teacher_seconds",0.),
                backward_and_step_overhead_seconds=(elapsed - self.last_forward_seconds
                    - getattr(self,"online_teacher_seconds",0.)),
            )
            with (
                Path(self.args.output_dir)
                / f"microtimes_rank{self.args.process_index}.jsonl"
            ).open("a") as f:
                f.write(json.dumps(record) + "\n")
        return result

    def _save(self, output_dir=None, state_dict=None):
        path = Path(output_dir or self.args.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        # torch.save preserves tied storage for this thin nn.Module wrapper.
        # HF5.11's generic safetensors.save_file path rejects tied parameters.
        torch.save(
            state_dict if state_dict is not None else self.model.state_dict(),
            path / "pytorch_model.bin",
        )
        torch.save(self.args, path / "training_args.bin")
        self.model.config.save_pretrained(path)
        (path / "navigation.json").write_text(
            json.dumps(self.serializer.metadata(), indent=2) + "\n"
        )
        (path / "resume_contract.json").write_text(
            json.dumps(
                {"navigation": self.serializer.metadata(), "data": self.data_contract},
                indent=2,
            )
            + "\n"
        )
        self.serializer.processor.save_pretrained(path)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        saved = json.loads(
            (Path(resume_from_checkpoint) / "navigation.json").read_text()
        )
        if saved != self.serializer.metadata():
            raise ValueError("Resume navigation contract differs from checkpoint")
        return super()._load_from_checkpoint(resume_from_checkpoint, model)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        torch.cuda.synchronize()
        started = time.perf_counter()
        diagnostics = None
        if model.training and "episodic_layout" in inputs and self.episodic_diagnostics:
            diagnostics = {}
            inputs = dict(inputs, episodic_diagnostics=diagnostics)
        from qwen_vl.models.activation_offload import activation_arena

        arena = activation_arena()
        if arena is not None and arena.live:
            raise RuntimeError("Activation arena still holds inputs from a previous episode")
        loss, output = super().compute_loss(model, inputs, True, num_items_in_batch)
        torch.cuda.synchronize()
        self.last_forward_seconds = time.perf_counter() - started
        if model.training:
            prediction = output["predictions"]
            truth = inputs["action_class_ids"]
            record = dict(
                update=self.state.global_step,
                episode_uid=inputs["episode_uid"],
                actions=inputs["num_actions"],
                tokens=inputs["input_ids"].numel(),
                loss_sum=float(output["loss_sum"].detach()),
                update_global_actions=int(num_items_in_batch),
            )
            if 'aux_loss_sum' in output:
                record.update(aux_loss_sum=float(output['aux_loss_sum'].detach()),
                              aux_support=output.get('aux_pair_count',output.get('aux_frame_count')),
                              **({'aux_pair_count':output['aux_pair_count']} if 'aux_pair_count' in output else {'aux_frame_count':output['aux_frame_count']}),
                              update_aux_count=output['update_aux_count'],aux_weight=output['aux_weight'])
            if prediction is not None:
                record["confusion"] = (
                    torch.bincount(truth * 4 + prediction, minlength=16)
                    .reshape(4, 4)
                    .cpu()
                    .tolist()
                )
            if "unweighted_loss_sum" in output:
                record["unweighted_loss_sum"] = float(output["unweighted_loss_sum"])
            if diagnostics is not None:
                from qwen_vl.stream.episodic import plan_summary

                record["episodic"] = {
                    str(layer): plan_summary(plan, inputs["episodic_layout"])
                    for layer, plan in diagnostics.get("plans", {}).items()}
            if hasattr(self, "action_metrics_callback") and prediction is not None:
                self.action_metrics_callback.add(
                    truth,
                    prediction,
                    output["loss_sum"],
                    output.get("unweighted_loss_sum", output["loss_sum"]),
                )
            with (
                Path(self.args.output_dir)
                / f"exposures_rank{self.args.process_index}.jsonl"
            ).open("a") as f:
                f.write(json.dumps(record) + "\n")
        return (loss, output) if return_outputs else loss


def train_episode():
    wall_start = time.perf_counter()
    p = argparse.ArgumentParser()
    p.add_argument("--vln_config", required=True)
    p.add_argument("--output_config", required=True)
    p.add_argument("--memory_config")
    p.add_argument(
        "--policy_config",
        help="Optional policy ablation overlay; preserves production schedule",
    )
    p.add_argument("--runtime_profile",
                   help="Validated H4/H2 runtime overlay (GAS, loader, offload); merged last")
    p.add_argument("--manifest", required=True)
    p.add_argument("--model_name_or_path")
    p.add_argument("--episode-limit", type=int)
    p.add_argument("--selection", choices=["first", "shortest"], default="first")
    p.add_argument("--debug-repeat-episodes", action="store_true")
    p.add_argument("--max-optimizer-updates", type=int, default=-1)
    p.add_argument("--run-name", default="r2r_episode")
    p.add_argument("--output_dir", default="outputs/r2r_episode")
    initialization = p.add_mutually_exclusive_group()
    initialization.add_argument("--resume_from_checkpoint")
    initialization.add_argument("--init-policy-checkpoint",
                                help="Strict trained-wrapper weights, fresh lanes and optimizer")
    p.add_argument("--deepspeed", default="deepspeed.json")
    p.add_argument("--gradient_accumulation_steps", type=int)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--recovery-save-steps", type=int, default=0)
    p.add_argument("--first-recovery-save-step", type=int, default=0,
                   help="Extra early recovery save (campaign winner verification)")
    p.add_argument("--stop-after-epochs", type=int,
                   help="Stop at this epoch boundary; the full declared schedule is kept")
    p.add_argument(
        "--profile-only",
        action="store_true",
        help="Measure updates without publishing a checkpoint",
    )
    args = p.parse_args()
    cfg = resolve_config(
        args.vln_config, args.output_config, args.memory_config, args.policy_config,
        args.runtime_profile,
    )
    if args.gradient_accumulation_steps:
        cfg["training"]["gradient_accumulation_steps"] = (
            args.gradient_accumulation_steps
        )
    world = int(os.environ.get("WORLD_SIZE", 1))
    validate_config(cfg, world)
    if args.debug_repeat_episodes and args.max_optimizer_updates < 1:
        raise ValueError("Debug repetition requires explicit optimizer-update budget")
    t = cfg["training"]
    transformers.set_seed(t["seed"])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.recovery_save_steps < 0:
        raise ValueError("Recovery interval cannot be negative")
    if args.resume_from_checkpoint == "auto":
        from qwen_vl.train.recovery import latest_checkpoint

        selected = latest_checkpoint(out)
        if selected is None and list(out.glob("checkpoint-*")):
            raise ValueError(
                "No complete recovery checkpoint; refusing to restart silently"
            )
        args.resume_from_checkpoint = str(selected) if selected else None
    if not args.resume_from_checkpoint and (
        list(out.glob("checkpoint-*")) or (out / "final").exists()
    ):
        raise ValueError("Explicit resume required for existing training outputs")
    from qwen_vl.models.action_loss import resolve_class_balance

    resolve_class_balance(cfg, args.manifest, args.episode_limit, args.selection)
    initialization_report = None
    if args.init_policy_checkpoint:
        from qwen_vl.train.lane_initialization import (
            initialize_step_lane_from_policy, checkpoint_identity,
        )
        model, serializer, initialization_report = initialize_step_lane_from_policy(
            args.init_policy_checkpoint, args.model_name_or_path, cfg,
            lane_seed=cfg["model"]["step_lane"].get("init_seed", 429),
        )
        if int(os.environ.get("RANK", 0)) == 0:
            initialization_report["parent_identity"] = checkpoint_identity(args.init_policy_checkpoint)
            (out / "lane_initialization.json").write_text(
                json.dumps(initialization_report, indent=2) + "\n"
            )
    else:
        model, serializer = load_model(cfg, args.model_name_or_path)
    dataset = EpisodeDataset(
        args.manifest, serializer, args.episode_limit, args.selection
    )
    offload_backend = t.get("activation_offload_backend", "save_on_cpu_pinned")
    threshold = t.get("activation_offload_min_tokens")
    arena_record = None
    if offload_backend == "pinned_arena_v1" and threshold is not None:
        longest = max(dataset.encoded_lengths)
        if longest >= threshold:
            from qwen_vl.models.activation_offload import arena_bytes_for

            text = model.config.text_config
            capacity = arena_bytes_for(longest, text.hidden_size, text.num_hidden_layers)
            # Installed at on_train_begin: after DeepSpeed init and any checkpoint load,
            # whose transient host copies would otherwise coincide with the arena.
            arena_record = dict(longest_tokens=longest, capacity_bytes=capacity)
    data_contract = dict(
        manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        episode_uids=[e["episode_uid"] for e in dataset.episodes],
        seed=t["seed"],
        tail_policy=t["tail_policy"],
        world_size=world,
        accumulation=t["gradient_accumulation_steps"],
    )
    # HF's DeepSpeed branch bypasses _load_from_checkpoint. Validate before
    # Trainer starts and before publishing any new provenance files.
    if args.resume_from_checkpoint:
        saved = json.loads(
            (Path(args.resume_from_checkpoint) / "resume_contract.json").read_text()
        )
        validate_resume_contract(saved, serializer.metadata(), data_contract)
        if args.recovery_save_steps:
            from qwen_vl.train.recovery import (
                validate_complete,
                quarantine_incomplete,
                rewind_reports,
            )

            recovery_record = validate_complete(args.resume_from_checkpoint)
            if int(os.environ.get("RANK", 0)) == 0:
                quarantine_incomplete(out)
                rewind_reports(out, recovery_record["step"])
    training_args = TrainingArguments(
        output_dir=str(out),
        run_name=args.run_name,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["backbone_lr"],
        mm_projector_lr=t["classifier_lr"],
        weight_decay=t["weight_decay"],
        adam_beta1=0.9,
        adam_beta2=0.95,
        bf16=True,
        gradient_checkpointing=t["checkpoint_decoder_layers"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        average_tokens_across_devices=True,
        dataloader_drop_last=False,
        accelerator_config={
            "even_batches": True,
            "split_batches": False,
            "dispatch_batches": False,
        },
        num_train_epochs=t["epochs"],
        max_steps=args.max_optimizer_updates,
        warmup_steps=t["warmup_steps"],
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
        max_grad_norm=1.0,
        logging_steps=1,
        **checkpoint_policy(t, args.profile_only, args.save_steps),
        dataloader_pin_memory=t.get("dataloader_pin_memory", True),
        dataloader_persistent_workers=t.get("dataloader_persistent_workers", False),
        dataloader_prefetch_factor=t.get("dataloader_prefetch_factor"),
        report_to="none",
        seed=t["seed"],
        deepspeed=args.deepspeed,
        model_max_length=t["model_max_length"],
        ddp_find_unused_parameters=False,
    )
    if training_args.process_index == 0:
        (out / "resolved_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        (out / "selected_episodes.json").write_text(
            json.dumps([e["episode_uid"] for e in dataset.episodes], indent=2) + "\n"
        )
        (out / "navigation.json").write_text(
            json.dumps(serializer.metadata(), indent=2) + "\n"
        )
        print(
            "TRAINING_CONTRACT",
            json.dumps(
                dict(
                    world_size=world,
                    microbatch=1,
                    accumulation=t["gradient_accumulation_steps"],
                    global_episode_batch=t["nominal_episodes_per_update"],
                    episode_records=len(dataset),
                    unique_episodes=len({e["episode_uid"] for e in dataset.episodes}),
                    action_exposures_per_dataset_pass=sum(
                        len(e["steps"]) for e in dataset.episodes
                    ),
                    min_tokens=min(dataset.encoded_lengths),
                    max_tokens=max(dataset.encoded_lengths),
                )
            ),
            flush=True,
        )
    callbacks = [MeasurementCallback(out, cfg["runtime"].get("malloc_trim_each_update", False))]
    if arena_record is not None:
        callbacks.insert(0, InstallActivationArena(arena_record["capacity_bytes"]))
    if not args.profile_only:
        callbacks.append(CheckpointRequest(out))
    if args.stop_after_epochs:
        if args.stop_after_epochs >= t["epochs"]:
            raise ValueError("stop-after-epochs must precede the declared final epoch")
        callbacks.append(StopAfterEpochs(args.stop_after_epochs))
    if getattr(model, "step_lane_spec", None) is not None:
        from qwen_vl.train.lane_reporting import LaneParameterCallback
        callbacks.append(LaneParameterCallback(out))
    if args.recovery_save_steps and not args.profile_only:
        from qwen_vl.train.recovery import RecoverySaves

        if training_args.save_total_limit is not None:
            raise ValueError(
                "Recovery retention requires save_total_limit=null to preserve epochs"
            )
        callbacks.append(RecoverySaves(args.recovery_save_steps, args.first_recovery_save_step))
    if (
        t.get("expected_total_steps") is not None
        and not args.profile_only
        and not args.debug_repeat_episodes
    ):
        callbacks.append(
            ScheduleGuard(t["expected_total_steps"], t["expected_warmup_steps"])
        )
        callbacks.append(CampaignReports())
    trainer = EpisodeTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=episode_collator,
        processing_class=serializer.tokenizer,
        serializer=serializer,
        data_contract=data_contract,
        recovery_save_steps=args.recovery_save_steps if not args.profile_only else 0,
        episodic_diagnostics=cfg["runtime"].get("episodic_diagnostics", False),
        callbacks=callbacks,
    )
    if cfg["model"]["output_mode"] == "candidate_logits":
        from qwen_vl.train.action_reporting import ActionMetricsCallback

        trainer.action_metrics_callback = ActionMetricsCallback(out)
        trainer.add_callback(trainer.action_metrics_callback)
    if list(out.glob("checkpoint-*")) and not args.resume_from_checkpoint:
        raise ValueError("Explicit resume required for existing checkpoints")
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    if not args.profile_only:
        trainer.save_model(str(out / "final"))
        if training_args.process_index == 0:
            (out / "final" / "export_info.json").write_text(json.dumps(dict(
                global_step=trainer.state.global_step, max_steps=trainer.state.max_steps,
                epoch=trainer.state.epoch)) + "\n")
    if training_args.process_index == 0 and not args.profile_only:
        model.eval()
        from qwen_vl.research.vggt_g1 import inference_sample
        sample = inference_sample(dataset[0],next(model.parameters()).device)
        with torch.inference_mode():
            check = model(**sample)
        (out / "final" / "reload_reference.json").write_text(
            json.dumps(
                dict(
                    episode_uid=sample["episode_uid"],
                    loss_sum=float(check["loss_sum"]),
                    logits=check["logits"].float().cpu().tolist()
                    if model.output_mode in ("classification", "candidate_logits")
                    else None,
                ),
                indent=2,
            )
            + "\n"
        )
    if training_args.process_index == 0:
        (out / "run_timing.json").write_text(
            json.dumps(
                dict(
                    wall_seconds_including_load_and_save=time.perf_counter()
                    - wall_start,
                    trainer_runtime_seconds=result.metrics.get("train_runtime"),
                    profile_only=args.profile_only,
                    stopped_after_epochs=args.stop_after_epochs,
                    activation_arena=arena_record,
                    update_timer_note="Update callbacks exclude loader prefetch and model load/checkpoint save",
                ),
                indent=2,
            )
            + "\n"
        )
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
