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


class MeasurementCallback(transformers.TrainerCallback):
    def __init__(self, out):
        self.out = Path(out)
        self.started = None

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
        )
        with (self.out / f"profile_rank{args.process_index}.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        if args.process_index == 0:
            print("UPDATE_PROFILE", json.dumps(record), flush=True)


class EpisodeTrainer(QwenSFTTrainer):
    def __init__(self, *args, serializer, data_contract, recovery_save_steps=0, **kwargs):
        self.serializer = serializer
        self.data_contract = data_contract
        self.recovery_save_steps = recovery_save_steps
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

    def _get_train_sampler(self, train_dataset=None):
        sampler = super()._get_train_sampler(train_dataset)
        self._episode_sampler = sampler
        return sampler

    def _run_epoch(self, model, epoch, *args, **kwargs):
        # Accelerate's resume SkipBatchSampler adds a third wrapper. Its
        # set_epoch traversal does not reach our underlying episode sampler.
        # Set the canonical Trainer epoch directly, including resumed epochs.
        sampler = getattr(self, '_episode_sampler', None)
        if hasattr(sampler, 'set_epoch'):
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
                path = Path(self.args.output_dir)/f'checkpoint-{self.state.global_step}'
                epochs = int(self.args.num_train_epochs)
                boundary = (epochs > 0 and self.state.max_steps % epochs == 0
                            and self.state.global_step % (self.state.max_steps // epochs) == 0)
                mark_complete(path, self.args.world_size, boundary)
                prune_recovery(self.args.output_dir, keep=2)
            except Exception as exc:
                error[0] = repr(exc)
        if dist.is_initialized():
            dist.broadcast_object_list(error, src=0)
        if error[0]:
            raise RuntimeError(f'Checkpoint completion failed: {error[0]}')

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
                backward_and_step_overhead_seconds=elapsed - self.last_forward_seconds,
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
            if prediction is not None:
                record["confusion"] = (
                    torch.bincount(truth * 4 + prediction, minlength=16)
                    .reshape(4, 4)
                    .cpu()
                    .tolist()
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
    p.add_argument("--manifest", required=True)
    p.add_argument("--model_name_or_path")
    p.add_argument("--episode-limit", type=int)
    p.add_argument("--selection", choices=["first", "shortest"], default="first")
    p.add_argument("--debug-repeat-episodes", action="store_true")
    p.add_argument("--max-optimizer-updates", type=int, default=-1)
    p.add_argument("--run-name", default="r2r_episode")
    p.add_argument("--output_dir", default="outputs/r2r_episode")
    p.add_argument("--resume_from_checkpoint")
    p.add_argument("--deepspeed", default="deepspeed.json")
    p.add_argument("--gradient_accumulation_steps", type=int)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--recovery-save-steps", type=int, default=0)
    p.add_argument(
        "--profile-only",
        action="store_true",
        help="Measure updates without publishing a checkpoint",
    )
    args = p.parse_args()
    cfg = resolve_config(args.vln_config, args.output_config, args.memory_config)
    if args.gradient_accumulation_steps:
        cfg["training"][
            "gradient_accumulation_steps"
        ] = args.gradient_accumulation_steps
    world = int(os.environ.get("WORLD_SIZE", 1))
    validate_config(cfg, world)
    if args.debug_repeat_episodes and args.max_optimizer_updates < 1:
        raise ValueError("Debug repetition requires explicit optimizer-update budget")
    t = cfg["training"]
    transformers.set_seed(t["seed"])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.recovery_save_steps < 0:
        raise ValueError('Recovery interval cannot be negative')
    if args.resume_from_checkpoint == 'auto':
        from qwen_vl.train.recovery import latest_checkpoint
        selected = latest_checkpoint(out)
        if selected is None and list(out.glob('checkpoint-*')):
            raise ValueError('No complete recovery checkpoint; refusing to restart silently')
        args.resume_from_checkpoint = str(selected) if selected else None
    model, serializer = load_model(cfg, args.model_name_or_path)
    dataset = EpisodeDataset(
        args.manifest, serializer, args.episode_limit, args.selection
    )
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
            from qwen_vl.train.recovery import validate_complete, quarantine_incomplete, rewind_reports
            recovery_record = validate_complete(args.resume_from_checkpoint)
            if int(os.environ.get('RANK', 0)) == 0:
                quarantine_incomplete(out)
                rewind_reports(out, recovery_record['step'])
    elif list(out.glob("checkpoint-*")) or (out / "final").exists():
        raise ValueError("Explicit resume required for existing training outputs")
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
    callbacks = [MeasurementCallback(out)]
    if args.recovery_save_steps and not args.profile_only:
        from qwen_vl.train.recovery import RecoverySaves
        if training_args.save_total_limit is not None:
            raise ValueError('Recovery retention requires save_total_limit=null to preserve epochs')
        callbacks.append(RecoverySaves(args.recovery_save_steps))
    if t.get("expected_total_steps") is not None and not args.profile_only:
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
        callbacks=callbacks,
    )
    if list(out.glob("checkpoint-*")) and not args.resume_from_checkpoint:
        raise ValueError("Explicit resume required for existing checkpoints")
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    if not args.profile_only:
        trainer.save_model(str(out / "final"))
    if training_args.process_index == 0 and not args.profile_only:
        model.eval()
        sample = {
            k: v.to(next(model.parameters()).device) if torch.is_tensor(v) else v
            for k, v in dataset[0].items()
        }
        with torch.inference_mode():
            check = model(**sample)
        (out / "final" / "reload_reference.json").write_text(
            json.dumps(
                dict(
                    episode_uid=sample["episode_uid"],
                    loss_sum=float(check["loss_sum"]),
                    logits=check["logits"].float().cpu().tolist()
                    if model.output_mode == "classification"
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
                    update_timer_note="Update callbacks exclude loader prefetch and model load/checkpoint save",
                ),
                indent=2,
            )
            + "\n"
        )
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
