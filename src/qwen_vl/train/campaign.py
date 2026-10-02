"""Fail-closed gates shared by scheduled training and publication."""
import hashlib
import json
import math
from pathlib import Path

from transformers import TrainerCallback


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path, expected):
    if sha256_file(path) != expected:
        raise ValueError(f"SHA256 mismatch: {path}")


def checkpoint_policy(training, profile_only, save_steps):
    return dict(
        save_strategy="no" if profile_only else training.get("save_strategy", "steps"),
        save_total_limit=training.get("save_total_limit", 2),
        save_steps=save_steps,
        dataloader_num_workers=training.get("dataloader_num_workers", 2),
    )


class ScheduleGuard(TrainerCallback):
    def __init__(self, steps, warmup):
        self.steps, self.warmup = steps, warmup

    def on_train_begin(self, args, state, control, **kwargs):
        actual = (state.max_steps, args.get_warmup_steps(state.max_steps))
        if actual != (self.steps, self.warmup):
            raise ValueError(
                f"Schedule gate failed: actual={actual}, expected={(self.steps, self.warmup)}"
            )
        print("SCHEDULE_GATE_PASS", actual, flush=True)


class CampaignReports(TrainerCallback):
    def _report(self, args, start, end, name):
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()
        if args.process_index == 0:
            from scripts.vln.report_campaign import write_report

            write_report(args.output_dir, start, end, name)
        if dist.is_initialized():
            dist.barrier()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 50:
            self._report(args, 1, 50, "first-50-updates")

    def on_save(self, args, state, control, **kwargs):
        epoch = int(round(state.epoch))
        epochs = int(args.num_train_epochs)
        if epochs != args.num_train_epochs or state.max_steps % epochs:
            raise ValueError('Epoch reports require a whole-epoch schedule')
        steps_per_epoch = state.max_steps // epochs
        if state.global_step != epoch * steps_per_epoch:
            return  # Periodic recovery saves are not epoch reports.
        self._report(args, (epoch - 1) * steps_per_epoch + 1,
                     state.global_step, f"epoch-{epoch}")


def check_memory_profile(root, max_reserved=78.0):
    root = Path(root)
    representatives = json.loads((root / "representatives.json").read_text())
    if {r["bucket"] for r in representatives} != {"short", "median", "p95", "longest"}:
        raise ValueError("Missing length representative")
    seen, peaks = set(), []
    for rank in range(4):
        rows = [
            json.loads(s)
            for s in (root / f"profile_rank{rank}.jsonl").read_text().splitlines()
        ]
        if not {1, 2}.issubset({r["update"] for r in rows}):
            raise ValueError(
                "Must measure at least two optimizer updates on every rank"
            )
        peaks.extend(r["peak_reserved_gib"] for r in rows)
        seen.update(
            json.loads(s)["episode_uid"]
            for s in (root / f"exposures_rank{rank}.jsonl").read_text().splitlines()
        )
    if not {r["episode_uid"] for r in representatives}.issubset(seen):
        raise ValueError("Not all representatives were trained")
    if not all(math.isfinite(p) and 0 < p <= max_reserved for p in peaks):
        raise ValueError(
            f"Memory gate failed: peak={max(peaks)} GiB; limit={max_reserved}"
        )
    return {"peak_reserved_gib": max(peaks), "limit_gib": max_reserved, "passed": True}
