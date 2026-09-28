"""Four-rank integration test: three epoch saves, tail updates, strict reload."""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import PretrainedConfig, TrainerCallback
from qwen_vl.train.argument import TrainingArguments
from qwen_vl.train.campaign import ScheduleGuard, checkpoint_policy
from scripts.vln.check_distributed_loss import Toy, Examples
from scripts.vln.check_distributed_resume import CheckedTrainer, Serializer


def build_dataset():
    dataset = Examples()
    dataset.encoded_lengths = [i % 4 + 1 for i in range(len(dataset))]
    return dataset


class CheckSaved(TrainerCallback):
    def on_save(self, args, state, control, model=None, **kwargs):
        if args.process_index == 0:
            path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            weights = torch.load(
                path / "pytorch_model.bin", weights_only=True, map_location="cpu"
            )
            actual = Toy()
            actual.load_state_dict(weights, strict=True)
            torch.testing.assert_close(
                actual.weight, model.weight.detach().cpu(), rtol=0, atol=0
            )
            assert (
                json.loads((path / "trainer_state.json").read_text())["epoch"]
                == state.epoch
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    out = Path(p.parse_args().out)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    args = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=5e-6,
        warmup_steps=1,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
        remove_unused_columns=False,
        average_tokens_across_devices=True,
        report_to="none",
        logging_steps=1,
        bf16=False,
        deepspeed="deepspeed.json",
        accelerator_config={
            "even_batches": True,
            "split_batches": False,
            "dispatch_batches": False,
        },
        **checkpoint_policy(
            {
                "save_strategy": "epoch",
                "save_total_limit": None,
                "dataloader_num_workers": 4,
            },
            False,
            500,
        ),
    )
    model = Toy()
    model.config = PretrainedConfig()
    trainer = CheckedTrainer(
        model=model,
        args=args,
        train_dataset=build_dataset(),
        data_collator=lambda rows: rows[0],
        serializer=Serializer(),
        data_contract={"test": "epoch-save"},
        callbacks=[ScheduleGuard(6, 1), CheckSaved()],
    )
    trainer.pending = []
    trainer.train()
    dist.barrier()
    if args.process_index == 0:
        assert {p.name for p in out.glob("checkpoint-*")} == {
            "checkpoint-2",
            "checkpoint-4",
            "checkpoint-6",
        }
        for step in (2, 4, 6):
            assert (
                len(list((out / f"checkpoint-{step}").rglob("*optim_states.pt"))) == 4
            )
        (out / "PASS.json").write_text(
            json.dumps({"passed": True, "epochs": 3, "steps": 6}) + "\n"
        )
        print("EPOCH_SAVE_RELOAD_PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
