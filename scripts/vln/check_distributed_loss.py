"""Exercise actual Trainer/Accelerate/ZeRO-2 GAS and partial-tail updates.

Each update is compared with a separate single-process AdamW reference on the
exact exposed variable-length examples gathered from the two-rank loader.
"""
import argparse
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from transformers import TrainerCallback
from qwen_vl.train.argument import TrainingArguments
from qwen_vl.train.trainer import QwenSFTTrainer


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.3, -0.2]))

    def forward(self, x, y, num_actions):
        pred = x @ self.weight
        return {"loss_sum": ((pred - y) ** 2).sum()}


class Examples(Dataset):
    def __len__(self):
        return 9

    def __getitem__(self, i):
        n = i % 4 + 1
        x = torch.arange(n * 2, dtype=torch.float32).reshape(n, 2) / 10 + i / 20
        return dict(x=x, y=torch.linspace(0.1, 0.7, n), num_actions=n)


class TextToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor([[0.3, -0.2, 0.1], [-0.1, 0.2, 0.4]])
        )

    def forward(self, x, y, action_index, num_actions):
        from qwen_vl.models.nav_model import per_action_token_loss

        return {
            "loss_sum": per_action_token_loss(
                x @ self.weight, y, action_index, num_actions
            ).sum()
        }


class TextExamples(Examples):
    def __getitem__(self, i):
        n = i % 4 + 1
        counts = [(i + j) % 3 + 1 for j in range(n)]
        nt = sum(counts)
        return dict(
            x=torch.arange(nt * 2, dtype=torch.float32).reshape(nt, 2) / 10 + i / 20,
            y=torch.arange(nt) % 3,
            action_index=torch.repeat_interleave(torch.arange(n), torch.tensor(counts)),
            num_actions=n,
        )


class CheckedTrainer(QwenSFTTrainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        self.pending.append(
            {
                k: v.detach().cpu() if torch.is_tensor(v) else v
                for k, v in inputs.items()
            }
        )
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)


class Reference(TrainerCallback):
    def __init__(self, trainer, text_output=False):
        self.trainer = trainer
        self.text_output = text_output
        self.reference = TextToy() if text_output else Toy()
        self.optimizer = torch.optim.AdamW(
            self.reference.parameters(),
            lr=0.001,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
        )
        self.records = []

    def on_step_end(self, args, state, control, **kwargs):
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, self.trainer.pending)
        examples = [e for rank in gathered for e in rank]
        self.optimizer.zero_grad()
        if self.text_output:
            # Independent explicit per-action loop, not the production reducer.
            sums = []
            for e in examples:
                token_losses = torch.nn.functional.cross_entropy(
                    e["x"] @ self.reference.weight, e["y"], reduction="none"
                )
                sums.extend(
                    token_losses[e["action_index"] == j].mean()
                    for j in range(e["num_actions"])
                )
            loss = torch.stack(sums).mean()
        else:
            loss = sum(self.reference(**e)["loss_sum"] for e in examples) / sum(
                e["num_actions"] for e in examples
            )
        loss.backward()
        self.optimizer.step()
        actual = self.trainer.model.weight.detach().cpu()
        torch.testing.assert_close(actual, self.reference.weight, atol=2e-6, rtol=2e-6)
        self.records.append(
            dict(
                update=state.global_step,
                exposures=len(examples),
                actions=sum(e["num_actions"] for e in examples),
                reference_loss=float(loss.detach()),
                max_weight_error=float(
                    (actual - self.reference.weight).abs().max().detach()
                ),
            )
        )
        self.trainer.pending = []

    def on_train_end(self, args, state, control, **kwargs):
        assert [r["exposures"] for r in self.records] == [4, 4, 2]
        if args.process_index == 0:
            name = "distributed_text_loss" if self.text_output else "distributed_loss"
            Path(f"artifacts/{name}.json").write_text(
                json.dumps(self.records, indent=2) + "\n"
            )
            print("DISTRIBUTED_REFERENCE_PASS", self.records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-output", action="store_true")
    text_output = parser.parse_args().text_output
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    args = TrainingArguments(
        output_dir="artifacts/toy_loss",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        learning_rate=0.001,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        lr_scheduler_type="constant",
        max_grad_norm=0.0,
        bf16=False,
        remove_unused_columns=False,
        average_tokens_across_devices=True,
        group_by_modality_length=False,
        save_strategy="no",
        report_to="none",
        logging_steps=1,
        accelerator_config={
            "even_batches": True,
            "split_batches": False,
            "dispatch_batches": False,
        },
        deepspeed="deepspeed.json",
    )
    trainer = CheckedTrainer(
        model=TextToy() if text_output else Toy(),
        args=args,
        train_dataset=TextExamples() if text_output else Examples(),
        data_collator=lambda batch: batch[0],
    )
    trainer.pending = []
    trainer.add_callback(Reference(trainer, text_output))
    trainer.train()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
