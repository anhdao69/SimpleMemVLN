"""Optimizer-boundary resume equivalence through the production save hooks.

Run two torchrun processes separately: --phase baseline then --phase resume.
The independent CPU AdamW state at checkpoint 1 is the reference continuation.
"""
import argparse, json
from pathlib import Path
import os
import torch
import torch.distributed as dist
from transformers import PretrainedConfig, TrainerCallback
from qwen_vl.train.argument import TrainingArguments
from qwen_vl.train.train_episode import EpisodeTrainer
from qwen_vl.train.trainer import QwenSFTTrainer
from qwen_vl.contracts import validate_resume_contract
from scripts.vln.check_distributed_loss import Toy, Examples


class Serializer:
    def __init__(self):
        self.processor = self

    def metadata(self):
        return {"kind": "resume_test_v1"}

    def save_pretrained(self, path):
        pass


class CheckedTrainer(EpisodeTrainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        self.pending.append(
            {
                k: v.detach().cpu() if torch.is_tensor(v) else v
                for k, v in inputs.items()
            }
        )
        return QwenSFTTrainer.compute_loss(
            self, model, inputs, return_outputs, num_items_in_batch
        )


class Reference(TrainerCallback):
    def __init__(self, trainer, phase, prefix):
        self.trainer, self.phase = trainer, phase
        self.prefix = prefix
        self.reference = Toy()
        self.optimizer = torch.optim.AdamW(
            self.reference.parameters(),
            lr=0.001,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
        )
        self.records = []
        if phase == "resume":
            saved = torch.load(
                f"artifacts/{prefix}_baseline/reference_1.pt", weights_only=True
            )
            self.reference.load_state_dict(saved["model"])
            self.optimizer.load_state_dict(saved["optimizer"])

    def on_step_end(self, args, state, control, **kwargs):
        all_examples = [None] * dist.get_world_size()
        dist.all_gather_object(all_examples, self.trainer.pending)
        examples = [e for rank in all_examples for e in rank]
        self.optimizer.zero_grad()
        loss = sum(self.reference(**e)["loss_sum"] for e in examples) / sum(
            e["num_actions"] for e in examples
        )
        loss.backward()
        self.optimizer.step()
        actual = self.trainer.model.weight.detach().cpu()
        torch.testing.assert_close(actual, self.reference.weight, atol=2e-6, rtol=2e-6)
        signatures = [float(e["x"][0, 0]) for e in examples]
        if self.phase == "resume":
            uninterrupted = torch.load(
                f"artifacts/{self.prefix}_baseline/reference_{state.global_step}.pt",
                weights_only=True,
            )
            assert signatures == uninterrupted["exposure_signatures"]
            torch.testing.assert_close(
                actual, uninterrupted["model"]["weight"], atol=2e-6, rtol=2e-6
            )
        if self.phase == "baseline" and args.process_index == 0:
            torch.save(
                {
                    "model": self.reference.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "exposure_signatures": signatures,
                },
                f"artifacts/{self.prefix}_baseline/reference_{state.global_step}.pt",
            )
        self.records.append(
            {
                "step": state.global_step,
                "exposures": len(examples),
                "exposure_signatures": signatures,
                "max_error": float(
                    (actual - self.reference.weight).abs().max().detach()
                ),
            }
        )
        self.trainer.pending = []

    def on_train_end(self, args, state, control, **kwargs):
        assert [r["exposures"] for r in self.records] == (
            [4, 4, 2] if self.phase == "baseline" else [4, 2]
        )
        if args.process_index == 0:
            Path(f"artifacts/distributed_{self.prefix}_{self.phase}.json").write_text(
                json.dumps(self.records, indent=2) + "\n"
            )
            print("RESUME_EQUIVALENCE_PASS", self.records)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["baseline", "resume"], required=True)
    p.add_argument("--episode-sampler", action="store_true")
    cli = p.parse_args()
    phase = cli.phase
    prefix = "resume_bucket" if cli.episode_sampler else "resume"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    output = f"artifacts/{prefix}_{phase}"
    args = TrainingArguments(
        output_dir=output,
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
        save_steps=1,
        report_to="none",
        logging_steps=1,
        deepspeed="deepspeed.json",
        accelerator_config={
            "even_batches": True,
            "split_batches": False,
            "dispatch_batches": False,
        },
    )
    model = Toy()
    model.config = PretrainedConfig()
    serializer = Serializer()
    data_contract = {"example_ids": list(range(9))}
    checkpoint = (
        f"artifacts/{prefix}_baseline/checkpoint-1" if phase == "resume" else None
    )
    if checkpoint:
        validate_resume_contract(
            json.loads((Path(checkpoint) / "resume_contract.json").read_text()),
            serializer.metadata(),
            data_contract,
        )
    dataset = Examples()
    if cli.episode_sampler:
        dataset.encoded_lengths = [i % 4 + 1 for i in range(len(dataset))]
    trainer = CheckedTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=lambda rows: rows[0],
        serializer=serializer,
        data_contract=data_contract,
    )
    trainer.pending = []
    trainer.add_callback(Reference(trainer, phase, prefix))
    trainer.train(resume_from_checkpoint=checkpoint)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
