# ActiveVLN GRPO Baseline on SimpleMem — Goal Specification

> Saved verbatim from the user's goal statement on 2026-10-10. This is the governing
> specification for the RL baseline work; no implementation had started when it was saved.
> (The only edit: the branch name, written as `` `RL`\_baseline `` in the original, is spelled `RL_baseline`.)

## Goal

Reproduce the official ActiveVLN GRPO RL baseline on our SimpleMem model, complete RL training on the 4K R2R dataset, and evaluate the trained checkpoint against the original SimpleMem checkpoint using an identical R2R evaluation protocol.

## Research context

Our long-term objective is to develop a novel, CVPR/ICML-level RL method that significantly improves SimpleMem's R2R Success Rate (SR) over the standard ActiveVLN GRPO baseline. However, the immediate objective is to establish a correct, reliable, reproducible GRPO baseline before proposing new algorithms.

Project: `/data/anh/code/SimpleMem`

Starting checkpoint: `/data/anh/checkpoints/simplemem/joint-window8-text-b`

Reference implementation: [https://github.com/arvillion/ActiveVLN](https://github.com/arvillion/ActiveVLN)

Research plans: `/data/anh/code/SimpleMem/plans`

All implementation changes must be developed on the `RL_baseline` Git branch. Create it if it does not exist, without overwriting existing work.

## Phase 1 — Understand the project

Thoroughly inspect the current codebase, relevant local and remote branches, architecture, model checkpoint, inference pipeline, training implementation, dependencies, and R2R evaluation.

Read the existing research reports and RL plans. Understand previous decisions, experiments, and limitations. Inspect the differences between relevant branches rather than assuming the current branch contains every important implementation.

Identify the precise integration points for RL. Check whether the model's action representation, observation format, tokenizer, policy outputs, and training architecture are compatible with ActiveVLN.

Inspect the existing R2R baseline inference job. Preserve its processes, GPUs, checkpoints, logs, and outputs.

## Phase 2 — Study and integrate ActiveVLN

Clone and inspect the official ActiveVLN implementation at a recorded commit. Understand its complete RL pipeline: rollout generation, environment interaction, group sampling, reward calculation, GRPO loss, policy updates, dynamic early stopping, and evaluation.

Use the official 4K R2R training dataset. Verify dataset integrity and train/validation separation.

Adapt the official RL approach to SimpleMem while preserving the original architecture and existing SFT checkpoint. Do not replace SimpleMem with ActiveVLN's pretrained model.

Implement the necessary model adapters, environment interfaces, loss functions, training scripts, configurations, and logging.

Maintain the original ActiveVLN algorithm as faithfully as possible. Document unavoidable differences from its reference implementation.

Verify whether the existing SimpleMem R2R environment and ActiveVLN's R2R-VLNCE environment are equivalent. Do not compare success rates from incompatible evaluation protocols.

## Phase 3 — Validate and train

Before full training:

1. Verify the RL environment and simulator.
2. Test model inference and action execution.
3. Validate rewards, grouped rollouts, log probabilities, and GRPO optimization.
4. Run a small end-to-end RL training smoke test.
5. Check finite losses, gradients, parameter updates, and checkpoint saving/loading.
6. Verify sufficient GPU, CPU RAM, disk space, and simulator resources.

Then train the SimpleMem checkpoint using ActiveVLN's GRPO method on the full 4K R2R dataset, following the official hyperparameters wherever compatible.

Do not overwrite the original checkpoint or disrupt existing jobs. Use isolated environments, outputs, and ports. Save configs, logs, checkpoints, and reproducibility metadata.

Diagnose and fix implementation errors autonomously. If insufficient resources or an incompatible environment prevents training, document the precise blocker instead of reporting success.

## Phase 4 — Evaluation and comparison

Evaluate the resulting GRPO checkpoint using exactly the same evaluation protocol as the original SimpleMem checkpoint, prioritizing R2R val-unseen.

Report SR, SPL, NE, OSR, and other available metrics, along with absolute and relative improvements.

Compare:

- Original SimpleMem (without RL)
- SimpleMem + ActiveVLN GRPO
- Published ActiveVLN results as contextual reference, clearly distinguishing different backbones or protocols

Use the existing baseline inference results when complete. Never invent missing metrics or compare incomplete runs as if they were final.

## Phase 5 — Research handoff

After the baseline experiment, analyze strengths, weaknesses, failure cases, and limitations of GRPO for SimpleMem.

Review the existing RL plans and identify promising novel research directions. Rank ideas by technical novelty, expected performance benefit, implementation feasibility, and experimental cost.

Do not implement speculative novel algorithms in this first goal. Establish the baseline before beginning that research stage.

## Required deliverables

Write a reproducible technical report under `plans/RL/` containing the codebase audit, integration decisions, training configuration, experiment commands, completed runs, checkpoint paths, evaluation results, failure analysis, and proposed next research steps.

## Completion criteria

Do not declare success until the GRPO integration passes its tests, full 4K-data RL training has completed successfully, the trained model has been evaluated, and the baseline comparison has been documented with evidence.

Performance improvement is an experiment outcome, not a prerequisite for declaring the baseline reproduction successful.

Work autonomously through implementation and debugging. If blocked by missing resources, permissions, or irrecoverable incompatibilities, report the specific blocker and all completed work. Do not fabricate results or silently substitute another experiment.
