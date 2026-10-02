# Joint FullContext candidate-logit submission

Slurm job **4623**, submitted to `main`, no node pinning. Launcher:
`train/slurm/candidate_joint_full.slurm`. Source snapshot is immutable
`streaming_logits@8a6a28acabd43d378e610345b2ebbe24547c37f8`.

Campaign directory:
`/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/outputs/candidate_joint_full_20261002-W6vveI`.
Logs: `slurm-4623.log`, `train/train.log`, `train/action_metrics.jsonl`.

## Recipe

- One joint model, FullContext (no Window8 overlay), candidate logits.
- Pretrained Qwen3.5-4B @851bf6e; trainable tied LM rows, candidate-token history.
- Clean unweighted CE baseline (`class_weighting: none`).
- R2R + English RxR_15deg: 30,815 episodes, 3,128,624 actions.
- Canonical action counts: 1,659,308 forward; 740,699 left; 697,802 right; 30,815 STOP.
- 2 epochs, global batch 8 = 4 GPUs × 1 episode × GAS2; backbone LR 5e-6.
- 7,704 optimizer steps, 232 warmup steps, existing cosine/min-LR recipe.
- 4 H100 GPUs, 24 CPUs, 768 GiB host RAM, 72-hour time limit.
- Four loader workers/rank, OMP2, expandable CUDA allocator, activation offload.
- Save both epoch checkpoints; recovery every 100 updates with two rolling
  non-epoch recovery checkpoints retained, plus final serving export. A save
  can temporarily add a checkpoint before pruning completes.

## Checks completed before submission

The previous implementation smoke was R2R-only Window8; it was not a joint
FullContext training smoke. A new smoke was therefore run on interactive job
4620 with two GPUs, GAS4, global batch 8. It used four R2R and four RxR_15deg
episodes (short and median cases), 384 actions/pass, longest 113 observations,
and three optimizer updates from pretrained weights. Zero warmup was used only
for this debug run. The production run starts independently from pretrained.

| Update | Global CE | Update seconds (max rank) | Peak reserved GiB |
|---|---:|---:|---:|
| 1 | 0.991674 | 12.157 | 64.00 |
| 2 | 0.755047 | 9.588 | 64.00 |
| 3 | 0.562692 | 9.580 | 64.00 |

No OOM/nonfinite loss. This is not a convergence or Habitat-quality result.
STOP recall remains zero in this tiny smoke. No full-corpus duration estimate
is inferred from short/median trajectories.

All 30,815 episodes passed candidate-serializer token admission: 729–201,014
tokens, under the production 262,144 cap. Candidate IDs verified 32–35. Local
suite: 64 passed/10 tokenizer-GPU skips. Bash syntax, source/manifest SHA256
checks, and `sbatch --test-only` passed. Free shared disk was about 4 TiB.

## Mandatory checks inside the new allocation

Before any production updates, the job revalidates four-rank distributed
recovery equivalence and profiles three updates each of the formerly failing
joint batch and the eight-slot 627-observation worst case. The production
Trainer uses the candidate policy overlay in both profiles. It must complete
without OOM and peak reserved memory must be <=78 GiB on every rank. Failure
stops the job before full training; a two-GPU smoke is not substituted for this
four-GPU gate. The real Trainer additionally checks 7,704/232 schedule values.

Output locking and source/manifest hashes protect the run. Existing jobs 4593
and 4595 and interactive allocation 4620 were not cancelled or modified. No
Hub upload or additional full-training job is scheduled by this submission.
