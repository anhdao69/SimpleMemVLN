# Joint Window8 candidate-logits submission

Job **4649**, partition main, pinned worker-1, dependency **afterok:4623**.
No running training job was cancelled or modified. The dependency makes this
eligible after FullContext candidate training succeeds; it is not a priority
reservation or a guarantee of immediate start. If 4623 fails, this job will not
automatically run. The scheduler's test-only distant start projection is not an ETA.

Campaign:
`/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/outputs/candidate_joint_window8_20261003-BaVpjn`.
Launcher: `train/slurm/candidate_joint_window8.slurm`.
Logs: `slurm-4649.log`, `train/train.log`, `train/action_metrics.jsonl`.

## Matched experiment

Immutable source is an exact SHA256-verified copy of job 4623's source,
commit `8a6a28acabd43d378e610345b2ebbe24547c37f8`. Same joint R2R + English
RxR_15deg manifest, 30,815 episodes. The resolved config comparison confirmed
identical model, observation/feedback, and training dictionaries. Only memory
and the attention runtime differ: Window8 keeps the instruction prefix and
eight complete step groups with persistent GDN state.

- Candidate logits, trainable pretrained LM rows, candidate-token feedback.
- Serializer `vln_candidate_logits_v1`; unweighted four-way CE baseline.
- Independently starts from Qwen3.5-4B @851bf6e, NOT the weights of job 4623.
- Four H100 GPUs, 24 CPUs, 768 GiB host RAM, 72 hours.
- Global batch 8 = four ranks × one episode × GAS2; 2 epochs; LR 5e-6.
- 7704 updates, 232 warmup updates, same cosine schedule and optimizer.
- Four loader workers/rank, OMP2; expandable CUDA allocator.
- Keep both epoch checkpoints; two rolling recovery points every 100 updates;
  final serving export. No CPU optimizer offload (matches four-GPU FullContext).

## Gates and validation

Local suite: **87 passed, 10 GPU/tokenizer skips**. Bash syntax passed.
Actual staged server source passed config validation at world size four and
the matched-recipe comparison. Source/manifest hashes passed; 2.4 TiB shared
disk was available at submission (450 GiB minimum is checked again at startup).
Slurm test-only accepted the request. Cached batch-script SHA256 matched the
submitted launcher; resources, node, and dependency were checked before release.

No new GPU smoke was run before submission because the target allocation is
still in use by 4623. The new job must pass distributed checkpoint/recovery
checks and fresh three-update profiles of both the mixed failed batch and the
all-longest 627-observation batch using the Window8 candidate overlay. Every
rank must remain at or below 78 GiB reserved. Any failure prevents production.
No FullContext or text-policy memory PASS is reused for this new experiment.
