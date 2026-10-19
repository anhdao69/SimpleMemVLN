# G1 future-conditioned Omega teacher

Status: implemented and under four-H100 qualification on worker-2. Full training has not started. The user subsequently requested that the uploaded G3+A1 run be resumed and completed first, followed automatically by this experiment. Qualification and queue evidence will be appended below.

## Scientific definition

This branch implements **G1**, following the user's explicit correction of the initial G3 request. It starts from `origin/c2_vggt_omega_g1` at `6833bee`. The abandoned G3-future worktrees were restored before this implementation and never launched training. The user's original local worktree and all historical training outputs are untouched.

For a sampled current observation index `t`, the frozen VGGT-Omega aggregator receives the ordered pair `[I_t, I_(t+1)]`. The tensor is `[B,2,3,448,592]`, where each batch item is a separate pair. Omega's frame and inter-frame attention run jointly over those two views. The current output is selected **after the complete aggregator**, using cached aggregator layer 23 and frame slot zero. There are 17 special tokens and 28×37 image tokens. The future slot is never averaged into the target or selected as the supervised spatial grid.

The target transformation remains original G1: discard special tokens; adaptive average pooling to 15×20; per-token, no-affine LayerNorm over 2048 channels with epsilon 1e-6; add the original fixed sinusoidal spatial grid scaled by 0.5. The target is an ordinary detached FP32 tensor, safe for autograd's saved tensors. Omega stays in evaluation and inference mode, all its parameters are frozen, and it is owned by the trainer rather than the student/DDP/optimizer.

The original G1 student is unchanged: layer-23 spatial hidden rows enter the 2560→4096→2048 training-only MLP (18,885,632 parameters), with cosine distance averaged over 300 positions. This is **not** the G3 delayed native-state read objective. Original G1 lambda_max=0.2, 771-update ramp, head LR 1e-4, and positional term are retained. There is no PCA, offline descriptor extraction, GeoAddress adapter, episodic M5 memory, or additional inference input in this experiment.

## Temporal pairing and causal boundary

The deterministic original 16-stratum sampler runs unchanged, keyed by collected RGB paths, dataset and scene, seed 429. A sampled current frame is retained only when `t+1<T`. The terminal frame is not clamped, wrapped, duplicated, or replaced by a new sample. Navigation still trains on every original episode frame and action. A one-frame episode therefore has zero auxiliary support and normal navigation supervision.

The dataset serializes the student episode exactly as before, then prepares teacher-only CPU image pairs. Repeated image paths within the sampled union are decoded once. Current visual-row indices are derived from the original serialized positions after filtering unavailable future pairs, keeping target order and row order identical. The trainer consumes `teacher_images` before its normal BF16 input conversion, removes raw teacher images before the student forward, and attaches detached FP32 targets afterward.

Complete-episode student training naturally contains later observations at their later positions. Causal Window8 attention and GDN recurrence prevent those positions from influencing the supervised current hidden state. Tests check zero auxiliary input gradients after the current visual rows, current-prefix output invariance to later student observations, and identical student forward values when teacher targets change, both with and without activation checkpointing. Future targets may alter learned gradients; that is the intended training-only supervision, not an inference-time future observation.

## Loss, distributed normalization, and compatibility

The existing G1 loss machinery is unchanged. Navigation is normalized by global action count. The auxiliary loss is normalized by the total **retained current-frame count across all four ranks and both accumulation slots**, with the existing world-size correction and no extra accumulation division. Empty support retains differentiable participation of all auxiliary parameters. The four-GPU diagnostic additionally checks each exposure's expected retained support and the logged global auxiliary denominator.

Old configs default to `single_frame`; they keep their original teacher identity and behavior. The original `vggt_omega.py` is unchanged. New configs explicitly select `teacher_context: current_future_v1` and positive integer `teacher_future_offset: 1`. The new identity nests the original teacher/source/preprocessing/position identity and adds the new adapter hash, future offset, ordered frame slots, selected current slot, and terminal policy. A changed offset or teacher contract is rejected before loading the model. Normal full-state checkpoint contracts also include the research configuration, dataset, sampler and topology.

Serving uses the original G1 export: remove the training projector and research metadata, retain native Window8 text behavior. Neither Omega nor future images are required by the exported policy.

## Recipe and isolation

One complete joint R2R + English RxR15deg epoch from original Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`: 30,815 episodes, 3,128,624 actions, 3,852 optimizer updates, warmup 116, cosine floor 0.1, seed 429, backbone LR 5e-6, head LR 1e-4, weight decay .01, Adam betas .9/.95, clipping 1. Global batch is eight episodes: one complete episode per rank, four ranks, accumulation two. Full BPTT and original activation checkpointing are preserved. Canonical action history remains enabled, as in original G1.

Four H100 80GB GPUs on worker-2 allocation 4925, 60 CPUs and 768 GiB host RAM. Runtime starts from the optimized G1 profile: ZeRO-2 DeepSpeedCPUAdam offload, pinned activation arena, staged BF16 student pixels, vision batch eight, attention batch 32, eight persistent data-loader workers per rank, OMP/MKL four. The teacher initially uses eight **pairs** per microbatch (16 frames), compared with original G1's 16 single frames. Pair batch sizes are benchmarked without changing student batch or training semantics.

Campaign root: `/mnt/data/vmo-ai-task/anhdh35/g1-future-worker2-20261019`. Separate diagnostics, frozen production source, outputs, locks and supervisor logs prevent collisions. RAM and disk guards are retained. No existing jobs are stopped. The final queue must require successful G3+A1 completion, rather than merely its process exit, before starting G1-future.

## Validation evidence so far

- Test-first CPU run: ten intended failures before implementation; four baseline G1 tests passed, CUDA RNG check skipped on CPU.
- After implementation: 14 passed, one CUDA-only skip.
- Full worker-2 GPU suite: **362 passed**, 115 dependency warnings, 49.87 seconds. Includes actual pinned Omega paired attention and CUDA RNG isolation.
- Focused real-Omega/trainer/identity/causal checks: **17 passed**, 26.12 seconds.
- Additional current-prefix/student-target causality tests: **2 passed**, 5.04 seconds, checkpointing off/on.
- Independent fresh-context whole-branch review of `6833bee..2ddcbce`: no actionable correctness issues. Reviewer did not independently execute the GPU qualification.
- Four-GPU mixed-episode smoke at pair batch eight: passed three updates, finite losses/gradients, correct frame denominators, zero OOM. Detailed measurements and remaining qualification results follow when complete.

Passing these checks establishes tested implementation behavior, not guaranteed navigation improvement or guaranteed failure-free training. No navigation evaluation of this new model exists yet.

## Follow-up: restoring the uploaded G3+A1 run first

The user confirmed `anhdao69/g3a1_training` as the predecessor. HF revision `47043b9693d30f5f4b423e9fddf780f92a6f189e` contains checkpoint2100, 66.46 GiB including four ZeRO-2 optimizer shards. All SHA256SUMS entries passed. The exact production source is877815e; the provided git bundle's latest260e6d8 adds its report and a diagnostic. This branch is not on GitHub. Latest GitHub refs were fetched, and the G1 base remains6833bee. Resuming an existing optimizer uses its exact packaged source rather than importing unrelated upstream changes.

Checkpoint audit confirms global step2100/3852, epoch0.5451713396, five Adam parameter groups on each of four ranks all at step2100, ZeRO2 partition count4, no saved overflow, expected betas/epsilon, matching scheduler/group LRs, G3 head and four GeoAddress tensors, and four RNG files. Its next backbone LR is2.5308736594e-6. The scheduler is owned by HF and saved in `scheduler.pt`; the DeepSpeed model-state scheduler field is intentionally null. An initial audit incorrectly assumed that field was populated; inspecting the installed Trainer's restoration path confirmed it restores the separate scheduler. No checkpoint data or training code was changed to resolve that audit assumption.

The recipe remains G3+A1: G3 delayed native GDN24 reads plus A1 rank8 GeoAddress adapters, online unpositioned Omega targets, lambda0.1 for each objective, ramp771 already complete, headLR1e-4, adapterLR5e-5, full one-epoch cosine decay3852/116, batch8, full BPTT, Window8/canonical history. There are1752 remaining updates. The supplied preparation tool permits only image-root/teacher-path changes and runtime loader workers2→8. The worker-2 launcher adds an early recovery save at2110, then continues ordinary100-update saves; this does not change optimization.

### Migration hazard found and preserved sampling identity

The original image root is `/groups/yshang/an221229/data/JanusVLN_data`, with images immediately under `R2R/…` and `RxR/…`. The local physical root is `/mnt/data/vmo-ai-task/anhdh35/JanusVLN/data/trajectory_data`. A first preparation using the parent `data` root correctly failed its image-existence check; no training was launched.

More importantly, the frozen source's `collection_trajectory_key` splits path strings at `/trajectory_data/`. The original Newton paths lack that marker and therefore hash the **full original absolute path**. Rebasing directly to the ordinary local root would silently change G3 source/read pairs and A1 query-cell choices. The supplied root-rebase preparation tool does not detect this.

To keep the production source and every original sampling key unchanged, this migration uses an isolated filesystem alias:

```
<resume-root>/image-alias/trajectory_data//groups/yshang/an221229/data/JanusVLN_data
```

The directory entry at that location is a symlink to the real local `trajectory_data` directory. The deliberate double slash after the marker makes the sampler's unchanged string split yield the exact original `/groups/…` path, including its leading slash. Image I/O follows the symlink to the existing dataset. Manifest byte hashes and resume contracts pin this spelling: **do not normalize these manifest paths**. The original downloaded checkpoint/manifest remain untouched. Prepared checkpoint metadata is replaced atomically, even when large state files are hardlinked.

Full-corpus sampling and image-existence verification is required before launch. It must show identical trajectory keys for all30815 episodes and identical deterministic sampling inputs; explicit pair/query-cell comparisons cover examples across both datasets. This is a compatibility alias, not a change to the training sampler, targets, data order, or model.

### Recovery and serving diagnostic distinction

Four-rank fresh-process recovery passed at update 3: global relative L2 2.6672e-8, maximum absolute difference 2.3842e-7; the entire auxiliary head is bitwise identical, both losses are 0.4154797792, and all three projector biases demonstrably update. All 22 recovery inventory files validate.

The initial serving diagnostic reached an untrained greedy-output failure: the three-update model emitted 16 tokens without the required assistant terminator. This is not a trained navigation evaluation. The diagnostic now has an explicit smoke-only option to record that exact failure and still require identical reset outcomes and logits; all other errors remain fatal. The default remains strict. Source/export logits must still match exactly and 17-frame controlled replay on both datasets must still meet the original BF16 tolerance, including retry and Window8 eviction. Production serving and model code are unchanged. Four unit tests cover valid output, strict default, the explicit incomplete-response case, and propagation of unrelated failures. The original failed evidence is retained.
