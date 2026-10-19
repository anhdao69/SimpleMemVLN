# G1 future-conditioned Omega teacher

Status: G1-future implementation and four-H100 qualification are complete. The verified G3+A1 continuation was launched first on worker-2 in step4925.138 at2026-10-19 01:41UTC. G1-future is the second automatic stage and has not yet trained on the full manifest. The live continuation verification is recorded below.

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

## Validation evidence

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

## Measured four-H100 diagnostics

Each row uses full episodes and global batch8. B8 means eight teacher pairs per forward; B4 is the only alternative runtime knob benchmarked. Representative measurements are warm short-fixture throughput, not a prediction of end-to-end full-corpus wall time. Repeated fixture updates may also include boundary housekeeping.

| Diagnostic | Updates | Seconds/update | Peak GPU reserved GiB | Result |
|---|---:|---:|---:|---|
| Mixed, B8 pairs | 3 | 20.49 | 49.92 | pass |
| Mixed, B4 pairs | 3 | 20.63 | 49.92 | pass |
| Terminal-only eight-record fixture | 2 | 2.23 | 28.24 | pass; zero auxiliary support |
| R2R representative | 8 | 12.09 | 35.80 | pass |
| RxR representative | 8 | 21.51 | 48.42 | pass |
| R2R longest,183frames | 3 | 25.25 | 46.76 | pass |
| RxR longest,627frames/199735tokens | 3 | 76.64 | 76.40 | pass |

The B4/B8 timing difference is too small to establish a speed improvement; B8 is retained. At the longest RxR fixture, peak allocated memory was61.79GiB and peak working host memory393.92GiB; no CUDA/cgroup OOM occurred. Total cgroup usage can approach768GiB because clean reclaimable file cache is included; the existing supervisor guards working memory separately. Memory limit/reclaim events occurred, so this is not a claim of unlimited headroom or no cache pressure. No host-wide cache dropping or changes to other users' jobs were performed.

Four-rank distributed-loss validation passed unequal support, all-empty support, and uniform support with GAS2; maximum gradient error versus the single globally normalized reference was2.98e-8.

The supplied checkpoint migration completed all23 state-file hash checks and the trainer's own metadata/data-order/world-size/teacher preflight. Full manifest image checks passed3128624/3128624; all30815 old/new sampling keys matched, with explicit128pair-list and1024query-set comparisons. A canonical-root byte comparison also proves this is the established joint R2R/RxR15deg manifest rather than a different dataset.

## Final qualification and queue

Final GPU regression:371 passed,115 dependency warnings,50.45s. The exact G3+A1 archived production source passed419 tests,118 warnings,49.21s, including its real-H100 fused GDN/GeoAddress tests. Four additional export-diagnostic tests passed in6.61s. No tests were skipped in these final GPU suites. A mixed-extreme three-update stress passed at54.89s/update,61.79GiB allocated/68.42GiB reserved,344.12GiB host working set, zero OOM. This covers heterogeneous sequence-length transitions in addition to repeated longest episodes.

Final serving export: source/export logits are bitwise identical. Seventeen-frame controlled replay gives R2R maximum absolute difference0.59375/RMS0.05543 and RxR0.75/0.04959, within the original0.75 absolute/0.03relative/RMS0.15 protocol. Both datasets pass retry idempotence, exact reset and Window8 eviction. The smoke RxR policy produces a valid greedy response. The smoke R2R policy still lacks a terminator; the diagnostic records `greedy_valid=false` and checks that its failure and every captured logit are identical after reset. This qualification verifies implementation/export equivalence, not navigation quality.

The frozen G1 production checkout is `g1-future-worker2-20261019/production/source-1fcc2af`, commit1fcc2af. Earlier hardware runs recorded dirty development snapshots or55741d0; every recorded `src/` file was hashed against the final frozen source before sealing qualification. The final change after55741d0 affects only the export diagnostic and its tests. The final report/operations commit can therefore be newer than the immutable training checkout without changing the qualified model code. `evidence/future/qualification.json` binds source hashes, all ten passed experiment results, test suites, recovery and export records.

Queue root: `/mnt/data/vmo-ai-task/anhdh35/g3a1-then-g1-future-20261019`. Its persistent controller runs under Slurm step4925.138 on worker-2, within the existing4H100/60CPU/768GiB allocation4925. It survives client SSH disconnection. G3+A1 uses `g3a1-resume-worker2-20261019/production`; G1-future uses a fresh `g1-future-worker2-20261019/production/g1_future_joint_e1`. No original worktree, historical output, shared environment or other user's job was modified.

The queue checks pinned source and input hashes before each stage, records an immutable queue identity, takes one queue lock plus per-run launcher locks, and updates a heartbeat every15s. Restart waits for any surviving owned launcher before deciding whether to skip a completed stage or resume an incomplete one. Transition requires a validated four-rank epoch-boundary checkpoint3852, matching navigation contract, trainer/export global step=max step3852 and epoch>=1, a nonempty final model, and finite final forward reference. A failed child or partial successful exit blocks the next stage. The normal checkpoint auto-resume logic selects only complete snapshots; it does not restart silently from base when recovery files are incomplete.

G3+A1 keeps original source877815e, full optimizer/scheduler/RNG, data order and sampling. G1-future begins independently from base, one complete joint epoch with3852 updates, not from G3+A1 weights. Operational commands and audit scripts are in `operations/`; input hashes/results are in `evidence/`. If the Slurm allocation itself is canceled, both processes stop: resume the same queue on a new four-H100 worker-2 allocation by changing only the `--jobid` in the launch invocation. The controller and checkpoints are restartable; this does not claim the cluster will recreate a canceled interactive allocation automatically.

## Verified full-run continuation

At2026-10-19 01:46:56UTC, all four ranks completed step2115/3852, leaving1737 updates. The first resumed optimizer update was2101. Across the15 completed resumed updates, all120 episode exposures are unique and none overlap the pre-checkpoint2100 exposures. Navigation/state/relation losses are finite, both original auxiliary weights remain0.1, and no CUDA OOM was recorded. Observed peak allocated GPU memory37.31GiB, reserved46.80GiB, and host working set453.57GiB during this initial continuation window. These are initial-window measurements, not whole-epoch maxima or an ETA based on a representative full-corpus sample. The progress bar's rate immediately after resume is misleading because it includes the restored2100 steps; it must not be used for finish-time estimates.

Checkpoint2110 was newly saved and independently audited: all22 recovery inventory files validate, trainer global_step2110/max_steps3852/epoch0.5477673936, all four ZeRO2 optimizer shards have five groups at step2110 with the original beta/epsilon settings and exact scheduler-matching learning rates, DeepSpeed model global_steps2110/world_size4, G3 head plus four GeoAddress tensors present, and all four RNG states present. Training continued beyond the save. Evidence is in `evidence/g3a1/checkpoint2110_audit.json`, `checkpoint2110-RECOVERY_COMPLETE.json`, and `resumed_progress_audit.json`. The persistent queue heartbeat was fresh and its active stage wasG3+A1.

G1-future has not begun the full run yet. Its automatic start follows verified G3+A1 epoch completion; it will then use the already-qualified source, manifest and fresh base initialization. Neither full-run final weights nor navigation results are claimed at this point.
