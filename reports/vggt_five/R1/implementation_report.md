# R1: online Omega supervision of the existing dual-lane state

## Status and scope

Implementation, independent review and all18 H100 qualification gates passed. Production started on worker-2 in Slurm step4925.62 at2026-10-15T15:44:30Z. At2026-10-15T15:50:21Z, all four ranks had completed16updates. Checkpoint10 passed the production recovery validator:4ranks,22files,71,004,556,006bytes, correct one-epoch trainer state. Training continued after the save. The epoch itself remains running.

Branch: `v2_vggt_omega_r1`, based on completed A1 code `4f772d3`. Work is isolated from the original checkout and all completed models. The authorized run is one joint R2R + English RxR **15-degree** epoch from original Qwen3.5-4B weights, on four H100 GPUs in allocation4925, **worker-2 only**. R0, G1, G3, A0 and A1 are not retrained or overwritten.

## Scientific design and changes to the original plan

R1 retains the existing second-bank policy and directly supervises delayed visual information in its lane24 matrix. It does not replace native GDN memory or change the deployed lane architecture. Online, frozen, FP32, single-view `VGGT-Omega-1B-512` replaces the historical pre-extracted/PCA target proposal. Targets are the same unpositioned, normalized 2048-dimensional five-region descriptors used by G3. There is no PCA fitting, PCA projection, stored feature corpus, or cross-episode teacher context.

The student query/decoder are training-only MLPs. Query context comes from the existing native layer24 preparation, detached and normalized, with a region embedding and lag encoding. The decoder maps lane reads to the full 2048-dimensional teacher space. No learned teacher projector or teacher gradient is used.

The user explicitly selected the one-epoch schedule used by G1/G3/A1. The existing R0 epoch1 comparison checkpoint was trained under a two-epoch schedule. **R1 versus R0 epoch1 is therefore not a controlled auxiliary-only comparison:** warmup and LR decay differ. R1 is initialized from the original base and original lane recipe, never from completed R0/A1 weights.

## Baseline binding

The read-only registry is `evidence/baseline_registry.json`. It records full metadata, original source identity, file hashes, completed checkpoint paths, manifest hash, topology and schedule.

- R0 artifact: `/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN-streaming_text_dual-runs/window8_text_base_e2_20261007/train/checkpoint-3852`.
- R0 original source: `a7e442cbd3c43cf3ec238eaa527e9c67f5ce2408`.
- R0 E1 weights SHA-256: `cd7b36d160c19ebbb8fbdc92e3ce451f08a2a566a9a3507b382dd828b06ea6d4`.
- Joint manifest: `/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl`.
- Joint manifest SHA-256: `5deb425d2594ce96931dd6ce12bd6084b066cd44f04808c1dd3ee2e4b3672933`.

Qualification and immediate launch verify the manifest's actual path and bytes, and the baseline weights/navigation metadata. The registry validator also rejects changed lane layout, clock, initialization, output/history, base revision, seed and lane/backbone LR. Baseline weights are used only for diagnostics, not initialization.

## Lane state and causality

All four original lanes remain installed at decoder layers16/20/24/28. Each has two key heads repeated to four value heads, key/value dimension128, step-end clock, read-only zero prefix, half-lives8/24/64/192 steps, beta0.1, projection initialization standard deviation0.02, zero initial output projection, and lane seed429+layer.

Only the final existing separator token of a completed observation/action group writes and decays a lane state. At a pre-action anchor for step t, the visible lane state is the state after completed group t−1. The current action and current group's final write are unavailable. The selected historical source is at lag8..32.

At most eight source/read pairs are selected per episode, with five regions per pair (at most40 read rows). R1 inserts probe rows after lane QKV/SiLU/head repetition and effective write gating. Probe k is a fixed unit vector, v is zero, and effective g/beta are zero. Probes enter the lane kernel only; they do not enter native convolution, native GDN, token positions, attention, or the original role sequence. Probe reads bypass lane RMSNorm, z gate and output projection. Original rows retain their original postprocessing.

Native layer24 preparation/kernel executes once. The lane24 kernel executes once. The layer preserves the original BF16 residual grouping `hidden + (native + lane)`, followed by the original MLP/residual. Other installed lanes use the original wrappers. Empty-support episodes execute the original lane path and retain zero participation for every auxiliary parameter. Decoder checkpoint closures bind layer-specific arguments; full BPTT remains enabled.

Direct recall loss can train earlier lane writer parameters even while the lane output projection is zero. Tests demonstrate nonzero earlier writer/input gradients, and no future-input or detached native-query/teacher-target gradients. This does not imply only lane24 changes: shared writer inputs let gradients reach earlier native layers and earlier lanes. All language/lane weights remain trainable; the existing frozen vision setup is unchanged.

## Auxiliary head and targets

The native query context has32×128=4096 dimensions. Region and lag contribute32 dimensions each. Query MLP: affine-free LN4160 → Linear4160→512 → SiLU → Linear512→512 → four128-dimensional queries. Lane reads receive per-head FP32 RMS normalization, then a Linear512→1024 → GELU → Linear1024→2048 decoder. The head has **5,017,760 parameters**; this differs from the larger G3 native-state head and must not be described as identical capacity.

Teacher preprocessing and targets reuse the established Omega pipeline: RGB640×480, balanced512 preprocessing to448×592, single-view aggregator layer23, patch offset17, adaptive15×20 pooling and LN2048 before unchanged G3 region pooling. A CUDA test compares R1 targets against direct Omega aggregation plus the G3 region function with exact equality. Teacher parameters are frozen, targets are detached ordinary tensors, and no persistent target cache is added.

Identity hashes bind teacher weights, source, used preprocessing/descriptor code, R1 adapter and G3 region code. R1 adopts the portable A1 descriptor identity excluding unused positional values; it does not modify old G1/G3 checkpoint identities.

## Training configuration

| Item | Setting |
|---|---|
| Base | Qwen3.5-4B, revision851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a |
| Data | 10,819 R2R + 19,996 English RxR15 episodes =30,815 |
| Exposure | One joint epoch, 3,852 optimizer updates |
| Batch | Four ranks × one complete episode × accumulation2 =global8 |
| Schedule | One-epoch cosine_with_min_lr, minimum ratio0.1; warmup116 |
| Peak LR | Backbone5e-6; all lanes1e-4; auxiliary head1e-4 |
| Auxiliary | Pair-normalized regional cosine loss; maximum0.1; ramp771updates |
| History | Canonical text actions, native Window8, guard5120 |
| Optimizer | ZeRO2, CPU optimizer offload, DeepSpeed CPUAdam |
| Activation memory | Decoder checkpointing and pinned CPU arena; no TBPTT |
| Teacher | Frozen resident FP32 Omega; batch8 |
| Loading/runtime | Four CPU threads/rank, eight workers/rank, vision batch8 |
| Checkpoints | First complete recovery save10, then every100; latest two rolling saves and complete epoch retained |

Attention batch32 is retained. On the identical mixed fixture, eight-update trials averaged17.1567s/update for32,17.2441 for16, and17.3905 for64; all passed at40.110GiB peak allocated memory. Differences below1.5% do not establish a robust gain from changing the existing profile. See `evidence/speed_selection.json`. Short profiler diagnostics retain the existing engineering harness (bounded update count and inherited science-common warmup); they are not a completed production schedule. The full launcher explicitly overrides this with one epoch,3852updates and116warmup, and its live schedule gate must pass. Runtime tuning does not shorten episodes, alter source/read sampling, truncate gradients, or change the objective. Save/resume restores model, optimizer, scheduler, rank RNG and sampling state; `auto` selects only completed checkpoints. An incomplete write is not eligible for resume.

## Implementation map

- `research/vggt_r1.py`: exact registry and lane/init contract.
- `research/vggt_r1_teacher.py`: online teacher identity and regional targets.
- `models/vggt_r1_forward.py`: pure native preparation and causal lane-only probe path.
- `models/vggt_g3_heads.py`: independent context-head count, retaining G3 defaults.
- `models/research_forward.py`, `models/nav_model.py`: selected-layer dispatch and original roles through all installed lanes.
- `data/episode_dataset.py`, `train/train_episode.py`: reuse G3 pair sampling and instantiate the online R1 teacher.
- `scripts/vln/profile_vggt_g1.py`: R1 diagnostic profiling with bound baseline metadata.
- `scripts/vln/export_vggt_g1.py`, `check_vggt_export.py`: strict export retaining every installed lane, and same-history score/response/retry/reset/eviction audits.
- `scripts/vln/qualify_vggt_r1.py`, `queue_vggt_r1.py`: source-bound18-gate qualification and durable worker-2 launch.
- `train/launch_r1_omega_joint_e1.sh`: protected one-epoch full training with automatic resume.

## Verification evidence so far

Core snapshot `cac27d9` passed **423 tests on the H100 server**; earlier qualification source `56353571c024a6655b590ee75bb03550b16bead7` passed **425 tests,118warnings,no skips in93.57seconds** in its initial qualification suite, including actual FLA/native parity, lane writer causality, checkpoint gradients and online teacher target equivalence. The initial four-GPU mixed smoke completed three optimizer updates with finite loss/gradient evidence and no new GPU or host OOM events. Its worst allocated/reserved GPU memory was40.110/50.377GiB. Job memory sampling reached625.442GiB including reclaimable file cache; sampled working set was283.878GiB. These are mixed-fixture measurements, not the longest-episode claim.

Final audit source `dca86b6` passed432 server tests with118warnings in68.05seconds in its fresh initial suite and432tests/118warnings in65.69seconds in its final suite, with no skips. All18 qualification records have returncode0;9 are explicitly carried unchanged training/recovery evidence and9 are freshly executed.

Local full-suite environment gaps (no Slurm, DeepSpeed CPUAdam, pinned teacher assets/model) are recorded separately from defects; corresponding server tests passed. The independent reviewer approved core/launch code after fixes for longest-episode checkpoint-save coverage and manifest binding. The trained R0 baseline-serving audit passed on17steps/50response tokens per dataset: R2R RMS0.023997/maxabsolute0.75; RxR RMS0.022874/maxabsolute0.5. Both greedy actions were valid, reload scores/outcomes exact, reset/retry exact, and Window8 eviction correct. The baseline-serving audit establishes consistency within the current pinned runtime; it does not claim an independent rerun of the historical original environment.

Longest RxR qualification passed on627-action/199,735-token episodes with three full optimizer updates and complete four-rank saves. Worst GPU allocation was65.7004GiB, reserved76.0918GiB; supervisor peak host working set456.9798GiB including writes, with no new OOM events. Longest R2R qualification also passed on183-action/58,290-token episodes, with37.2823GiB allocated and47.2715GiB reserved. The166.28s RxR endpoint interval includes a full checkpoint after every update and is not an ordinary training-throughput estimate.

Completed qualification gates: full suite; exact trained R0 loading; trained R0 streaming replay; mixed and empty-support DDP; longest RxR and longest R2R with full saves; activation-offload threshold; mixed extremes; uninterrupted save reference; fresh-process resume; weight/loss recovery comparison; trained serving export; representative R2R and RxR timing; actual joint timing; four-rank pair normalization; final full suite. Saving profiles additionally validate complete four-rank checkpoint inventories. Production requires all18 gate records, exact source hashes, dataset/baseline integrity, worker2 ownership, and idle GPU step state.

## BF16 export audit investigation and revised qualification

The original direct optimized-offline versus streaming gate failed on R2R: one of12,416,000 vocabulary scores exceeded the unchanged absolute/relative threshold by0.008125. Streaming was2.625 versus optimized offline1.8125, max absolute0.8125 and RMS0.0587968. The token was far below the winning response score. RxR passed the original direct gate. This failed attempt remains in the original qualification log and in the revised report as `legacy_direct_streaming_vs_optimized.passed=false`; it is not reclassified as a pass.

Source/export optimized offline logits were already bitwise exact. Diagnostics showed all four lane outputs exactly zero before the first STEP_END at the offending row, identical vision features across microbatches8/1, and identical same-hidden LM-head results across batching. Offline attention32 versus1 localized the difference to optimized offline attention batching. The streaming path already returns through the sequential Window8 attention call and ignores that batching option.

The established `check_text_step_lane.py` protocol uses sequential offline attention as a reference. The revised audit independently compares optimized offline and streaming to that reference using the same absolute0.75/relative0.03/RMS0.15 tolerances, and additionally requires bitwise-exact source/export forced-history streaming over17steps (including eviction). It restores the original config after the reference call. Production attention remains32, and no model, objective, teacher, schedule or training runtime changed. Approximate equality is not transitive: passing both reference comparisons does not establish that the original direct comparison passes.

The actual revised H100 diagnostic passed both datasets:

| Dataset | Optimized vs sequential max / RMS | Streaming vs sequential max / RMS | Exact source/export streaming |
|---|---|---|---|
| R2R |0.46875 /0.046945|0.6875 /0.062588|Pass|
| RxR15 |0.46875 /0.040423|0.578125 /0.051882|Pass|

Raw scores and forced action sequences, greedy scores/outcomes, reset, retry and Window8 eviction were checked. The three-update R2R base-policy diagnostic does not yet produce a valid bounded greedy action; the same quality failure occurs exactly in source/export/reset and is preserved explicitly. This is not a navigation accuracy evaluation.

Final qualification uses immutable audit revision `dca86b64742f779d6f4b5ebdb50e21dd1a5e7d3b`. Nine passed training/recovery gates are carried from the previous runtime only after verifying that every tracked runtime source hash is identical except the two audit/orchestration scripts, all command arguments match apart from output paths, fixture bytes match their recorded hashes, and full saved checkpoints remain complete. The carry records identify original manifests/log hashes. The failed original export is never carried. Both full suites, trained baseline checks, revised serving/export checks and remaining timing/distributed gates execute anew. This avoids repeating unchanged long-episode training while preserving explicit source/evidence lineage.

A staging launch exited before Python because the source archive transfer was incomplete; no training ran in that attempt. Before retrying, all274 tracked runtime/test files were checked against local SHA-256 values. The failed staging log is preserved separately from the successful verified-source qualification log.

## Deployment and limits

Serving export strips only `research_aux`; it keeps all four lanes and exact lane metadata. Total lane parameters are21,054,000. The four FP32 recurrent lane matrices occupy1,048,576bytes (1MiB) per episode, excluding temporary working buffers. R1's training-only head and Omega teacher add no deployed modules relative to R0. R0 itself remains more expensive than native V0.

Launch uses a detached login-side watcher, so an SSH/VPN drop does not terminate training. It waits for non-shell allocation steps to finish and never falls back to another worker. A source/arm qualification mismatch blocks launch. Host guard is691.2GiB within768GiB, minimum free disk220GiB; the chosen profile must pass allocated≤72GiB and reserved≤78GiB qualification. These checks reduce risk; they are not a proof against every hardware, filesystem or future runtime failure.

Navigation success-rate gains and serving latency are unmeasured; throughput figures here describe training. A short base-policy smoke may not yet generate a valid bounded action; serving audits preserve and compare that outcome rather than representing it as task success. The head can exploit historical information in its detached query context, as already qualified in the G3 plan. R1/R0 schedule mismatch and head-capacity differences must accompany later model comparisons.

## Measured dataset throughput

The same four-H100 production profile completed eight-update representative tests per dataset. Measurements use synchronized optimizer-update endpoint intervals after the first update; startup and checkpoint I/O are excluded.

| Dataset | Mean seconds/update | Observed interval range | Peak allocated GPU GiB | Extrapolated separate one-epoch compute |
|---|---:|---:|---:|---:|
| R2R |9.0744|6.38–11.82|26.8468|3.41 hours /1,353 updates|
| English RxR15 |18.7191|9.09–28.48|38.6526|13.00 hours /2,500 updates|
| Actual joint manifest (12 updates) |12.3078|5.79–31.43|50.5731|13.17 hours /3,852 updates|

These are small fixed representative fixtures, not completed standalone epochs. The actual mixed campaign can incur different rank imbalance and episode-length distributions; separate dataset estimates should not simply be added as its ETA. The joint trial covers96 complete episode exposures and11 endpoint intervals. Its13.17-hour compute extrapolation, with checkpoint overhead and substantial episode-length uncertainty, supports an initial planning range of14–18hours after launch; it is not a full-epoch measurement. A full checkpoint after every100updates adds I/O; the longest/save stress-test timings are deliberately excluded from steady training estimates.

## Final runtime, launch evidence and resume command

Verified launch evidence is in `evidence/production_launch.json`, `initial_progress.json`, `checkpoint10.json`, and `checkpoint10_trainer_state.json`. At2026-10-15T15:50:21Z (11:50:21America/New_York):

- Allocation4925, step4925.62, worker-2, four H100s; launcher started15:44:30UTC.
- All four ranks completed16updates and continued after checkpoint10.
- Four live schedule gates confirmed3852updates/116warmup. The training contract confirmed30815uniqueepisodes, globalbatch8, and token range700..199735.
- Checkpoint10 is complete:22listed files,71,004,556,006bytes. The shared production recovery validator passed. Trainer state records global_step10, max_steps3852, num_train_epochs1, epoch0.002596054.
- Step10 loss0.3360443 and gradient norm19.77109 are finite. Per-rank auxiliary exposure losses are finite and positive when support exists; the ramp is active after step0.
- Across the first16updates, maximum observed GPU allocation/reservation was50.5731/54.7754GiB. All rank GPU OOM counters and host OOM/kill counters remained zero. The production host has substantial reclaimable filesystem cache; total cgroup usage must not be mistaken for nonreclaimable working memory.
- Initial finish range:14–18hours after launch, roughly2026-10-16 01:45–05:45America/New_York. This is a short-sample estimate, not a guarantee; update from a longer production window later.

No model weights are published as a completed R1 epoch yet. The training process is independent of this SSH session and remains active.

The queue is `/mnt/data/vmo-ai-task/anhdh35/r1-omega-worker2-20261015`. Its immutable runtime source is `source-dca86b6`, full commit `dca86b64742f779d6f4b5ebdb50e21dd1a5e7d3b`. The final queue/output root is its `attempt2` subdirectory; original qualification and diagnostics are preserved at the campaign root. Qualification artifacts are in `qualification`; production output will be `r1_joint_e1/train`. The successful qualification/queue log is `attempt2/queue-launch-verified.log`, state is `queue_state.json`, and every training restart has a separate `r1_joint_e1/supervisor-*/command.log`.

After a cancellation, first obtain a new four-H100/60CPU/768GiB interactive allocation on worker-2. Then restart the same durable queue from the login node (substitute its actual allocation ID):

```bash
R1_ROOT=/mnt/data/vmo-ai-task/anhdh35/r1-omega-worker2-20261015
R1_SOURCE="$R1_ROOT/source-dca86b6"
R1_RUN_ROOT="$R1_ROOT/attempt2"
nohup env G1_CODE_DIR="$R1_SOURCE" \
  bash "$R1_SOURCE/train/g1_omega_env.sh" \
  python -m scripts.vln.queue_vggt_r1 \
  --source "$R1_SOURCE" --root "$R1_RUN_ROOT" --allocation NEW_ALLOCATION_ID \
  > "$R1_RUN_ROOT/queue-resume-$(date -u +%Y%m%dT%H%M%S).log" 2>&1 < /dev/null &
```

Keep the source, world size, manifest, teacher and output root unchanged. The existing seal is revalidated, completed epochs are not repeated, and interrupted production resumes only from a complete recovery checkpoint. Updates after the last complete checkpoint repeat. The queue/launcher locks reject duplicate processes targeting this run. This command is for interrupted production after successful qualification; a failed qualification needs investigation and a new audited attempt rather than bypassing its seal.
