> **Latest scope:** A0 only for qualification/queue; A1 implementation retained but deferred at the user’s request. Earlier two-arm plans below are historical and superseded.

# A0/A1 online Omega implementation and qualification ledger

Current status: A0 is qualified on worker-2 and queued after G3. A1 is implemented but deferred and not queued. G3 resumed as Slurm step4925.40 and all four ranks reached update2508 with zero CUDA OOMs. The chronological ledger below preserves failed attempts and their fixes.

## Scope and decisions

Branch: `v2_vggt_omega_a1`, isolated from the running G3 source. Runtime implementation commit `85f0f5fa8b55579288248a3fa1cca8759eca4e46`; qualification/recovery safeguards commit `3d61bfa2d8300975de2e4b6db0c282180cb893d8`; immutable report compatibility commit `ea9ade569d205c20f23b04ba7a9a50a954203f88`.

A0 and A1 independently initialize the original Qwen3.5-4B base. They use Window8, the existing text action/history policy, full episode BPTT, seed429, and one joint R2R/RxR epoch. They are not initialized from trained G3 weights. Global batch8 comprises one episode per GPU per microstep and two gradient accumulation microsteps across four GPUs. Expected training length3852 optimizer updates, warmup116.

The user explicitly selected direct native Q/K relation supervision after discussing an MLP alternative. Consequently A1 contains neither PCA nor a feature-width MLP. The comparison occurs between equal-shaped similarity distributions, so the native128-wide vectors and teacher2048-wide vectors need not have the same width.

## Runtime architecture

Both arms add the same trainable GeoAddress module at decoder layer24, a native gated delta-rule layer. Two independent bias-free rank8 residual paths map normalized hidden states of width2560 to Q and K updates, respectively. Each emits256 values, reshaped into the first two raw key/query heads of width128. The module has45056 parameters. Additions occur only at image tokens, after the causal convolution/SiLU and raw Q/K split, before native head expansion and kernel L2 normalization. Text token Q/K, all other heads, V, gates, convolution and state update equations remain native.

Down projections use deterministic normal initialization and up projections start at zero. This makes the initial adapted mixer equal to the original native mixer. The adapter has its own RNG context, avoiding changes to the surrounding model's random stream. Up projections receive gradients immediately; down projections begin receiving them after the up projections move.

The runtime installation is per model instance, checks the pinned native source, preserves native parameter names, and refuses incompatible double installation. Prefill, cached multi-token continuation, and one-token recurrent decode use the appropriate original cache equations. Serving receives an explicit image mask constructed from serialized token IDs. A missing mask is an error, avoiding silently different training/serving behavior.

The new serializer contract `vln_append_only_chat_geoaddress_v1` makes incompatible older loaders fail explicitly. GeoAddress weights survive inference export. Online teacher data and training-only auxiliary modules do not. The relation callback is confined to the selected layer's checkpointed forward rather than installed as a persistent graph hook.

## A1 online target and loss

The teacher is frozen VGGT-Omega-1B-512, loaded online with pinned source and weight hashes. It runs FP32 under no-grad. Each image is an independent single-view sequence (`[B,1,3,448,592]`), even when multiple images are batched. RGB640x480 images are resized with the established Omega preprocessing. Aggregator layer23 features exclude17 special tokens, then pool the28x37 patch grid to15x20 and normalize2048-wide cell descriptors. A1 adds no positional encoding; its identity explicitly distinguishes these unpositioned targets from G1.

For each episode, deterministic sampling selects up to eight delayed source/read pairs with lag8–32. Each read frame contributes32 spatial queries distributed over a4x8 partition of the15x20 grid. The source frame contributes all300 cells. At most16 unique teacher images are needed for eight pairs. Sampling uses collected trajectory identity and actual serializer image positions. Sources must precede queries.

Teacher probabilities are softmax(cosine similarity /0.1), shape `[pairs,32,300]`. Each of two selected native Q/K heads forms a distribution of the same shape, using the native FP32 L2 normalization convention and temperature0.1. The loss is KL(teacher || student), averaged over32 queries and two heads, summed across pairs, then normalized using the shared distributed pair-count machinery. Teacher descriptors/probabilities are detached. Both the selected current queries and earlier source keys retain their student gradients.

Empty auxiliary support is legal, with differentiable zero and consistent distributed collectives. A0 loads no teacher and has zero auxiliary weight. A1 ramps auxiliary weight to0.1 over771 updates. Native trainable policy weights retain learning rate5e-6; GeoAddress uses5e-5. The optimizer checks parameter coverage rather than allowing duplicate or omitted groups.

## Memory and speed configuration

The configuration inherits the qualified G3 setup: four H100s, ZeRO stage2 with vectorized DeepSpeed CPU AdamW, pinned host activation arena, full episode activation checkpointing, four CPU math threads, eight loader workers, vision microbatch8, Window8 attention batch32, teacher batch8, and a resident online teacher. This retains full BPTT rather than truncating gradients to improve speed.

The launch guard limits host working memory to691.2GiB of a768GiB allocation and checks at least220GiB free disk. Short smokes are insufficient evidence for the longest episode; qualification therefore includes the627-frame RxR fixture (199735 serialized tokens), the longest R2R fixture, mixed/extreme lengths and threshold cases. The qualification memory threshold is72GiB allocated per GPU, with the existing78GiB reservation ceiling checked separately.

## Evidence and unresolved gates

Initial tests ran in allocation4989 before the user restricted execution to worker-2. That allocation was cancelled and is no longer used. Its evidence remains under `/mnt/data/vmo-ai-task/anhdh35/a1-omega-20261014`.

| Check | Observed result | Limit |
|---|---|---|
| Initial cluster regression suite |382 passed |Before final inference metadata and queue fixes |
| A0 mixed fixture,3 updates |Pass;17.5179s/update;36.2637GiB allocated;46.285GiB reserved |Short fixture, not epoch ETA |
| A1 mixed fixture,3 updates |Pass;18.9599s/update;39.6472GiB allocated;49.9258GiB reserved;192 auxiliary pairs |Short fixture, not epoch ETA |
| Latest local targeted regression |34 passed,2 GPU-only skips |Mac CPU evidence only |
| Queue regression after review |7 passed including finalization check |Does not replace H100 training qualification |
| Initial teacher batch-size QC |Failed coordinatewise threshold:5/614400 values, max absolute difference0.0001567602 |Failed evidence retained; replacement checks not yet GPU-qualified |

The revised teacher check separates two different properties: identical batch shape with perturbed neighboring images must give exactly equal features for the unchanged image; batch-size1 versus8 may differ by FP32 kernel reduction order, but must satisfy relative feature L2<=1e-4 and probability agreement atol1e-5/rtol1e-4. It also checks relation permutation behavior and nonuniform distributions. Calibration and held-out diagnostic scenes are disjoint subsets of TRAIN, not a navigation validation benchmark. Heatmaps are labelled descriptor similarity, not physical correspondence.

Local full-suite failures were environment-specific (missing Slurm, DeepSpeedCPUAdam, remote teacher files, and cached base model). The corresponding initial cluster suite passed. Final cluster rerun is required for the current snapshot.

## Review fixes

1. Final reference inference could receive auxiliary metadata with its pair count removed. `inference_sample` now strips all auxiliary and teacher staging fields before reference inference. A regression reproduced the failure and passed after the fix. A saved GPU smoke must still exercise complete finalization/export.
2. A qualification seal originally lacked a stable source binding. Qualification now records an initial source hash manifest, verifies it before and after every gate, compares every profile's own source manifest, and seals only that exact source. Later production launches revalidate the seal and source hashes.
3. A successful supervisor with a valid partial checkpoint must not count as a completed epoch, nor crash the resume path. It now remains incomplete/waiting until full checkpoint3852 and epoch completion are verified.
4. Every GPU step explicitly uses one Slurm task and worker-2, avoiding inherited task multiplication or placement on another worker.

## G3 pause, qualification, and recovery

User authorized pausing G3 only after a complete recovery checkpoint2500. The server-side watcher validated that checkpoint's full recovery inventory and four-rank topology, then cancelled only training step4925.33. Interactive allocation4925 remains on worker-2. The separate controller waits for the immutable A0/A1 source snapshot, runs qualification, and resumes G3 from its ORIGINAL immutable launcher and existing output directory in a finally block. A10-minute source-upload deadline and90-minute qualification timeout avoid indefinitely pausing G3 during a VPN outage.

The G3 resume launcher automatically restores optimizer, scheduler, RNG, sampler and model state from the latest complete recovery checkpoint. It rewinds post-checkpoint reporting before continuing. The A0/A1 runtime is never substituted into G3's source tree.

Qualification includes current regression suite, teacher QC, mixed/empty/longest/threshold training, separate-process full recovery comparison, trained serving export replay, R2R/RxR timing, distributed unequal/empty pair-count checks, and a final regression suite. Passing three optimizer steps alone does not mark the branch ready.

The durable production queue waits for verified successful G3 completion, then launches A0 and A1 sequentially from base, one joint epoch each. It requires a passing source-bound qualification seal at every launch. Missing/incomplete predecessor state, failed tests, wrong allocation placement, or source mismatch block production. Reusing a run directory resumes its full recovery checkpoint; completed runs are not trained twice. Queue activation and final worker-2 qualification remain pending in this ledger.

## Original pending work (closed for A0; A1 deferred)

- A0 worker-2 stress, recovery and trained export completed; evidence is archived below.
- Original G3 training resumed and all four ranks advanced beyond2500.
- A0-only persistent queue is active; it will launch after verified G3 completion.
- Representative speed sweeps and A1 qualification/training are deferred by the user's final scope change.
- Plans/report and branch publication accompany this result.

## Worker-2 campaign updates

- Initial immutable snapshot run:387 passed,4 failed in52.97s. All four failures were the same legacy report helper calling Git in an archive without `.git`; no model training failure was diagnosed from these tests. Fixed report metadata to read `source.commit` for archives and represent unknown Git dirty status as null. Four affected tests passed from an actual local archive; ten targeted helper/queue tests passed from the worktree.
- The first controller automatically submitted G3 recovery as step4925.35 after that failed gate. The corrected snapshot was uploaded before restarting qualification; only that owned resume step was stopped to continue the user-authorized debugging pause from checkpoint2500.
- Corrected campaign root: `/mnt/data/vmo-ai-task/anhdh35/a01-omega-worker2-20261014-retry`. Immutable source: `source-ea9ade5`. Controller accepts up to five explicitly prepared corrected snapshots, waiting at most10minutes between failed attempts, with a90minute total qualification deadline. It always submits original G3 recovery in its finally block. If all gates pass, it starts the durable G3→A0→A1 watcher after observing the new G3 supervisor. It does not start production on failure.

- Corrected worker-2 initial suite passed; online teacher QC passed on all four panels. Maximum batch-size relative feature drift2.153e-6 and maximum relation probability drift9.686e-8. Neighbor perturbation at fixed batch size was exactly invariant. All1024 teacher rows were nonuniform.
- Worker-2 A0 mixed3-update optimizer smoke passed at16.3135s/update with36.2637GiB peak allocated. Longest-RxR stress is next/in progress.

## Scope reduced to A0 only

The user requested finishing A0 quickly and leaving A1 for next time. The prior two-arm controller2770412 has been terminated and its absence verified. The A0-only controller2774700 continues the in-flight A0 recovery test, waits for the separate-process comparison and trained export gates, then cancels only qualification step4925.36. It creates an explicitly A0-only qualification record with the original source hashes and required gate results, resumes G3 from its unchanged source, and launches only A0 when G3 checkpoint3852 and successful supervisor prove completion. Optional timing runs and all A1 training are skipped. A1 is not qualified for production by this record.

The one-off controller lives at `/mnt/data/vmo-ai-task/anhdh35/a01-omega-worker2-20261014-retry/a0_finish_controller.py`. Its state is in `a0_state.json`, its log in `a0-controller.log`, and its G3 resume output in `g3-resume-a0-only.log`. It runs independently of SSH. Its original, explicit G3→A0→A1 queue is not active.

## Final A0 qualification result

Qualified production source: `0c22a7bdac465aedd5d1413fa08d9eedc12206c1`, deployed as `source-0c22a7b`. Source hashing proves that `scripts/vln/export_vggt_g1.py` is the ONLY code/config/launcher difference from the training-qualified sourceea9ade5. Therefore the preceding full training and recovery evidence is retained without rerunning unchanged stress training. The exporter change was independently tested on H100 and with the actual trained A0 checkpoint. Its A0/A1 branch requires the exact persistent GeoAddress contract and four correctly shaped adapter tensors; it rejects missing adapters or unexpected auxiliary heads.

A0 results on worker-2:

| Fixture | Updates | Steady seconds/update | Peak allocated GiB |
|---|---:|---:|---:|
| Mixed R2R/RxR |3|16.3135|36.2637|
| Longest RxR,627frames/199735tokens |3|72.5403|58.3399|
| Longest R2R |3|20.8553|33.4819|
| Offload threshold |3|23.7798|36.2637|
| Mixed extremes across ranks |3|48.2467|58.3398|

These are complete-episode stress fixtures and exclude checkpoint-heavy recovery from speed claims. Optional representative timing runs were cancelled under the user’s latest instruction; no new measured R2R-only or RxR-only epoch ETA is claimed. Training has no episode truncation. The joint epoch uses30815 unique records (10819R2R,19996RxR), with one padded distributed slot,3852updates, global8.

Full recovery comparison passed on four ranks and22 inventory files. Uninterrupted/resumed loss was exactly0.4155527353286743. Whole-model relativeL2 difference9.431103631e-9 (max1.192092896e-7); adapter relativeL2 difference3.728612104e-8. Both2048-element up projections changed in every element between updates2and3; the down projections also showed learning. Thus the recovery test verifies actual adapter training and continued updates rather than merely loading zero-initialized weights.

The latest exporter/finalization regression passed six tests on H100. Actual trained serving export passed R2R and RxR replay,17frames/50response tokens each. Source/export logits and greedy outcomes are exactly equal, reset and retry are exact, and Window8 eviction holds throughout. Offline/streaming BF16 replay: R2R maxabsolute0.53125/RMS0.06045646, RxR0.5625/0.05018968; both pass the existing atol0.75/rtol0.03 and RMS0.15 criteria. These parity checks are not navigation-quality evaluation.

The A0-only seal explicitly lists qualified_arms=[A0], deferred_arms=[A1], inherited and current source manifests, the one changed exporter path, and required passing gate records. The failed earlier export record is retained separately. The current controller is PID2775724, script `a0_export_controller.py`, log `a0-final-controller.log`. It submitted G3 resume step4925.40 using the ORIGINAL G3 source and run directory, then entered `A0-queued-after-G3`. It checks G3 every15seconds and launches A0 only after checkpoint3852, completed epoch, and successful supervisor. A0 starts from original base, not G3 weights, on four H100s in allocation4925, worker-2 only. The one-off controller is archived as operational evidence; its historical PID/step constants must not be blindly rerun.

The production command is the qualified source’s `train/launch_a_omega_joint_e1.sh A0 <campaign>/a0_joint_e1 <campaign>/a0_qualification_pass.json <G3-run>`, invoked by the controller with exactly one Slurm task, four GPUs and60CPUs. It uses the original base, shared one-epoch schedule, CPU optimizer offload, periodic complete recovery checkpoints and automatic full-state resume in the same run directory. A1 is not in the active sequence.

## Final operational verification

At2026-10-14T05:38:27UTC, all four G3 ranks had reached update2508 after recovery from2500, with zero CUDA OOMs. The allocation remained4925 on worker-2; its only training step was4925.40 (`g3_omega_resume`). Host working set was483.83GiB. Controller2775724 was alive and reported `A0-queued-after-G3`. See `evidence/g3_resumed_progress.json` and `evidence/a0_state.json`. This verifies actual resumed optimization, not just a submitted command. A0's future Slurm step will be created when G3 completes; no A1 step is queued.
