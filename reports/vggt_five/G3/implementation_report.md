# G3: delayed native-state supervision with online VGGT-Omega

Status: implementation and qualification passed: **366 tests,118warnings,48.30seconds, no skips**; eight four-GPU training diagnostics passed, along with recovery, distributed reduction and serving export. The full one-epoch production run is active in **Slurm4925.33 onworker-2** from immutable source22593831c80b742ee77b478ce7cb76fb44bc328b. All four ranks completed at least14updates, and checkpoint10 is validated complete with22files. No navigation-quality improvement is claimed from smoke tests.

## Scope and provenance

This branch, `c2_vggt_omega_g3`, starts from the completed G1 implementation at `67b206b`. Work is isolated in sibling local/server checkouts; the original user checkout and G1's immutable training source are untouched. G1 completed its one joint epoch at3852 updates with supervisorPASS, exit0 and no new OOM events. Its checkpoint3852 inventory contains22 files and trainer epoch1.0. G3 GPU qualification began only after that successful exit.

The requested online/no-PCA change preserves G3's native-state read mechanism. It does not turn G3 into G1 current-frame alignment. Training uses native Window8 Qwen text with canonical action-text history, frozen vision/merger, trainable decoder, full episodes and full BPTT. G3 adds neither a dual lane nor an episodic bank. The policy remains the native architecture after serving export.

The production scope follows G1: original Qwen3.5-4B base, one joint R2R+RxR epoch, four H10080GB GPUs, one complete episode/rank/microstep and GAS2 (eight episodes/update). Dataset manifest contains30815 episodes and3128624 actions: R2R10819/631244; RxR19996/2497380. The one repeated final record fills30816 distributed slots. No truncation or dropped long episodes is used.

## Target construction and online ownership

The teacher is the same pinned Omega model/source used by G1:

- Omega source commit `ad02778c052e636a623ef425377f342ee41c4fa6`.
- Omega weights SHA256 `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`.
- Qwen base revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Joint manifest SHA256 `5deb425d2594ce96931dd6ce12bd6084b066cd44f04808c1dd3ee2e4b3672933`.

CPU loader workers decode only selected source images for teacher work. The Trainer owns a frozen eval-mode FP32 teacher per GPU, outside the student/DDP/optimizer/state dictionary. Independent views have shape[B,1,3,448,592]; there is no multi-view teacher context. Aggregator layer23 returns17 special tokens and28x37 spatial patches with2048 channels. The returned offset is checked and used, patches are adaptively pooled to15x20, and each token receives no-affine FP32 LayerNorm2048 with epsilon1e-6.

G3 adds no positional target term. Four quadrants are rows0:7/7:15 crossed with cols0:10/10:20. The fifth target is the mean of all300 tokens, correctly weighting the unequal-area quadrants. Tokens are not L2-normalized before regional averaging; cosine normalization happens at loss time. There is no pre-extraction, PCA or trainable teacher compressor.

Only unique source frames used by selected pairs are evaluated, at most8 per episode. Repeated pairs referring to one source share its target, while navigation episodes remain distinct. Final regional targets occupy at most8x5x2048x4=327680bytes (~0.313MiB), before transient dense pooling. Targets are ordinary detached FP32 tensors, cloned outside inference_mode before student backward. Teacher assets are reconstructed and identity-checked after resume.

The G3 identity nests the unchanged G1 adapter identity for provenance and adds its own unpositioned-region target version, wrapper hash and regional/sampling code hash. The nested G1 positional-grid hash describes inherited source provenance; the top-level G3 target mode and wrapper explicitly disable that position addition. This preserves the G1 implementation and resume contract.

## Pair sampling and causal anchors

The fixed base sample uses min(16,T) temporal strata and SHA256 draws. Candidate pairs use sampled source tau and a read t satisfying8<=t-tau<=32. Stable hash ranking selects3/3/2 pairs from lag bins8–15/16–23/24–32, then fills unused quota from remaining candidates, capped at8 unique pairs. Execution order is(t,tau,region). Repeated read anchors are legal.

As in G1, trajectory identity uses dataset, scene and ordered normalized collected RGB paths. Instruction/UID changes with identical paths do not change samples. Physically identical trajectories copied into different directories are not claimed to be deduplicated; no verified physical-alias registry exists in the manifest.

For action t, the anchor is one token before its first supervised response target. It therefore includes the current observation/cue but excludes the current gold action. Source images8–32 steps earlier are outside directly retained Window8 visual K/V. Lower layers and nearby contextualized tokens may still carry old information; the mechanism is not a proof that all recalled information resides solely in one matrix.

## Query, state probe and decoder

Only native GDN layer24 is supervised. All real Q/K/V projections and convolution/SiLU are computed before probe insertion. Native post-convolution q at each anchor uses the additive-epsilon L2 rule(sum of squares+1e-6), repeats to32 value heads and is detached. Its4096 values are concatenated with a learned32-dimensional region embedding and a fixed32-dimensional lag sin/cos vector. The query network is:

```
LayerNorm(4160, affine=False, eps=1e-6)
Linear(4160,512) -> SiLU -> Linear(512,4096)
reshape(32,128)
```

Each pair creates5 probe rows, at most40 per episode. Probe q is learned; k is the unit vector[1,0,...], v is zero, and effective beta and log-decay g are literal zeros. Probe rows never enter convolution, Qwen token inputs, MRoPE, attention masks/KV or labels. The native FLA chunk kernel normalizes q/k once and produces the read `S_anchor.T @ normalized(q) / sqrt(128)` without changing the recurrent state. The original real-token rows are gathered before native gated RMSNorm/z, output projection, residual and MLP. Upper layers see the original sequence length.

Probe reads[P,5,32,128] receive FP32 per-head RMS normalization(eps1e-6, no affine), flatten to4096, then:

```
Linear(4096,1024) -> GELU -> Linear(1024,2048)
```

The original G3 decoder's1024 hidden width is retained; only its PCA-target output256 becomes2048. This is the learned student-side dimension conversion requested by the user. Copying G1's2560-input spatial projector would change G3's distinct state-read interface. Total query/decoder/region parameters:10526368. Matrices and region embeddings initialize Normal(0,.02), biases zero, with an isolated CPU generator seed429 that preserves student CPU/CUDA RNG.

## Loss, gradients and checkpointing

FP32 cosine distance is averaged over5 regions per pair and summed over valid pairs. Navigation action loss retains its original independent numerator and denominator. Across four ranks and both accumulation slots:

```
L = navigation_sum / global_action_count
  + lambda(completed_updates) * auxiliary_pair_sum / global_pair_count
lambda(u) = 0.1 * min(1, u/771)
```

The existing world-size factor cancels DDP gradient averaging, without an extra GAS division. Short episodes with no valid pairs still train navigation; their auxiliary branch returns a differentiable zero involving every auxiliary parameter. Empty global support is valid. Head LR is1e-4 and backbone LR5e-6; production warmup116 updates and cosine decay follow the one-epoch3852-update schedule.

Auxiliary gradients reach earlier native K/V, write strengths and decay through the differentiable state recurrence. Detaching query context prevents its direct gradient path into the supplied native q. No later-token gradient is permitted for an isolated earlier anchor loss. Query context can carry information even with a fixed identity state; reconstruction alone is not evidence of useful causal memory.

The selected decoder layer executes inside a single nonreentrant checkpoint closure, with bound per-layer selection/metadata. Only the original hidden tensor is an explicit checkpoint/offload input; small targets/layout tensors are closure values. No global hooks, latest-output caches or full-state snapshots are used. The production path requires the pinned FLA chunk kernel and refuses the slow reference; the FP64 recurrence exists only in tests.

## Runtime and qualification evidence

Initial profile reuses G1: resident FP32 teacher, teacher batch8(G3 maximum), attention batch32 with Window8 unchanged, vision batch8, eight loader workers/rank, four OMP/MKL threads, BF16 student pixel staging, pinned-arena activation offload, DeepSpeed ZeRO2 with CPUAdam optimizer, and no per-update malloc_trim. The separate matching CUDA12.9.1 toolkit and node-local CPUAdam cache are reused; no shared packages or driver changes are made.

The interactive allocation is4925 onworker-2:4H100,60CPUs,768GiB RAM. The supervisor guards its owned process group at691.2GiB working set and checks disk headroom. File cache is reclaimable and reported separately from working memory.

Completed measured evidence:

| Check | Result |
|---|---|
| Independent FP64 recurrence | Real outputs/state/navigation gradients preserved; exact no-write/no-decay; repeated anchors and future-gradient checks passed |
| Actual FLA32x128 BF16 | Real-output relativeL2 .002869; final state .003316; probe/reference .004162; gradient errors .000485–.004523 |
| Native probe-free GDN | Output equality exact |
| Native checkpoint on/off | Output, auxiliary loss, hidden and head gradients equal |
| Four-GPU mixed complete episodes |3updates,24exposures,192pairs;17.016s/update;39.809GiB peak activeGPU;PASS |
| Longest RxR |3updates/24exposures, all8pairs per episode;74.572s/update,67.874GiB activeGPU, noOOM;PASS |

BF16 insertion changes FLA chunk boundaries, so augmented-vs-native parity is numerical rather than bitwise. Predeclared gates are2% relative output/state/read error and4% gradient error on the controlled kernel fixture. Serving source-vs-export equality is a different, exact check with probes disabled on both sides.

Evidence root: `/mnt/data/vmo-ai-task/anhdh35/g3-omega-20261013`. Every diagnostic records source/config/fixture hashes, complete exposures, losses, memory and owned supervisor exit state. The initial `mixed-initial` attempt used a wrong fixture path and failed before training; it is retained as historical failure evidence. The corrected `mixed-smoke` passed. The profiler now rejects a nonexistent fixture before allocating a run.

## Recovery, export and launch

Recovery retains native weights, all `research_aux.state_head.*` parameters, optimizer/scheduler, all-rank RNG/sampler state and teacher/data/config contracts. The fresh-process checkpoint2-to3 comparison passed on all four ranks: auxiliary weights bitwise identical, whole-model relativeL2 9.410618e-9 and maximum difference1.192093e-7, with exact final loss0.4165976941585541. All22 checkpoint inventory files were validated. Query and decoder biases changed in all512/4096/1024/2048 elements between updates2 and3, demonstrating auxiliary learning beyond matrix weight decay. Teacher-free export/replay passed on both datasets. Source/export logits and greedy score/outcome equality are exact, reset/retry equality is exact and all17steps preserve Window8 eviction. R2R50response-token replay: maxabsolute .71875,RMS .0563032; RxR50tokens: maxabsolute .8125,RMS .0457611. Both satisfy the unchanged combined atol.75/rtol.03 and RMS.15 gates; .75 is not a standalone maximum-error cap. The three-update R2R greedy policy is not yet action-valid, as explicitly recorded below. Serving export removes exactly `research_aux.*` and the research config; no broad missing-key exemption is permitted.

Production launcher: `train/launch_g3_omega_joint_e1.sh`. It uses one-epoch policy/runtime config, a per-run flock, unique restart supervisor logs, recovery every100updates and first checkpoint10. Resume selects a complete checkpoint only. An immutable source archive has been launched; live branch edits cannot change its imported code. The launch was recorded at2026-10-13T20:21:20UTC; startup and checkpoint10 verification passed as recorded below.

## Scientific limits

These checks establish implementation, numerical, resource and recovery behavior, not downstream navigation gains. The proposed frozen-policy zero-state/shuffled-state/query-only and teacher-baseline panels, equal-progress comparisons and closed-loop navigation evaluation remain post-training scientific work. Their results must not be inferred from finite auxiliary loss. G3 versus G1 is a method-level comparison: layer interface, target density and decoder differ. No trained recent-control arm or undeclared policy run is added.

## Independent review

A fresh-context read-only reviewer examined all changes and new files against67b206b and compared native GDN preparation with the installed Transformers source. No Critical, Important or actionable Minor findings were reported. Review does not replace runtime evidence; completed recovery/export/regression and immutable launch remain separate gates.

The teacher-batch4 alternative measured16.905s/update on the same mixed fixture versus17.016 forbatch8. This0.7% difference is too small to establish a speed improvement. Batch8 remains selected because it handles all selected source frames in one independent-view call and is already longest-episode qualified. G3 preserves the rest of G1's optimized settings.

## Export-audit diagnosis and operational failure history

The original export audit required a valid free-running canonical action from a checkpoint trained for only3updates from base. That policy instead generated16 ordinary-language tokens and reached the existing response limit. This was not an export discrepancy: separate source and exported inference produced exactly the same tokens and full-score SHA256 `8da9553977cefdbbd25b0cda605fec69c1760843bf1fd2c76985a6e4d8666267`. The corrected audit explicitly records this bounded policy-quality failure and requires identical source/export/reset outcomes and scores. Other runtime errors remain fatal. Controlled17-frame response replay still has the original .75/.03/RMS.15 tolerances. No training, serving budget or model behavior was changed to make the diagnostic pass.

One isolated single-GPU diagnosis command omitted `--ntasks=1` while requesting4CPUs inside a60-CPU allocation. Slurm inherited multiple tasks, causing duplicate model copies and CUDA OOMs in that diagnostic. The corrected single-task run completed and confirmed the equality above. This launch error is retained in `greedy-diagnosis.log`; corrected evidence is `greedy-diagnosis-single.log`. It is distinct from the four-GPU training stress tests, which completed without CUDA/cgroup OOM. Remaining qualification and production commands explicitly specify one Slurm task; torchrun creates the four training ranks. The interactive allocation remains running.

## Final qualification and duration estimates

All366 regression tests passed with118warnings and no skips in48.30seconds, including the real pinned Omega regional-target test. Actual G3 regional targets equal independently pooled, normalized and region-averaged Omega output exactly, and contain no positional term. The four-rank NCCL/DDP/GAS test covers unequal, completely empty and equal pair support; maximum gradient error versus the global-mean reference is2.384186e-7. All57 source files exactly match those used by the longest training qualification.

| Four-H100 fixture | Warm s/update | Projected separate epoch | Peak active GPU GiB |
|---|---:|---:|---:|
| R2R16 length-quantile episodes,8updates |9.0635 |3.4064hours/1353updates |26.6886 |
| RxR16 length-quantile episodes,8updates |18.2886 |12.7004hours/2500updates |38.5985 |
| Mixed empty/nonempty pair support,3updates |15.5768 |Not a corpus projection |39.8086 |
| Longest RxR repeated across all ranks,3updates |74.5723 |Stress test only |67.8741 |

Representative estimates discard the first two updates and use rank-completion endpoint intervals, including loading/teacher work. They are warm repeated-fixture projections, not completed R2R-only/RxR-only training runs. G1's comparable rates were9.97 and19.24s/update: G3 is about9.1% and4.9% faster on these fixtures, respectively. The longest stress case is about1.5% slower than G1's73.45s/update. Small differences depend on cache/storage/rank shape histories.

The completed G1 joint production epoch took14.28hours, materially faster than its original short cold-data projection. Combining this actual history with G3's fixture timings gives an initial **14–18hour planning range for the joint G3 epoch**, to be revised from sustained production progress. This is a planning estimate, not a measured completed G3 epoch or confidence interval. Checkpoint-heavy recovery timings are excluded from ordinary-step speed claims.

Evidence: [benchmark_summary.json](evidence/benchmark_summary.json), [recovery.json](evidence/recovery.json), [g3_export_check.json](evidence/g3_export_check.json), [full-suite.txt](evidence/full-suite.txt), [distributed-pairs.txt](evidence/distributed-pairs.txt). The failed original export attempt remains on the server; the corrected final qualification record is [final_qualification.json](evidence/final_qualification.json).

## Production launch and recovery command

The run was submitted at **2026-10-13 20:21:20UTC /16:21:20America/New_York**, in allocation4925, step4925.33, onworker-2, four H100s/60CPUs/768GiB RAM. The explicit Slurm task count is1; torchrun creates4ranks. G1's successful exit and the absence of other active GPU steps were verified before launching.

- Source commit: `22593831c80b742ee77b478ce7cb76fb44bc328b`.
- Run root: `/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN-c2_vggt_omega_g3-runs/g3_omega_joint_e1_20261013`.
- Immutable source: `<run-root>/source`, archived from Git, read-only, with exact hashes in `<run-root>/source_manifest.json`.
- Outputs: `<run-root>/train`; launch log `<run-root>/slurm-launch.log`; each restart creates its own `<run-root>/supervisor-*/command.log` and resource record.
- Initialization is the original Qwen base. No smoke-test or G1 adaptation weights are loaded into this new run.

To resume after allocation cancellation, use the same snapshot and run directory inside a new four-GPU/768GiB allocation:

```bash
R=/mnt/data/vmo-ai-task/anhdh35
RUN="$R/SimpleMemVLN-c2_vggt_omega_g3-runs/g3_omega_joint_e1_20261013"
srun --jobid=<new-job-id> --overlap --ntasks=1 --gpus=4 --cpus-per-task=60 \
  bash "$RUN/source/train/launch_g3_omega_joint_e1.sh" "$RUN"
```

The complete checkpoint includes optimizer, scheduler, all-rank RNG, sampler and auxiliary state. Unsaved updates since the latest complete checkpoint must be repeated. Serving exports intentionally lack this recovery state. Keep world size, manifest, teacher/config contract and source snapshot consistent. The launcher lock rejects duplicate starts into the same output directory.

## Verified production state at handoff

All four ranks completed at least **14/3852 updates**. Navigation/auxiliary losses observed at startup are finite, with zero CUDA OOMs. Peak initial active GPU allocation was52.10GiB; sampled working set408.57GiB stays below the691.2GiB supervisor guard. The longest prelaunch stress test reached67.87GiB activeGPU. Total cgroup usage can approach768GiB because it also includes hundreds of GiB of reclaimable file cache; it is not equivalent to nonreclaimable working memory.

All ranks printed `SCHEDULE_GATE_PASS (3852,116)`. The trainer audited all30815 records/3128624 actions, token lengths700–199735, four ranks/GAS2/globalbatch8, and installed30.48GiB activation arenas per rank. Recovery checkpoint10 is complete with all22 inventory files and four-rank state. See [production_launch.json](evidence/production_launch.json), [initial_progress.json](evidence/initial_progress.json) and [checkpoint10.json](evidence/checkpoint10.json). Subsequent rolling recovery saves occur every100updates, retaining two rolling snapshots plus epoch checkpoints.

The code and report are published on `c2_vggt_omega_g3`; the running source stays fixed at2259383 even when later documentation commits advance that branch. Updated plans and this report are also copied into the original local workspace's plans/vggt and reports/vggt_five/G3 folders. The completed G1 weights and its source snapshot are preserved.
