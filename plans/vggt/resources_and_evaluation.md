> **R1 update (2026-10-15):** `R1_lane_state.md` and `R1_omega_execution.md` supersede historical R1 cache/PCA/dimension/schedule defaults: online frozen FP32 Omega, full2048 regional targets and training-only student MLP, exact registered four-lane R0 architecture, one joint R2R + English RxR15 epoch. R0 E1 used a two-epoch LR schedule; report this confound.

> **G1 update (2026-10-13):** User requests ONLINE VGGT-Omega, no pre-extraction/PCA, the SpatialForcing projector, and4H100 tests. For G1, `G1_current_alignment.md` supersedes conflicting offline/PCA/two-GPU sections below. Other arms are not implemented by this task.

# Resource qualification, existing-baseline reuse and evaluation

## 1. This is not a launch authorization or a new baseline campaign

The only five new policy IDs are G1/G3/A0/A1/R1. Existing V0/R0 may be read, replayed and evaluated; never retrained automatically. No cancellation of existing work, no new Hub publication, no environment upgrade and no modified running source snapshot is authorized by this document.

Use an isolated worktree and immutable source for each actual run. The currently listed Slurm IDs in historical reports are not current available allocations. Discover live resource ownership in the execution environment. Report what is unavailable; do not infer access from an old job number.

## 2. Proposed profiles, subject to baseline registry and smoke results

| Profile | GPUs | Episodes/rank | GAS | Global episodes | Optimizer |
|---|---:|---:|---:|---:|---|
| H4 | 4xH10080 | 1 | 2 | 8 | existing ZeRO2 GPU optimizer first |
| H2 | 2xH10080 | 1 | 4 | 8 | existing ZeRO2 CPU optimizer offload |

Keep per-action objective and full BPTT. CPU optimizer offload/number of ranks can alter numerical trajectories; same global batch alone does not prove identical optimization. Prefer the registered baseline topology for the corresponding arms when practical, and require A0/A1 the same profile. If G1/G3 and R1 references used different historical profiles, compare each against its own matched reference and label cross-family comparisons as systems comparisons.

Start from the inspected H2 settings where appropriate: 8 OMP/MKL threads/rank, 4 loader workers/rank, prefetch2, attention batch8, vision microbatch4, BF16 pixel staging, loader pinning disabled, bounded pinned activation arena. These were reported for non-VGGT runs, not automatically safe defaults for all new code. Do not select topology by a short easy episode alone.

## 3. Host RAM risks requiring explicit accounting

The new branch's pinned arena addresses allocator caching/fork-copy risks; it is not proof that every graph fits. Record per job/cgroup: total memory.current, anon, file/inactive_file, dirty/writeback, pinned/locked totals where observable, OOM/oom_kill counters and available cgroup memory.max. Host working-set estimates do not replace the hard cgroup limit.

New sources of memory: teacher-cache worker copies, active target tensors, original/probe-expanded Q/K/V, native/lane activations, optimizer states, checkpoint serialization and graph references held by closures. No persistent lists of hidden states, no entire target shards loaded into every worker, no teacher model in training ranks, no parent process retaining a decoded corpus.

### 3.1 Forty probe rows can still require large scratch copies

Although probes add <=40 sequence rows, a naive `cat`/index-insert of full `[1,200000,32,128]` Q/K/V arrays can allocate several additional GiB. Do not budget only40 rows or claim overhead below5% without profiling.

Preferred implementation constructs repeated/expanded arrays once, after retaining raw head views, instead of creating full repeated arrays and copying them again. An engineering fallback is **sparse scan state taps** at the same anchors: run prepared Q/K/V segments, request differentiable final states, read `S.T@q` outside the kernel, carry state without detach, then concatenate original outputs. This computes the same objective and full-BPTT recurrence but changes finite-precision chunk boundaries/call costs. It is NOT TBPTT.

If needed, choose one backend through isolated correctness/resource tests before production, record it in the research checkpoint, and use the chosen backend consistently. Do not silently switch mid-run. Verify initial/final-state gradients in pinned FLA; no production Python recurrence fallback. R1's much smaller4-head lane has a different scratch profile than G3's32-head native state.

## 4. Mandatory qualification fixtures

Re-audit actual serialized lengths from full joint manifest. Historical guideposts: longest R2R183 observations/58,290 tokens; longest RxR627 observations/199,735 tokens; offload threshold65,536. These are not hardcoded admissions. Include cases immediately below/above the actual offload threshold.

Each structurally different arm must pass:

1. Real short mixed R2R/RxR forward/backward and parameter gradients.
2. Longest R2R, longest RxR and near-threshold episodes at normal full lengths.
3. Mixed-tail changing-length accumulation windows, not repeated one-size-only tests.
4. At least three optimizer updates with Adam state allocated by the second; measure all ranks.
5. Full recovery checkpoint save, new-process load and at least one resumed update on the SAME topology.
6. Trained serving export reset/retry/window eviction and complete-response replay.

Stress fixtures may repeat complete real episodes to fill global8 slots and must be labeled diagnostic, not unique corpus training. Never truncate, drop longest episodes or change gradient horizon to pass a gate. An emergency OOM stops only the owned process/job, preserves evidence and prompts a runtime fix/retest.

Suggested engineering limits: prefer <=76 GiB reserved on the H100s; hard ceiling <=78 GiB and below actual measured device capacity. Host working-set guard <=90% of the actual allocation limit, with explicit dirty/writeback accounting and zero new OOM/oom_kill counters. These are protective thresholds, not no-OOM guarantees. Keep independent GPU/host/disk failure reasons.

## 5. Production schedule and checkpoint progress

Default two-epoch trajectory7,704/232 per master, verified against actual corpus/exposure policy. Save E1/E2 plus two rolling complete recovery checkpoints. A pause at512 or E1 does not shorten/restart the LR schedule. Resume preserves optimizer/scheduler/RNG/sampler and topology. A0 and A1 are independent initializations from the same pretrained checkpoint, not checkpoint dependencies.

If existing baselines actually completed a different budget, bind the schedule before launch and compare same progress. Do not change all five to warm-start from V0 merely because its weights are available: that would add exposure absent from the reference. If matching is impossible with available artifacts, label the experiment a practical screen and state the confound; do not allocate V0/R0 retraining against the user's constraint.

No extra seeds or combined model are part of this plan. Optional diagnostics are not authority to create extra full policies.

## 6. Evaluation protocol

Use the existing `EVALUATION_SETUP.md` and model/simulator split with exact dataset/config identities. Preserve canonical text generation, actual selected-action feedback, primitive scales, STOP/cap behavior, RGB resolution and failure-in-denominator policy. Evaluate identical R2R IDs across arms, and a fixed declared English RxR15 set. Report panels as panels; never compare a partial/panel number as full RxR.

Before full rollout, same-history offline vs streamed checks cover real varied64/128-step histories, all response tokens including EOS, not just first-token logits. Raw unmodified logits must be captured even when fixture history is forced to match. Forced fixture agreement is not unconstrained action agreement. Qualification distinguishes FP64 math, BF16 score drift, token argmax flips and rollout success.

For G1/G3/R1 export, no teacher/query/projector calls in serving. A0/A1 adapters and R1 lanes persist as intended. For comparisons to V0/R0, use actual existing trained weights. No inference-only lane suppression is substituted for the trained baseline.

Metrics: SR, SPL, NE, oracle success, model STOP, forced cap STOP, failures, action counts, per-episode outcomes. Efficiency: measured target/episode throughput, update time on representative length distributions, total training GPU-hours including extraction/smokes, GPU/host peaks, p50/p95 policy latency on varied trajectories, state/KV bytes. Repeated-image cache-friendly timing is a diagnostic workload, not the only benchmark.

## 7. Selection and claims under a five-run budget

A0/A1 comparison is the strongest controlled effect because both are new matched runs. G1/G3/R1 use historical baseline artifacts; report any unresolved runtime/schedule difference. Select best practical fusion based on quality/cost and complete matched evaluation, not minimal auxiliary loss or arbitrary projection norm.

Use paired differences on the same episode IDs, bootstrap by scene if feasible, and show success/failure switches. Single seed does not measure training-seed variance. Frequent validation feedback creates selection bias; use a fixed diagnostic panel and a separate larger/full set for finalists when possible.

G3/R1 recall cosine must be accompanied by query-only and temporal-stratum diagnostics. A1 relation KL is not physical matching accuracy. A winning single-view VGGT arm does not prove multiview teacher benefits, teacher superiority over DINO, or a novel metric map.

## 8. Reporting directory

`reports/vggt_five/README.md`, `baselines.json`, `teacher_qc.json`, `cache_manifest.json`, and one directory per new ID with:

- design/resolved config, deviations and source/runtime hashes;
- startup/reload/recovery checks, rejected tests and warnings;
- sample/cache identities and valid auxiliary counts;
- training nav/aux/ramp/gradient/resource/exposure logs;
- E1/E2 checkpoints/provenance and served export schema;
- evaluation manifests/journals and aggregate comparisons.

Keep private data/credentials/weights out of Git unless publication explicitly authorized. Report actual status as proposed/implemented/tested/submitted/running/completed/evaluated, not all as “done”.
