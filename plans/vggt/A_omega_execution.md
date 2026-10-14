> **Latest user scope (2026-10-14): finish A0 qualification only, resume G3, then run A0 immediately after G3. A1 is deferred and must not be queued.**

# A0/A1 online Omega implementation plan

> Implement inline using the executing-plans workflow; final independent review.

Goal: implement the user's A0/A1 plans in branch v2_vggt_omega_a1, using online Omega without PCA, preserving G3 production.
Spec: A0_address_control.md, A1_geoaddress_sv.md, common_infrastructure.md. User's online/no-PCA and subsequent direct-native-Q/K instructions override the old offline target section.

Architecture: shared image-only rank8 Q/K adapters at native decoder24, raw heads0/1. Independent base initialization, same seed429 and one-epoch joint R2R/RxR schedule as current G3 (3852 updates, warmup116, global8). A0 navigation only. A1 frozen single-view FP32 online Omega descriptors and relation loss; direct native Q/K KL, explicitly selected by the user. Native Window8 and full BPTT. All runtime adapters survive export; teacher and training-only modules do not.

Global constraints: no running source modifications, no GPU contention with G3; WORKER-2 ONLY per latest user instruction; use allocation4925; the user authorized a checkpoint2500 pause for qualification before G3 completes. Four H100s, CPU optimizer and bounded pinned arena inherited from qualified G3. No zero-error guarantee: explicit measured qualification and fail-closed launch.

Review focus: image mask at cached prefill/continuation/decode; raw head grouping; stale metadata under checkpoint recomputation; earlier source-key gradients and causal positions; strict serving export/resume identities; queued gate failures cannot start production.

## Tasks
- [x] 1. Pure rank8 adapter: zero equality, mask/head select, finite/shape guards, gradients open in two updates, RNG isolation. Tests before implementation.
- [x] 2. Native per-instance version-checked adapter runtime: exact native cached modes, no tensor hooks, checkpoint-local relation scalar. Tiny CPU and pinned H100 parity.
- [x] 3. A1 online unpositioned descriptors, deterministic 32-query rectangle sampling, delayed pairs, objective; teacher detached, source keys differentiable. Empty support/DDP normalization.
- [x] 4. Dataset, trainer optimizer grouping, config/versioned serialization/export and recovery integration. Independent A0/A1 hashes and same schedule. Run regression suite.
- [x] 5. A0 qualification (A1 deferred): teacher TRAIN/held-out QC, mixed/longest/threshold optimizer smokes, resume and trained export; record runtime/memory. Queue A0 then A1 only after successful G3 and all required gates.
- [x] 6. Independent branch review, evidence/report, updated plans, commit/push user branch; verify queue/job state.

## Decisions / progress
- Existing user plan plus explicit implementation request authorizes execution; avoid redundant approval stages.
- Start from clean G3 source 9d658c2 in isolated sibling worktree, not its trained weights.
- User explicitly selected direct native Q/K relation loss after clarification. No PCA or MLP is needed because relations have equal shape despite different feature widths.

- Worker-1 qualification4989 cancelled after user restricted execution to worker-2. Preserve its initial evidence separately; all final qualification will run on worker-2.
- Independent review identified final inference metadata leak: added inference_sample and failing-then-passing regression; runtime source requires final suite/qualification rerun.
- Queue changed to persistent login watcher plus worker-2 Slurm steps in allocation4925. No held batch jobs or other-worker fallback. Every launch checks qualification source hash and full successful predecessor checkpoint.

- User authorized pausing G3 after COMPLETE checkpoint2500, debugging A0/A1 on its fourH100s, then resuming G3. Pause watcher launched on login with strict checkpoint validation; only Slurm step4925.33 is cancelled, allocation4925 retained.
- Initial worker-1 suite382passed. A0 mixed3updates passed17.52s/update,36.26GiB active; A1 passed18.96s/update,39.65GiB active. These are short smoke timings, not full epoch estimates.
- Initial teacher QC rejected five of614400 feature coordinates on overly strict pointwise batch1-vs8 tolerance (largest0.000157). Replacement check directly tests SAME-B neighbor perturbation with exact equality; separately measures batch-size roundoff (relativeL2<=1e-4) and actual relation probability agreement (atol1e-5/rtol1e-4). Pending GPU rerun, failed evidence preserved.
- Local new-suite22passed/1CUDA-skip; targeted regression32passed/2GPU-skips. Mac full suite had4environment failures (Slurm binary, DeepSpeedCPUAdam, unavailable teacher path, unavailable cached model),342passed/34skips; cluster suite passed all382 before final inference fix.

- 2026-10-14 04:58UTC: checkpoint2500 validated and G3 training step4925.33 stopped; allocation4925 retained. Immutable source3d61bfa deployed to worker-2 campaign. Persistent controller resumes original G3 after qualification (including failure/90-minute timeout). VPN interruption delayed upload briefly but source readiness succeeded before timeout.
- Additional review regressions now pass: source manifest checked throughout qualification, and validated clean partial checkpoints remain incomplete instead of crashing the recovery queue. Latest local targeted suite34passed/2CUDA-skips. Worker-2 final gates running/pending; do not mark ready until their evidence is complete.

- First worker-2 archive suite387passed/4failed: legacy episodic helper required `.git`. Fixea9ade5 verified with4affected tests from an actual archive and10targeted tests. G3 automatic recovery step4925.35 was stopped only to continue corrected qualification from complete checkpoint2500. Retry campaign uses sourceea9ade5 with a90minute controller deadline and automatic original-G3 resume. Production queue starts only on a passing source-bound seal.

- User narrowed scope during recovery qualification: A0 only now, A1 next time. Old two-arm controller2770412 was terminated and verified absent. A0-only persistent controller2774700 monitors the remaining recovery/export gates, skips optional timing and all A1 runs, resumes original G3, and waits to launch only A0 after verified full G3 completion.
- A0 full recovery passed: identical final loss0.4155527353; whole-model relativeL2=9.4311e-9; adapter relativeL2=3.7286e-8; all22 checkpoint files valid; all2048 entries of each up projection moved betweenupdates2and3. Export then exposed a legacy G1/G3-only allowlist. Fix0c22a7b adds strict A0/A1 runtime adapter preservation; six export/finalization regression tests passed locally. Final H100 controller2775724 reruns only exporter tests/audit, verifies training source is byte-identical to the passed snapshot, resumes G3, then queues only A0. No repeated stress training and no A1 run.

- Final A0 exporter audit PASSED on both R2R/RxR: exact source/export logits and greedy/reset/retry, correct Window8 eviction; replay max/RMS R2R0.53125/0.06046 andRxR0.5625/0.05019. Six exporter/finalization tests passed on H100. Qualified source0c22a7b differs from training-tested sourceea9ade5 only in exporter utility; verified inheritance is recorded in the A0-only seal.
- Active A0-only controller2775724 submitted G3 step4925.40 and entered A0-queued-after-G3. It launches only A0, from base, one joint epoch, fourH100s/worker-2; A1 deferred. Actual advancing G3 progress still being verified.

- Final verification2026-10-14T05:38:27UTC: all four G3 ranks reached2508, zero CUDAOOMs, original source resumed in4925.40 onworker-2. A0-only controller2775724 alive and queued. A0 starts after3852 successful G3 updates; A1 is excluded. Detailed evidence/report: `reports/vggt_five/A0_A1/IMPLEMENTATION_2026-10-14.md`. Tasks4–6 are complete for the user's reduced A0 scope; A1 remains deferred.
