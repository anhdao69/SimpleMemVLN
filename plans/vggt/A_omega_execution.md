# A0/A1 online Omega implementation plan

> Implement inline using the executing-plans workflow; final independent review.

Goal: implement the user's A0/A1 plans in branch v2_vggt_omega_a1, using online Omega without PCA, preserving G3 production.
Spec: A0_address_control.md, A1_geoaddress_sv.md, common_infrastructure.md. User's online/no-PCA and subsequent direct-native-Q/K instructions override the old offline target section.

Architecture: shared image-only rank8 Q/K adapters at native decoder24, raw heads0/1. Independent base initialization, same seed429 and one-epoch joint R2R/RxR schedule as current G3 (3852 updates, warmup116, global8). A0 navigation only. A1 frozen single-view FP32 online Omega descriptors and relation loss; direct native Q/K KL, explicitly selected by the user. Native Window8 and full BPTT. All runtime adapters survive export; teacher and training-only modules do not.

Global constraints: no running source modifications, no GPU contention with G3; WORKER-2 ONLY per latest user instruction; queue all remaining gates after G3 in allocation4925. Four H100s, CPU optimizer and bounded pinned arena inherited from qualified G3. No zero-error guarantee: explicit measured qualification and fail-closed launch.

Review focus: image mask at cached prefill/continuation/decode; raw head grouping; stale metadata under checkpoint recomputation; earlier source-key gradients and causal positions; strict serving export/resume identities; queued gate failures cannot start production.

## Tasks
- [x] 1. Pure rank8 adapter: zero equality, mask/head select, finite/shape guards, gradients open in two updates, RNG isolation. Tests before implementation.
- [x] 2. Native per-instance version-checked adapter runtime: exact native cached modes, no tensor hooks, checkpoint-local relation scalar. Tiny CPU and pinned H100 parity.
- [x] 3. A1 online unpositioned descriptors, deterministic 32-query rectangle sampling, delayed pairs, objective; teacher detached, source keys differentiable. Empty support/DDP normalization.
- [ ] 4. Dataset, trainer optimizer grouping, config/versioned serialization/export and recovery integration. Independent A0/A1 hashes and same schedule. Run regression suite.
- [ ] 5. Qualification: teacher TRAIN/held-out QC, mixed/longest/threshold optimizer smokes, resume and trained export; record runtime/memory. Queue A0 then A1 only after successful G3 and all required gates.
- [ ] 6. Independent branch review, evidence/report, updated plans, commit/push user branch; verify queue/job state.

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
