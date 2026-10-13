> **G3 online update (2026-10-13):** `G3_native_state.md` and `G3_omega_execution.md` replace this document's offline/PCA target requirements for G3. It now uses frozen online single-view Omega, full2048-dimensional regional targets, no added positional term, and four-H100 qualification. Other unimplemented arms retain their original specifications.

> Production steering (2026-10-13): the user authorized G1 on joint R2R+RxR for **one epoch** on four H100s after qualification. Use `G1_current_alignment.md` and `G1_omega_execution.md` for the active online-Omega design, selected runtime and declared one-epoch schedule override. Other proposed arms remain unchanged.

> **G1 update (2026-10-13):** User requests ONLINE VGGT-Omega, no pre-extraction/PCA, the SpatialForcing projector, and4H100 tests. For G1, `G1_current_alignment.md` supersedes conflicting offline/PCA/two-GPU sections below. Other arms are not implemented by this task.

# VGGT five-run campaign: design specification and master plan

**Status:** proposed implementation specification, not implemented production code.
**Date:** 2026-10-12.
**Goal:** compare five ways/configurations of injecting training-only VGGT information into the existing streaming VLN system, without retraining the user's completed V0/R0 baselines.
**Code base selected after live inspection:** `c2_h2_speed@2430fd235b0752e6119d1af809cb14182dc0d404`.
**Proposed isolated branch:** `c2_vggt_five`. This branch name was not present in the remote branch list inspected for this report.

## 1. Exact scope and authority

The ONLY new full policy-training runs authorized by this document are **G1, G3, A0, A1, R1**. V0 and R0 are existing reference artifacts, not launchable run configurations. Do not schedule a new V0/R0 to make the ablation table cleaner. Do not count a failed launch as an extra scientific arm, but retain its cost and artifacts. Tests, disposable resource smokes, feature extraction and frozen-policy probes are diagnostics; report their compute separately.

The user wants one teacher, single-view teacher extraction and no layer-count/teacher-view-count sweep. Keep the existing full R2R plus English RxR-CE 15-degree training corpus. No new trajectories, geometry ground-truth labels, RL, action chunks, candidate output conversion, PCDM, temporal teacher, episodic KV, surprise prioritization or combined G1+G3 run.

This is a **fusion-method screen**, not a complete causal claim about every component. Unequal extra parameter counts and historical execution differences remain reported limitations. Numerical equivalence and navigation improvement are different outcomes.

## 2. Why this branch, and what NOT to enable

The `streaming_text_dual` ancestor contains the native text/lane implementation and Window8 varlen attention. The newer `c2_h2_speed` branch inherits those paths and has inspected bounded activation-arena support, optimizer grouping and host-memory safety changes. The latest report documents H2 resource tests and their limits. Neither this document nor those previous tests qualifies new auxiliary losses.

Use the newer branch for source infrastructure, **not the episodic campaign's launcher/configuration**. Set `memory.mode: window8`, omit all `episodic_*` keys, and assert `model.episodic_spec is None`. B=0 inside `episodic_window_v1` is NOT equivalent to selecting the legacy implementation: it takes another forward/attention path. Do not use that mode just to turn off the bank.

- G1/G3/A0/A1: `model.step_lane.enabled: false`.
- R1: lane configuration copied exactly from the existing R0 registry entry.
- Existing auxiliary/episodic Slurm campaigns and source snapshots remain untouched.

Read `../reports/source_audit.md` before changing source. Native policy facts in that report were obtained from live connector reads. Mathematical and configuration decisions below are new proposals.

## 3. Reuse V0/R0 through an explicit baseline registry

### 3.1 Do not confuse three meanings of “baseline”

1. The original pretrained Qwen snapshot is the **initialization** for each independent navigation training run.
2. Existing V0/R0 checkpoints are **comparison references**, not automatically initialization weights.
3. A source code branch is neither a trained checkpoint nor proof that a run finished.

The public reports inspected include native Window8 text E1/E2 and a four-lane Window8 text R0-like E1. New `c2_*` reports also use the label V0 for a separate campaign and describe submitted/paused-at-E1 runs. The user's assertion that V0/R0 are done is respected. These names must be resolved from the local artifacts; do not substitute a same-named pending report or invent a checkpoint ID.

### 3.2 Required registry, created read-only from existing artifacts

Create `reports/vggt_five/baselines.json` with one record per reference containing:

- absolute artifact/checkpoint path and immutable weight hash;
- navigation/config/source hashes and full model/teacher-free inference contract;
- dataset semantic identity and ordered record hash, path-normalized when paths moved;
- pretrained initialization revision or actual warm-start identity;
- total declared schedule, actual completed update, optimizer and LR/warmup trajectory;
- topology, offload backend, sampler seed/order/tail policy;
- lane installed layers, dimensions, initialization seed, clock, gates and prefix mode for R0;
- available E1/E2 exports and evaluation journal identities.

The registry is a launch gate, not a request to train more baselines. Missing metadata blocks only the affected comparison/launch. Preserve uncertainty rather than “repairing” it by a new baseline job.

### 3.3 R1 must match R0's installed architecture

Previous conversational pilot descriptions mentioned one lane at layer 24; the published Window8 text lane report uses **four lanes [16,20,24,28]**. Do NOT silently turn four-lane R0 into single-lane R1. Default to the actual registered R0 architecture, and supervise ONLY `R^24`. If the user has an already completed local single-lane R0, copy that single-lane configuration instead. Layer 24 must exist; otherwise stop R1 for clarification, not introduce an arbitrary new read layer.

This is single-layer **supervision**, not a multi-layer ablation. Native banks in other layers and any other installed lanes remain present and train with the usual navigation objective. Auxiliary gradients may indirectly change them through hidden-stream dependencies.

## 4. Five run definitions

| ID | Native policy | New deployed module | Training-only supervision | Reference |
|---|---|---|---|---|
| G1 | Window8 text, no lane | None | current visual-token feature alignment, decoder output index 23 | existing V0 |
| G3 | Window8 text, no lane | None | delayed native matrix reads from GDN index 24 | existing V0 |
| A0 | Window8 text, no lane | Q/K rank-8 adapters at GDN24, raw heads 0/1 | none | existing V0; exact architecture control for A1 |
| A1 | identical to A0 | identical to A0 | single-view-teacher descriptor relations on actual native Q/K | A0 |
| R1 | exactly registered R0 lane architecture | existing R0 lane design | delayed reads from ONLY lane R24 | existing R0; system comparison with G3 |

**No hidden extras:** no G3-recent, no new R0, no additional seeds, no multiview G2, no combination model in this five-run budget. Recent/delayed lag probes are diagnostic measurements of frozen checkpoints, not additional full policy training.

## 5. Single-view GeoAddress is explicitly a restricted hypothesis

Each teacher call contains one view: `images.shape=[B,1,3,392,518]`. Two independently encoded cached views may be compared in a training relation loss. That is not joint multiview VGGT encoding.

A1 is named **GeoAddress-SV** in outputs. It transfers similarity distributions of VGGT descriptors into native Q/K. Similar descriptors are NOT certified same physical points. No pseudo metric pose, depth alignment, hard match labels or “verified different places” are invented. No anti-interference/protection loss is included in this first A1 run. It tests whether teacher-informed address relations help navigation, not whether exact spatial identities have been learned.

## 6. Initialization and common training policy

Proposed schedule, consistent with the discussed seven-run screen: independent runs from pinned pretrained Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; two epochs, expected 7,704 updates, warmup 232, global batch 8, seed 429. Use the full manifest: expected 30,815 trajectories/instruction episodes and 3,128,624 action targets per unique pass. Recompute exact counts, repeats and steps from the actual sampler.

Before launch, compare this recipe with the user's registered references. A0 and A1 MUST use identical backbone/data/optimizer schedules. If the actual registered V0/R0 use a different schedule, resolve a single written recipe decision before training; do not silently warm-start or rescale epochs. If a reference only has E1, compare new E1 against it. A new E2 versus old E1 is labeled an unequal-exposure system comparison, not a matched auxiliary effect. Do not train V0/R0 again.

Default optimizer: existing AdamW, betas (0.9,0.95), wd 0.01 for eligible matrices, grad clip 1.0, cosine minimum ratio 0.1; native LR 5e-6, new training-only head LR 1e-4, Q/K adapter LR 5e-5, R1 lane LR exactly R0's (reported default 1e-4). Native vision and merger stay frozen/eval. Preserve native tied LM weights, canonical action strings and action-token/EOS loss normalization.

Use full BPTT; attention window length does not set gradient horizon. Training sees teacher-forced canonical history; serving uses its own generated history. Teachers never enter the student input stream.

### 6.1 Auxiliary weights are proposed defaults, not evidence of optimality

- G1: maximum `lambda_align=0.2`.
- G3/R1: same maximum `lambda_recall=0.1`.
- A1: maximum `lambda_relation=0.1`.
- Linear ramp from zero over first 10% of declared total optimizer updates; for 7,704, ramp length 771.

All weights are per globally normalized unit (frames or source/read pairs). Record unweighted losses, active weights and diagnostic shared-trunk gradient ratios. A disposable 20–50-update engineering smoke can expose nonfinites/resource bugs. It is not permission for repeated accuracy-driven hyperparameter sweeps. A prelaunch numerical safety reduction requires a new config hash and consistent G3/R1 update if their shared coefficient changes; report the deviation. No weight schedule changes mid-production to rescue a run.

## 7. Common teacher and sampling

Read `teacher_single_view.md`. Shared teacher: official VGGT-1B code commit `a288dd0f14786c93483e45524328726ab7b1b4ce`, with actual weight revision/SHA resolved once at extraction. Use single-view aggregator index 23, expected feature width 2048 and validated patch_start_idx=5. Pool 28x37 to 15x20, per-token no-affine LayerNorm, shared train-only PCA to **256 dimensions**. No teacher/view/layer/PCA-dimension sweep.

Fixed base target sample: min(16,T) unique stratified frames per physical trajectory. Shared temporal pair plan: up to 8 `(source tau, read t)` pairs, with `8 <= t-tau <= 32`, sampled deterministically across 8–15, 16–23 and 24–32. Read times must be <T. Full actions still train. A1 may need targets for read frames in addition to source frames: extract the union, at most 24 unique frames per trajectory under this plan.

G1 aligns at sampled current visual tokens. G3/R1 query four quadrants and a global target per pair. A1 compares 32 spatially distributed current-query tokens against all 300 source visual keys per pair. Each loss averages its internal rows/regions before contributing one frame/pair unit.

These are different auxiliary tasks and supervision densities; matched backbone exposure is not the same as matched auxiliary FLOPs. Report counts/parameters/throughput; do not claim a perfectly equal-capacity head comparison.

## 8. State readout choice shared by G3 and R1

To preserve the attached plan's usable context-conditioned reader, use the SAME query-input recipe for both arms:

`stop_gradient(normalized native q24 at pre-action anchor, repeated to 32 heads)` + 32-d region embedding + 32-d lag encoding, total 4160.

G3 output query heads: 32. R1 output query heads: R0 lane's actual value-head count (reported default 4). Both reads are from the respective matrices, not final hidden features. This is **context-conditioned state supervision**, not a proof all decoded information resides only in the matrix. Stopping gradient does not remove information from the query. Frozen query-only, state-shuffle and lag probes are required diagnostics. Do not call them extra policy runs or substitute them for future trained ablations.

## 9. Success criteria and interpretation

Primary practical objective: SR/SPL on the same R2R IDs and fixed RxR15 evaluation set/panel, with failure accounting unchanged. Auxiliary cosine/KL is a mechanism metric, not navigation quality. E1 and E2 are evaluated at declared progress; do not pick different best epochs per arm without labeling selection.

- G1–V0: utility of a current spatial-feature objective.
- G3–V0: utility of context-conditioned delayed native-state recall.
- A1–A0: contribution of descriptor-relation supervision in identical deployed adapter architecture.
- R1–R0: utility of directly supervising the installed extra bank, subject to matched recipe/progress.
- R1–G3: whole-system quality/cost comparison; architecture/head capacity differs.

Five runs cannot establish teacher uniqueness, delay-specific causality, generalization across seeds, or a universal spatial specialization. Do not insert more full runs without approval. Decide one promising family from results, retaining negative results.

## 10. Implementation and execution order

1. Read-only baseline/source/environment audit and create registry. No jobs started.
2. Single-view teacher QC/cache and pure auxiliary infrastructure.
3. G1 plus native-off parity; G3 probe math/checkpoint gates.
4. A0/A1 paired installation and relation-target QC; R1 exact-R0 architecture resolution.
5. Full resource qualification for each structurally different arm on available H100 topology.
6. Launch five fresh runs only; checkpoints paused/resumed are the same run, not a new arm.
7. Strict trained-export replay, full evaluation and reports.

Preferred engineering implementation order: **G1 -> G3 -> A0/A1 -> R1**, but production jobs may run independently after their gates. Do not initialize a later scientific arm from an earlier arm's trained weights.

## 11. Reading map

- `common_infrastructure.md`: APIs, functional checkpointed forward, state probes, trainer reduction.
- `teacher_single_view.md`: precise targets, sampling, cache and relation construction.
- `G1_current_alignment.md`, `G3_native_state.md`, `A0_address_control.md`, `A1_geoaddress_sv.md`, `R1_lane_state.md`: five independent run plans.
- `resources_and_evaluation.md`: H100/RAM qualification and comparison rules.
- `../reports/source_audit.md`: what live source supports and what remains proposed.
