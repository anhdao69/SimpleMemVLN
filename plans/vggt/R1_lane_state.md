> **Authorized online implementation update (2026-10-15):** use frozen resident FP32 single-view VGGT-Omega-1B-512 online, no feature pre-extraction/PCA, the same unpositioned LN2048 regional targets as implemented G3. The lane-read decoder outputs2048, not256. User selected one joint R2R+EnglishRxR15 epoch from original Qwen base:3852updates,warmup116,cosine minimum0.1; global8 on worker-2 fourH100. R0 is the verified completed text-dual run in the registry (four lanes16/20/24/28). R0's E1 was halfway through a two-epoch schedule: this LR mismatch is explicit, not a controlled auxiliary-only comparison. `R1_omega_execution.md` records the implementation and qualification gates.

> **Distributed empty-support requirement:** production uses ZeRO2 with two accumulation slots. Empty support must follow the same zero-sized query/probe/decoder graph as positive support, inserting no extra sequence rows. A standalone zero-loss shortcut changes gradient readiness order and can deadlock the reducer despite every parameter receiving a gradient. Require matching four-rank bucket sequences and recovery past the actual failed batch; the implementation report preserves the initial update19 failure and repair evidence.

# R1 — Delayed VGGT supervision of the existing second memory bank

> **For agentic workers:** reuse the user's completed R0 as a comparison artifact and its ARCHITECTURE/INITIALIZATION SPEC as the recipe. Never submit a new R0, and never silently train a new one-lane architecture against a four-lane baseline.

**Goal:** test whether directly teaching the added lane to retain old visual evidence improves its navigation contribution.
**Architecture:** exactly the registered R0 lane layout; auxiliary reads only from lane24. Native banks remain present. No new lane variant is introduced.
**Tech stack:** existing ParallelStepLane and native policy plus the common training-only read-only probe infrastructure.
**Spec:** master, common, teacher and baseline registry.

## Global constraints

Copy installed lane layers/heads/widths/clock/prefix/gate initialization from R0. Reported default is installed [16,20,24,28], each4 value heads x128x128. Supervise only R24. No multi-layer supervision sweep. Window8 canonical text, full BPTT, single-view teacher, same source/read pairs/target representations as G3. R1 starts from R0's original initialization recipe, not finished R0 weights plus extra epochs.

## Review focus

- The matrix available before action t is R after group t-1, not the write at the end of group t.
- Native and lane outputs must each execute exactly once; wrapper reuse must not add lane residual twice.
- A read-only row must not receive STEP_END role or be passed through lane gates/norm/output as a real token.
- Direct R loss bypasses zero W_out; check writer gradients rather than assuming navigation-only gradient timing.
- Export retains every installed lane but removes the query/decoder.

## 1. Baseline binding and installed layers

Resolve `baselines.R0`. Assert it is the intended text Window8 control, not FullContext candidate/no-history or old external Memory64/R0 terminology. Require layer24 installed and key/value dimensions compatible with the reader.

If baseline has four lanes, R1 retains all four. If it has exactly one lane24, R1 retains one. This choice is **not an ablation and not selected from results**; it is copied to preserve the user-provided baseline. The only new supervision site is layer24 in either case.

If no valid reference metadata is available, block R1 launch while G1/G3/A0/A1 continue. Do not create another baseline job. The public report's four-lane E1 is a fallback reference description, not proof that it is the local artifact the user intended.

## 2. Memory lifecycle

At episode reset, every lane state is zero FP32. Prefix is read-only. All original observation/action tokens may read the lane. Only the last existing separator token of a complete observation/action group writes/decays it.

At pre-action anchor for step t:

`R_read(t) = R_after_complete_group(t-1)`.

The lane has NOT written observation t yet. For delayed target source tau<=t-8 this is fine: source tau has long been committed. This run is historical retention, not predicting the unobserved outcome of current action. No gt current action enters the query or earlier writer.

## 3. Query and decoder matched in recipe to G3

Use the same detached normalized native q24 context (4096), lag32 and region32 input as G3. Query output has `H_lane*128` dimensions, where H_lane is copied from R0 (default4).

Default H4:

`LN(4160,no affine) -> Linear4160,512 -> SiLU -> Linear512,512 -> reshape[4,128]`.

At each requested anchor add five no-write lane-kernel rows:

`q=query; k=fixed unit; v=0; g=0; beta=0`.

Probe read equals `R_read(t).T @ q_hat / sqrt(128)`. Keep probe reads before lane gated RMSNorm/z/out_proj; remove them from the real-token sequence. Native GDN24 receives no probe rows. Capture the detached native q context from the existing native preparation once, not a second model pass.

Decoder: per-head FP32 RMS normalization no affine -> flatten H_lane*128 -> Linear(H_lane*128,1024) -> GELU -> Linear1024,2048 (online Omega; no PCA).

Targets, five regions, per-pair cosine loss, lambda max0.1 and ramp are identical in recipe to G3. Head parameter counts differ because one bank has4 vs32 value heads; report this rather than pretending exact matched capacity.

## 4. Why direct supervision may change lane learning

For navigation-only lanes at initialization, W_out=0 prevents immediate navigation gradients through the branch into preceding writer parameters. Output projection can open first. This auxiliary reads R directly, so when lambda>0 it can train relevant earlier lane writes even while W_out remains zero. This is a testable gradient property, not a prediction of improved SR.

The probe's native-context query is detached, but historical context can still enter through it. R1 has the same shortcut qualification as G3. The auxiliary can also shape earlier native layers and earlier installed lanes through writer input dependencies. Do not claim the gradient changes only R24 parameters.

Native language weights and all installed lane weights remain trainable under the registered recipe. Do not freeze the backbone for R1 alone; that would add another factor.

## 5. Implementation tasks

### R1.1: registry and lane reader
**Files:** `research/vggt_r1.py`, `research/vggt_r1_teacher.py`, `models/vggt_g3_heads.py`; tests `test_vggt_r1_contract.py`.
- [x] Test exact-R0 layer preservation, layer24 requirement, wrong output/history rejection.
- [x] Query/decoder H4 shapes and parameter counts; generic H_lane supports validated metadata only, no automatic capacity conversion.

### R1.2: pure lane probe path
**Files:** `models/vggt_r1_forward.py`, `models/research_forward.py`; original `parallel_step_lane.py` serving implementation is preserved.
**Interface:** `lane_forward_with_queries(lane, x, roles, head, context, anchors, lags, targets) -> (real_lane_output, recall_sum)` for uncached training only.
- [x] Mirror the narrow pinned preparation/read/postprocessing in the R1 functional helper, preserving native serving `forward` unchanged; actual FLA parity tested.
- [x] Insert rows AFTER lane's QKV+SiLU/head repetition/effective gates, never by assigning synthetic token roles to x.
- [x] Probes bypass norm/out_proj; real rows still use original z and original output projection.
- [x] Test no-probe outputs/gradients match old lane, no double write, no current-source write before action.

### R1.3: gradient gate
- [x] With W_out=0 and lambda>0, a later recall loss must give nonzero earlier lane writer gradients in a nondegenerate fixture.
- [x] Native navigation output remains unaffected by probe presence mathematically when lambda=0, apart from declared finite-precision effects.
- [x] Isolated anchor loss has no future writer gradient; first-step/no-pair episodes use valid zero participation.

### R1.4: serving/checkpoint/resource
- [x] Train checkpoint keeps auxiliary parameters; export strips only those and retains every R0 lane.
- [x] Load existing R0 for diagnostic parity without changing its metadata/weights; not another training run.
- [x] Train a disposable smoke, save/reload, then run same-history raw-token and complete-response comparisons.
- [x] Longest real episode gate with all installed lanes and40 probe rows at R24, full optimizer/offload/save; launch fresh R1 only after PASS.

## 6. Evaluation

Compare R1–R0 at equal declared/completed exposure, output/history and lane layout. If only R0 E1 exists, matched inference at new R1 E1 is the available comparison. Do not silently compare R1 E2 against R0 E1 as pure supervision gain. Additional E2 export evaluation is allowed only when those existing weights are available; no request to train R0 further.

Compare R1 to G3 as a whole-model quality/cost choice. If native G3 is equal or better, keep the simpler native model; do not force the lane into the paper. A decrease when suppressing trained lane output is a dependence diagnostic, not a replacement for trained no-lane control.

Report installed lane parameters/state bytes and actual latency; training-only auxiliary adds no extra deployed modules relative to R0, but R0 itself is more expensive than native V0.

Serving qualification compares optimized offline and streaming independently against the established sequential offline BF16 reference at unchanged tolerances, plus exact source/export scores. The original one-logit direct R2R discrepancy is retained in the report; approximate equality is not transitive. No serving or training math was changed to address this audit difference.
