# M5 GDN-only S sweep — implementation, qualification and matched training campaign

Report date: **2026-10-20 (UTC)**. Plan: [`plans/M5_GDN_S_L_Technical_Plan.md`](../../../plans/M5_GDN_S_L_Technical_Plan.md) (revision 2).
Branch: `m5_gdn_s_sweep`. Production source (sealed, detached worktree on the cluster): **`dabf4ad085d20512640fe0ba3bda72ad4724dc76`**.
Scientific baseline: `fde1cbef213e27320920400a826addea8d5e114a` (branch snapshot `c2_vggt_omega_g3_m5_layers`).

This report describes work on Phase A of the plan (S arms). **No navigation result is claimed.** Training metrics,
replay logits and selection overlaps below are implementation and qualification evidence, not closed-loop
performance. Evaluation (Habitat, val_seen/val_unseen) runs on the separate evaluation workstations and has not started.

---

## 1. Status at a glance

| Lane | Allocation | Arms in order | State at report time |
|---|---|---|---|
| A | Slurm job **5248**, worker-2, 4×H100, 480 GiB | S1 → S2 → S9 → S4 → S7 | Shared qualification **passed** (21:18Z); **S1 production training since 21:25:59Z**; checkpoint 10 audited |
| B | Slurm job **4925**, worker-2, 4×H100, 768 GiB | S3 → S5 → S6 → S8N → S8W | Waiting (as designed) for the running M5 seed-431 epoch (update 3108/3852 at 21:31Z) to finish successfully; then starts automatically |

- **S0 is M5_E1 and is not retrained.** M5_E1 (seed 429, weights `92089fa1…`) is the matched S0 baseline:
  every S arm reuses its exact recipe and data order (verified below).
- Ten S arms are queued, 5 per lane. Each run takes about 13–14.5 h (M5_E1: 13.57 h; the R2/B1800 run on job 5248:
  14.2 h; S1's first 38 updates, including startup and the update-10 save: 13.67 s/update).
  Rough completion estimates (not guarantees): lane A around **Oct 23, 15:00–24:00 UTC**; lane B, which starts
  around Oct 21 00:15 UTC, around **Oct 24, 18:00 UTC – Oct 25, 02:00 UTC**.
- The same-checkpoint controls `C_RANDOM`/`C_RECENCY` are implemented and qualified for runtime screening.
  They are **not** queued for training: the plan treats them as same-checkpoint controls, not matched-training arms.

## 2. Request, scope and resource decisions

The request was to implement the plan's S arms on a new branch and train them on R2R + RxR15° using both
4×H100 interactive allocations. One allocation was idle (5248). The other (4925) runs the M5 seed-431 epoch.

**Why one run per 4 GPUs (not two 2-GPU runs per allocation).** The decision was measured, not assumed:

| Factor | 4 GPUs/run (chosen) | 2 GPUs/run |
|---|---|---|
| Throughput per GPU | M5_E1: 12.65 s/update on 4 GPUs | earlier 2-H100 full-manifest runs: 22.6–25.2 s/update (≈ same per GPU) |
| Makespan for 9–10 runs | 2 lanes × 5 runs → ~68–72 h | 4 lanes (3+2+2+2 or 3+3+2+2) → ~82 h, first result after ~27 h |
| Fair comparison with M5_E1 | identical: `EpisodeLengthSampler` assigns ranks with `world_size`, so 4 ranks × GAS2 reproduces M5_E1's episode order | world size 2 changes the episode order, so S0 would need retraining |
| Memory | one run peaks at 341–449 GiB host; longest RxR uses 73.0/79.2 GiB GPU | two runs exceed 480 GiB on job 5248; per-rank ZeRO-2 partitions double |

Speed was then maximized without changing the math (Section 8).

## 3. Source, environment and asset identities (Task 0)

The `audit_manifest` gate re-hashed everything on this host. Paths in older reports were not taken as evidence.
All 12 checks passed ([`provenance.json`](evidence/shared/provenance.json)):

- Source `dabf4ad` is clean and descends from `fde1cbe`. The native Qwen3.5 forward fingerprint equals the pinned
  `e56d7184…`. Stack: torch 2.10.0+cu129, Transformers 5.11.0, FLA 0.5.2, FlashAttention 2.8.3, DeepSpeed 0.16.4, CUDA 12.9.
- **M5_E1** = `m5-one-epoch-worker2-20261018/publication-m5-epoch1-20261019/epoch-1`. `pytorch_model.bin` SHA-256
  is `92089fa16096f5fb581a7fb85a670b539a887278d7f66bacd20b4d0424d70ea6`, and all 7 published files match `SHA256SUMS.json`.
  Its navigation config equals the production `resolved_config.json` (plain M5, `committed_mean_l2_v1`, seed 429).
- Base model `Qwen/Qwen3.5-4B@851bf6e8…` (all snapshot files hashed).
- Training manifest `r2r_rxr15_train_20260929.jsonl`, SHA-256 `5deb425d…`: 10,819 R2R + 19,996 English RxR15°
  = 30,815 unique episodes, 3,128,624 frames. A deterministic 4,000-image sample was all present on disk, and every fixture image exists.
  The original dataset, images, instructions, actions and checkpoints were only read; all derived data live under the campaign root.
- The plan's supplied reference package (`START_IMPLEMENTATION.md`, `reference/*.py`, `configs/*.yaml`, evidence) was
  **not present** on the laptop or the cluster. Independent FP64 oracles were written instead (Section 5).

## 4. What was implemented

New, separately versioned memory mode **`episodic_window_gdn_sweep_v1`**. Visibility is identical to M5:
- the prefix;
- R = 4 recent complete groups, including the current one;
- B = 1,200 old *native* visual K/V rows per full-attention layer, at their original logical positions and MRoPE;
- no merging and no re-encoding;
- old text/action rows are not bank candidates;
- the GDN/conv state still sees every token.

Only the fixed birth-time priority of old visual rows changes. Every selector operates on the resident bank plus
the newly eligible source `t − R`, so a deleted row can never return.

### 4.1 Arms (registry in `src/qwen_vl/research/gdn_sweep_contract.py`)

| ID | Priority (FP32) | Unit | Backend |
|---|---|---|---|
| S0 | mean_h β‖e‖ (legacy scorer, unchanged) | token | `qualified_m5_legacy_v1` |
| S1 | mean_h ‖e‖ | token | `fp32_torch_unscaled_residual_block64_v1` |
| S2 | mean_h β‖v‖ | token | `fp32_native_prepare_only_v1` |
| S3 | mean_h ‖v‖ | token | `fp32_native_prepare_only_v1` |
| S4 | min_h cosine error (`HAM-cos-local@M5`) | token | unscaled residual |
| S5 | mean_h cosine error | token | unscaled residual |
| S6 | mean of β‖e‖ over the frozen 8 slow heads/layer | token | `fp32_torch_wy_block64_headwise_v1` |
| S7 | arithmetic mean of a frame's 300 S0 scores (exact `math.fsum`) | whole 300-row frame | legacy scorer |
| S8N | ‖S_after − S_before‖_F / max(‖S_before‖_F, ε_l) | frame | `fp32_torch_wy_block64_boundary_v1` |
| S8W | ‖W‖_F / same denominator; W = exact decay-transported write | frame | boundary backend |
| S9 | mean_h (1 − β)‖e‖ | token | unscaled residual |
| C_RECENCY | latest eligible rows (priority = original token ID) | token | none |
| C_RANDOM | legacy `random_hash_v1` (identical to M5c plans) | token | none |

Numerical conventions are part of the contract and fixed:
- **Keys:** native additive-ε L2 normalization (k/√(‖k‖² + 1e-6)), with the native contiguous key→value-head repeat.
- **Prediction and residual:** v̂ = S̄ᵀk is the decayed prior (pre-write); e = v − v̂.
- **Cosine error:** 1 − clip(v̂·v/(‖v̂‖‖v‖ + 1e-6), −1, 1); it is 0 when ‖v‖ ≤ 1e-6 (flagged) and 1 for a zero prediction with a nonzero value.
- **Nonfinite inputs** raise and are never turned into priorities.
- **Ties:** priority descending, then original token ID (for frames, source index) ascending.

### 4.2 Mathematics (`src/qwen_vl/models/gdn_features.py`)

- **No division by β.** Residual arms solve the unscaled block system
  `E_t + Σ_{j<t} e^{c_t−c_j}(k_t·k_j) β_j E_j = v_t − e^{c_t} S0ᵀk_t` with β in **column j**, so β = 0 is exact.
  The prediction is evaluated directly as `e^{c}KS0 + (A−I)E` rather than as v − E.
- **S6/S8** repeat the qualified legacy WY recurrence expression for expression. With all heads, the result is
  **bitwise equal** to the legacy `_scan_blocks` (tested in FP32 and FP64).
- **S8:** the state before/after each observation block (framing + 300 visual tokens, excluding the action) uses
  per-head alpha products. W is carried as its own accumulator (`W ← αW + Kᵀδ`), so no `S_after − A·S_before`
  cancellation enters S8W. The cross term ⟨(A−1)S_before, W⟩ is logged.
- **Segmentation is identical to M5.** The prefix, each observation and each complete response are separate
  segments, in 64-token blocks from each segment start. Serving appends reproduce it.
- `gdn_raw_v2` raw records (β, g, ‖v‖, ‖e‖, `committed_norm`, `prediction_norm`, v̂·v, cosine, ‖k‖², flags) and
  F0_v1/F1_v2 encoders exist for Phase B. `F1_v1_legacy` can be reconstructed for analysis but is rejected by an F1_v2 consumer.

### 4.3 Selection and serving

- `src/qwen_vl/stream/gdn_selectors.py`:
  - exact running top-B for token arms;
  - `RunningTopFrames` (top B/300 = 4 whole frames, all-or-none rows, earlier source wins ties);
  - recency and random controls;
  - plans reuse the immutable `LayerSelectionPlan`, so the attention/backward path is unchanged.
- `src/qwen_vl/stream/gdn_sweep_state.py`: a separate session state with the same cache discipline as the legacy
  `EpisodicSessionState` (atomic promotion, no history archive, invariant checks).

### 4.4 Minimal edits to existing files (legacy behavior unchanged)

| File | Edit |
|---|---|
| `research/episodic_contract.py` | `GDN_SWEEP_MODE` constant; `parse_episodic_spec` dispatches that one mode to the new parser |
| `contracts.py` | accept the new mode in `validate_config` |
| `models/episodic_forward.py` | for sweep specs only, call `offline_sweep_plan`/`build_sweep_layer_plan`; the legacy statements are untouched |
| `stream/session.py` | choose `GdnSweepSessionState` for sweep specs |

`episodic_cache.py`, `committed_write.py`, `episodic.py` and `window_attention.py` were **not modified**.

### 4.5 Calibration artifacts and fail-closed contract

- S6 needs a frozen slow-head mask; S8 needs frozen per-layer floors.
- Both are **embedded in the arm's memory config with SHA-256**, so every checkpoint's `navigation.json` is self-contained.
- A missing or altered artifact, an unknown key, a mismatched backend/unit/reduction, a non-float epsilon, a
  non-frame budget or an inconsistent KV guard all raise. Nothing falls back to S0 or random.

### 4.6 Tooling (`scripts/memory_sweep/`)

| Script | Role |
|---|---|
| `audit_manifest.py` | Task 0 provenance |
| `calibrate.py` | training-only S6/S8 calibration (4 ranks) |
| `make_arm_configs.py` | generates every arm config from the registry (never hand-edited) |
| `replay_check.py` | real-model offline-vs-`StreamSession` replay per arm |
| `compare_recovery.py` | fresh-process recovery comparison |
| `init_campaign.py` | creates a fresh campaign root |
| `s_campaign.py` | two-lane worker-resident training queue |

## 5. Tests

**Branch full suite: 527 passed, 23 skipped.** This covers all 380 pre-existing tests plus 147 new ones; the 23
skips are the same CUDA/Habitat-gated set as the baseline. The pinned baseline's only failures were 4 helper-gate
tests that call `git rev-parse` inside a non-git archive; they pass in the git worktree. New tests are in `tests/memory_sweep/`:

- **Math, against an independent FP64 token-loop oracle** (`test_gdn_math.py`, tolerance 1e-10):
  - residual, prediction and dot products, and the final state;
  - segment lengths 1/63/64/65 and multiple boundaries;
  - zero and nonzero initial state; β = 0, 1e-30 and 1; g = 0 and −80;
  - the exact self-read identity v − S_iᵀk = (1 − β‖k‖²)e, and S1 = S0′ + S9;
  - the decayed prior differs from the post-write read;
  - S_b = A·S_{a−1} + W, ‖ΔS‖² = ‖D‖² + ‖W‖² + 2⟨D,W⟩, and W = 0 under pure decay;
  - no input mutation, and future perturbations leave earlier scores bitwise unchanged;
  - the headwise scan is bitwise equal to legacy S0;
  - cosine conventions, drift floors, order-independent frame means;
  - `gdn_raw_v2`/F0_v1/F1_v2/legacy-schema rejection;
  - serving appends equal offline scores for every arm on a real native GDN layer.
- **Selection** (`test_gdn_selectors.py`):
  - running plans equal independent batch oracles for every prefix, including tie-heavy, unsaturated and saturated cases;
  - whole frames only; no resurrection; future invariance; fail-closed validation;
  - C_RECENCY keeps exactly the last four expired frames and never the old text that Window8 would keep;
  - C_RANDOM digests equal legacy M5c plans; S0 plans equal legacy M5 plans.
- **Tiny real Qwen3.5 model** (`test_gdn_sweep_forward.py`):
  - offline training equals incremental serving for **every** arm;
  - sweep-S0 forward and serving are **bitwise** the legacy M5 path;
  - passive diagnostics do not change outputs;
  - gradient checkpointing gives identical outputs and gradients; future inputs never change earlier outputs.
- **Contract** (`test_gdn_sweep_contract.py`): legacy M5/M5c/V0 parse exactly as before; every arm validates;
  13 fail-closed mutations; artifact hashing; navigation metadata survives the resume JSON round trip.
- **Queue** (`test_s_campaign.py`):
  - **every arm's resolved config equals M5_E1's except `memory` and the two loader fields**;
  - lane split; allocation readiness; thread rule; optimizer-step estimator;
  - safe pruning, which refuses to run before an epoch checkpoint and export exist;
  - the production command matches the M5 recipe;
  - arm-error isolation and control files; the replay policy.

## 6. Calibration (S6 mask, S8 floors) — training data only

The selection was fixed before any navigation result:
- 128 training episodes (64 R2R + 64 RxR);
- four equal-count length strata per domain;
- distinct scenes first, then SHA-256(tag|uid) order;
- selection SHA-256 `04982363…`.

The policy ran its own qualified M5 memory (R4/B1200 committed) over complete expert episodes. Read-only
pre-hooks recorded the per-episode mean native g per value head and ‖S_before‖_F at all **11,322** observation boundaries.
- τ_h = log 2 / max(−mean g_h, 1e-8), with equal domain weight; the 8 largest-τ heads per layer form the mask (ties to the lower index).
- ε_l = max(1e-6, 1e-3·median‖S_before‖).
- Two roles were calibrated:
  - **`base`**: the original Qwen, i.e. the initialization these arms train from. These artifacts are used for **training**.
  - **`m5_e1`**: used for frozen-checkpoint screening only.

| GDN layer | Base slow heads (training) | M5_E1 slow heads | Overlap | Base ε | M5_E1 ε | Median ‖S_before‖ (base) | Floor hits | Bootstrap Jaccard |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 2 | [2, 3, 10, 14, 19, 23, 25, 29] | same | 8/8 | 0.002815 | 0.002908 | 2.815 | 0 | 1.0 |
| 6 | [9, 11, 20, 22, 23, 24, 27, 30] | same | 8/8 | 0.003170 | 0.003452 | 3.170 | 0 | 1.0 |
| 10 | [1, 11, 13, 16, 18, 22, 23, 24] | [0, 1, 11, 13, 16, 18, 22, 24] | 7/8 | 0.004198 | 0.004399 | 4.198 | 0 | 1.0 |
| 14 | [15, 17, 18, 21, 23, 25, 28, 30] | same | 8/8 | 0.003947 | 0.004511 | 3.947 | 0 | 1.0 |
| 18 | [0, 1, 7, 8, 9, 20, 22, 23] | [0, 1, 8, 9, 11, 20, 22, 23] | 7/8 | 0.003624 | 0.004398 | 3.624 | 0 | 1.0 |
| 22 | [7, 13, 15, 16, 17, 20, 25, 28] | same | 8/8 | 0.005562 | 0.007661 | 5.562 | 0 | 1.0 |
| 26 | [13, 16, 17, 19, 26, 27, 28, 29] | same | 8/8 | 0.014644 | 0.014559 | 14.644 | 0 | 1.0 |
| 30 | [5, 9, 11, 14, 16, 17, 19, 22] | [5, 9, 11, 16, 17, 19, 20, 22] | 7/8 | 0.010752 | 0.013412 | 10.752 | 0 | 1.0 |

- Masks are completely stable under 1,000 within-domain bootstrap resamples.
- No boundary ever fell below its floor.
- The first observation boundary is clearly nonzero (min 2.75): the instruction prefix makes the state nonzero, as the plan anticipated.
- τ is a decay-timescale estimate, not a measured information lifetime.
- Hashes:
  - calibration record: `7aafb0e2…`
  - base mask: `3d72a5d2…`
  - base floors: `40620ea7…`
- The generated configs with embedded artifacts: [`configs/arm_S6.yaml`](evidence/configs/arm_S6.yaml), `arm_S8N.yaml`, `arm_S8W.yaml`.
- Re-running the calibration in the second campaign reproduced the first attempt's masks and floors exactly.

## 7. Real-model serving replay on M5_E1 (fixed expert histories)

Real StreamSession serving was compared with the offline training forward for **all 13 arm configs plus legacy S0**.
- Setup: M5_E1 weights, one 64-step R2R and one 64-step RxR training prefix that cross 60 admissions and saturate
  the bank, forced canonical expert responses, 4 GPUs.
- Gates unchanged: atol 0.75, rtol 0.03, RMS ≤ 0.15.
- This is an **explicit `qualified=false` runtime-swap diagnostic**: M5_E1 was trained with S0 selection. Action NLL
  is teacher-forced, not navigation performance.

| Arm | Data | Gate | RMS | Max abs | Plan mismatches | Mean ID overlap | Peak KV | Action NLL | Mean / min overlap vs S0 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| S0_legacy | R2R | PASS | 0.0534 | 0.727 | 448/512 | 0.9925 | 2578 | 0.1134 | ref |
| S0_legacy | RxR | max-abs miss (inherited) | 0.0437 | 1.978 | 448/512 | 0.9909 | 2643 | 0.3002 | ref |
| S0 (sweep) | R2R | PASS | 0.0534 | 0.727 | 448/512 | 0.9925 | 2578 | 0.1134 | 1.0 / 1.0 |
| S0 (sweep) | RxR | = legacy | 0.0437 | 1.978 | 448/512 | 0.9909 | 2643 | 0.3002 | 1.0 / 1.0 |
| S1 | R2R / RxR | miss / miss | 0.0623 / 0.0385 | 1.148 / 0.768 | 447 / 448 | 0.985 / 0.986 | ≤2643 | 0.1120 / 0.3243 | 0.53 / 0.27 · 0.52 / 0.20 |
| S2 | R2R / RxR | miss / PASS | 0.0661 / 0.0358 | 2.545 / 0.500 | 447 / 448 | 0.993 / 0.991 | ≤2643 | 0.1163 / 0.3020 | 0.86 / 0.65 · 0.82 / 0.63 |
| S3 | R2R / RxR | PASS / PASS | 0.0560 / 0.0428 | 1.000 / 0.797 | 448 / 448 | 0.987 / 0.989 | ≤2643 | 0.1148 / 0.3225 | 0.34 / 0.02 · 0.44 / 0.11 |
| S4 | R2R / RxR | miss / miss | 0.0864 / 0.0629 | 3.266 / 2.422 | 448 / 448 | 0.966 / 0.971 | ≤2643 | 0.1096 / 0.3266 | 0.38 / 0.07 · 0.28 / 0.03 |
| S5 | R2R / RxR | miss / PASS | 0.0555 / 0.0397 | 1.000 / 0.781 | 448 / 447 | 0.984 / 0.986 | ≤2643 | 0.1150 / 0.3232 | 0.36 / 0.05 · 0.26 / 0.02 |
| S6 | R2R / RxR | miss / PASS | 0.0557 / 0.0355 | 0.850 / 0.531 | 448 / 448 | 0.992 / 0.990 | ≤2643 | 0.1195 / 0.3151 | 0.66 / 0.16 · 0.54 / 0.17 |
| S7 | R2R / RxR | PASS / miss | 0.0587 / 0.0447 | 0.781 / 1.812 | 27 / 2 | 0.986 / 0.999 | ≤2643 | 0.1105 / 0.3191 | 0.33 / 0.11 · 0.31 / 0.08 |
| S8N | R2R / RxR | PASS / PASS | 0.0496 / 0.0322 | 0.750 / 0.492 | 5 / 0 | 0.997 / 1.000 | ≤2643 | 0.1052 / 0.3093 | 0.26 / 0.03 · 0.27 / 0.06 |
| S8W | R2R / RxR | PASS / PASS | 0.0502 / 0.0389 | 0.750 / 0.688 | 1 / 67 | 1.000 / 0.965 | ≤2643 | 0.1045 / 0.3163 | 0.28 / 0.06 · 0.28 / 0.06 |
| S9 | R2R / RxR | PASS / miss | 0.0567 / 0.0391 | 0.750 / 1.223 | 448 / 447 | 0.982 / 0.984 | ≤2643 | 0.1059 / 0.3254 | 0.16 / 0.00 · 0.19 / 0.00 |
| C_RECENCY | R2R / RxR | PASS / PASS | 0.0479 / 0.0314 | 0.562 / 0.500 | 0 / 0 | 1.000 / 1.000 | ≤2643 | 0.1218 / 0.3198 | 0.23 / 0.01 · 0.23 / 0.01 |
| C_RANDOM | R2R / RxR | PASS / PASS | 0.0479 / 0.0325 | 0.625 / 0.424 | 0 / 0 | 1.000 / 1.000 | ≤2643 | 0.1149 / 0.3266 | 0.24 / 0.06 · 0.24 / 0.05 |

(The full per-record table, including argmax agreement and score errors, is in [`evidence/shared/replay/`](evidence/shared/replay).)

**What these numbers show:**
- **Harness parity is exact on real weights.** Sweep-mode S0 reproduces legacy S0 in every field and plan
  (overlap 1.0), and the legacy reference is deterministic across all four GPUs.
- Every arm keeps **RMS ≤ 0.087** (limit 0.15), decodes every forced action, and stays far below the 4,784-row KV guard.
- **The arms really are different mechanisms.** Mean selected-set overlap with S0 ranges from 0.16 (S9) to 0.86
  (S2). Per the plan, overlap is reported, never used to eliminate an arm.
- **Max-abs misses come from near-tie selections.** The BF16 offline-vs-serving numerical difference flips a few
  near-cutoff selections. These are 447/512 mismatching layer×step plans, but 98–99% ID overlap. A flipped row
  occasionally moves one logit past atol.
  - The controls with exact selections (C_*, 0 mismatches) stay ≤ 0.63.
  - Frame arms rarely flip (0–67 mismatches).
  - **S4 (min-over-heads cosine) is the most tie-sensitive arm.** It has the largest error (3.27); report it as a
    serving risk before any closed-loop run.

**Inherited S0 miss: investigated and reproduced, not relaxed.**
- The first campaign attempt (source `91eaf5c`) **blocked**, because my original gate required S0 itself to pass
  allclose. The legacy, unchanged M5 path on M5_E1 has a 1.978 max-abs on this RxR fixture (RMS 0.044).
- I re-ran the gate with **M5_E1's own original training source `f4d1c2f`**. It reproduced the result
  **bit-for-bit** (RMS 0.04369400069117546, max 1.9775390625, 448 mismatches), and a 128-step RxR fixture also
  exceeds the bound (max 1.256).
- So on these H100 RxR fixtures the qualified checkpoint itself exceeds the inherited max-abs bound. The earlier
  M5_E1 qualification used different fixtures on a Blackwell workstation.
- **The fix is in commit `dabf4ad`; no tolerance was loosened.** The harness gate now requires:
  - exact sweep-S0 = legacy-S0 parity on every fixture;
  - a deterministic reference across GPUs;
  - identical S0 plans;
  - S0 RMS/decoding/KV bounds.
- Every arm's unchanged allclose outcome is reported. `serving_tolerance_failed = [S1, S2, S4, S5, S6, S7, S9]` and
  `s0_inherited_allclose_failures = [rxrce]` are recorded in `QUALIFIED.json`.
- **These flags are a pre-rollout matter** (Task 8 serving qualification of each *trained* checkpoint before
  closed-loop evaluation). They do not affect offline training correctness.
- Evidence:
  - [`inherited_s0_replay_f4d1c2f.jsonl`](evidence/evidence/inherited_s0_replay_f4d1c2f.jsonl) and its script;
  - the first attempt's shards in [`first_attempt_replay_91eaf5c/`](evidence/evidence/first_attempt_replay_91eaf5c).
- The superseded root `m5-gdn-s-sweep-20261020` holds the blocked record and a `SUPERSEDED.txt`. No training ran there.

## 8. Training recipe, speed and storage

**Recipe (identical to M5_E1; asserted programmatically for every arm before launch):**
- fresh `Qwen/Qwen3.5-4B@851bf6e8`;
- joint R2R + English RxR15°, one epoch: 3,852 updates, 116 warmup, cosine to 10%;
- LR 5e-6, Adam β = 0.9/0.95, weight decay 0.01, clip 1, **seed 429**;
- global batch 8 = 4 ranks × 1 episode × GAS 2;
- full BPTT, canonical action-text history, frozen vision encoder;
- ZeRO-2 with DeepSpeed CPU-Adam offload, pinned activation arena (≥ 65,536 tokens), attention batch 32, vision batch 8, staged BF16 pixels.

The resolved config of every arm equals M5_E1's `resolved_config.json` except:
- `memory`: the arm's own section;
- `training.dataloader_num_workers`: 8 → 4;
- `training.dataloader_prefetch_factor`: 2 → 1.

**Speed decisions (none changes the objective or the data order):**
- **Loader 4 workers / prefetch 1 on both lanes.** This setting is qualified at 480 GiB, and the R2/B1800 epoch on
  job 5248 showed no loader starvation (mean inter-update gap 0.206 s vs 0.201 s for M5_E1 at 8/2). Using it on
  both lanes also keeps every arm's config identical.
- **CPU-Adam threads, measured A/B** on S1's longest-RxR smoke (627 steps, 199,735 tokens; two identical updates):

  | OMP threads | Optimizer step (median of ranks) | Update | Peak GPU reserved | Host working set | Losses |
  |---|---:|---:|---:|---:|---|
  | 4 (M5_E1 setting) | 1.785 s | 68.33 s | 73.02 GiB | 341.2 GiB | 0.43549585, 0.43549585 |
  | **8 (chosen)** | **1.290 s** | **66.49 s** | 73.02 GiB | 344.7 GiB | identical |

  Bitwise-identical losses confirm the math is unchanged. With ~0.5 s saved per update, that is ≈ 30 min per run. Lane B repeats this A/B on its own first arm.
- The new scorers add no measurable GPU memory: the longest-RxR peak of 73.02 GiB equals M5_E1's own qualification.
- **Storage.** Each full recovery checkpoint is 66 GB. After an arm completes and validates, the controller deletes
  *only that run's* redundant rolling checkpoints (e.g., 3700/3800). It keeps `checkpoint-3852` (full
  model/optimizer/RNG state) and the `final` export, so each finished arm uses about 76 GB. Free space was 2.6 TB at
  launch; the supervisors also refuse to start below 220 GiB free. The 151 GB of recovery-test checkpoints under
  `shared/recovery` can be removed later if space is needed.

## 9. Qualification sequence and production launch (sealed source `dabf4ad`)

Lane A gates, all **passed**, recorded in [`QUALIFIED.json`](evidence/shared/QUALIFIED.json):

| UTC | Gate | Result |
|---|---|---|
| 21:03:39 | CPU suite (new + episodic legacy tests, sealed source) | pass |
| 21:04:03 | Provenance (Task 0) | 12/12 checks |
| 21:04:29 | Calibration, both roles; generated S6/S8 configs | masks/floors above |
| 21:08:41 | Serving replay, 13 arms + legacy S0 on 4 GPUs | parity exact; flags reported |
| 21:11:03 | Fresh-process recovery (S1, mixed R2R/RxR fixture) | update-3 losses identical (0.4781087); weights relative L2 2.53e-8, max abs 2.38e-7 (22-file checkpoints 2 and 3) |
| 21:18:05 → 21:25:59 | S1 longest-RxR smokes (OMP 4 and 8) and thread choice | 8 threads |
| **21:25:59** | **S1 production started** (step 5248.21) | — |

**S1 production audit at update 20** ([`S1_production_audit.json`](evidence/evidence/S1_production_audit.json)):
- The complete 4-rank `checkpoint-10` has 22 files. All losses and gradient norms are finite.
- The LR follows the M5_E1 warmup exactly: 0 at update 1 and 9/116 × 5e-6 logged at update 10.
- The resolved config equals the bound identity, and the checkpoint `navigation.json` records `S1` / `fp32_torch_unscaled_residual_block64_v1`.
- **All 160 episode exposures so far match M5_E1's episode UID, action count and token count order exactly, on every rank.**
- Peak reserved GPU 49.5 GiB; 0 CUDA OOMs.
- First-10 losses track M5_E1 closely but not identically, as expected from different selections. For example, update 1 is 0.48298 vs 0.48645 and update 10 is 0.40796 vs 0.40934.

## 10. Operating the queue

- **Campaign root:** `/mnt/data/vmo-ai-task/anhdh35/m5-gdn-s-sweep-dabf4ad-20261020`.
  - Runs: `runs/<ARM>_seed429_e1/` (identity, smokes, `train/`, supervisors).
  - Shared qualification: `shared/`.
- **Sealed source:** `/mnt/data/vmo-ai-task/anhdh35/m5-gdn-s-sweep-source-dabf4ad` (clean detached worktree).
- **Controllers:**
  - Lane A is Slurm step `5248.21` and lane B is `4925.152`.
  - Each writes `lanes/<L>/state.json`, an append-only `history.jsonl`, and `heartbeat.json` every 15 s. Logs are in `lanes/<L>/`.
- **Status:** `cat lanes/A/state.json lanes/B/state.json` and `tail -1 runs/<ARM>_seed429_e1/train/profile_rank0.jsonl`.
- **Controls** (create an empty file in `lanes/<L>/`):
  - `STOP_AFTER_CURRENT` stops before the next arm.
  - `SKIP_<ARM>` skips an arm, e.g. `SKIP_S8W` if only nine runs are wanted.
  - `OVERRIDE_PREDECESSOR` (lane B only) starts even if the M5 seed-431 queue does not end with `complete`; by default lane B waits.
- **Failure handling:**
  - A failed production attempt resumes automatically from its newest validated checkpoint (3 attempts).
  - A failing arm (e.g., smoke over budget) is recorded and the lane moves to the next arm, so GPUs do not sit idle.
  - Identity changes, unbound outputs or foreign compute steps in the allocation stop the controller.
- **Restarting.** Controllers depend on their allocation staying alive (no automatic allocation recovery).
  Re-launch only after confirming no controller step for that lane is alive. The same command resumes from the
  bound identities, completed gates and newest validated checkpoints. Lane A, from the login node:

  ```bash
  B=/mnt/data/vmo-ai-task/anhdh35; ROOT=$B/m5-gdn-s-sweep-dabf4ad-20261020; SRC=$B/m5-gdn-s-sweep-source-dabf4ad
  cd $ROOT && setsid nohup srun --jobid=5248 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=60 --gpus=4 \
    --nodelist=worker-2 --job-name=gdn-s-laneA env G1_CODE_DIR=$SRC G1_CUDA_HOME=$B/cuda-12.9.1 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MAX_JOBS=2 bash $SRC/train/g1_omega_env.sh \
    python -u -m scripts.memory_sweep.s_campaign --campaign $ROOT/campaign.json --lane A \
    >> $ROOT/lanes/laneA-srun.log 2>&1 < /dev/null &
  ```

  For lane B, use `--jobid=4925`, `--job-name=gdn-s-laneB`, `--lane B` and `laneB-srun.log`. The controller's
  `OMP_NUM_THREADS=4` is only the default; each training subprocess receives the lane's measured thread choice from
  `lanes/<L>/threads.json`.

## 11. Deviations, limitations and what is not done

- **Matched training of all S arms**, at the user's request. The plan's economical order screens on the frozen
  M5_E1 first and trains only finalists. Fixed-history replay on two fixtures was done; frozen-trace replay over
  many training episodes (full Task 4) and closed-loop development rollouts were not.
- **One training seed (429).** Matched to M5_E1 is a single-seed comparison, not a robustness result. The plan
  requires paired seeds 429/430 for any final claim.
- **Serving qualification of each trained checkpoint** (Task 8) is required before its closed-loop evaluation.
  Arms flagged in Section 7, especially S4, carry a higher BF16 near-tie risk.
- **Phase B not started:** shadow LOO labels, near/tail horizons, L0–L3 rankers, hidden-state controls and headroom.
  The `gdn_raw_v2`/F1_v2 primitives exist and are tested; collection is not started.
- **Optional items not implemented as arms:** S6-V, the VaN control, S9 exact self-error (the identity is tested),
  untransported-write diagnostics.
- **Time estimates are not guarantees.** Node contention (both lanes share worker-2's CPUs, memory bandwidth and
  filesystem) and episode-length variation can move them.
- **S7 cost:** it uses the exact legacy scorer, so its overhead equals S0's. The S8/S6/residual scans cost about the
  same as the legacy scan; S2/S3 skip the scan.

## 12. Evidence index (copied from the cluster)

- [`campaign.json`](evidence/campaign.json), [`configs/`](evidence/configs) (training and screening arm configs, runtime profile)
- [`shared/provenance.json`](evidence/shared/provenance.json), [`shared/calibration/calibration.json`](evidence/shared/calibration/calibration.json)
- [`shared/gates.json`](evidence/shared/gates.json), [`shared/QUALIFIED.json`](evidence/shared/QUALIFIED.json), [`shared/replay/`](evidence/shared/replay), [`shared/recovery/comparison.json`](evidence/shared/recovery/comparison.json)
- [`lanes/A/threads.json`](evidence/lanes/A/threads.json), [`evidence/S1_production_audit.json`](evidence/evidence/S1_production_audit.json)
- Inherited-S0 investigation: [`evidence/inherited_s0_replay_f4d1c2f.jsonl`](evidence/evidence/inherited_s0_replay_f4d1c2f.jsonl), [`evidence/inherited_s0_replay_f4d1c2f.py`](evidence/evidence/inherited_s0_replay_f4d1c2f.py), [`evidence/first_attempt_replay_91eaf5c/`](evidence/evidence/first_attempt_replay_91eaf5c)

Commits on `m5_gdn_s_sweep`:
- `793b2a4`: scorers, selectors, serving state, tests.
- `91eaf5c`: calibration, provenance, replay, recovery, queue.
- `dabf4ad`: S0 replay harness-parity gate.
- This report is in a later documentation-only commit, which does not change the sealed training source.
