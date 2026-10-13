# G1 online Omega execution plan

Active revision: 2026-10-13. The user's online-teacher, SpatialForcing-projector and four-H100 instructions supersede the initial offline/PCA execution plan. The authoritative scientific specification is [G1_current_alignment.md](G1_current_alignment.md).

## Constraints

Start from c2_h2_speed commit 2430fd2 in the isolated c2_vggt_omega_g1 branch. Preserve native Window8 text, canonical action history, full episodes and full BPTT. Do not modify existing training jobs or their source snapshots. Use allocation 4925 on worker-2: four H100 80GB, 60 CPUs and 768 GiB RAM. Global batch remains eight episodes: one episode per rank and two accumulation slots.

Frozen online Omega produces independent single-view targets at up to 16 sampled frames per collected image sequence. Keep the full 2048-dimensional teacher target. Follow SpatialForcing's affine LayerNorm(2560), Linear(2560,4096), GELU, Linear(4096,2048). No feature pre-extraction, PCA or trainable teacher-side compressor.

## Implementation and acceptance sequence

1. Teacher and sampling contracts: deterministic frame sampling, exact official image preprocessing, correct special-token removal and spatial order, detached FP32 targets, source/weight/adapter/preprocessing/position hashes. Keep teacher ownership outside student/DDP/checkpoints.
2. Student integration: decoder 23 after the full residual block; checkpoint-local scalar auxiliary loss; global frame normalization over ranks and accumulation slots; separate navigation and auxiliary logs; dedicated head learning rate. Verify zero-weight native gradients and auxiliary-only gradient reach.
3. Four-GPU resource qualification: three optimizer updates on the complete longest RxR episode (627 frames / 199,735 tokens), longest R2R (183 frames / 58,290 tokens), and mixed lengths across the 65,536-token activation-offload threshold. Include lazy Adam allocation. No truncation or TBPTT.
4. Memory/speed tradeoff: compare resident teacher/GPU optimizer, teacher CPU offload between episodes, and CPU optimizer. A no-OOM test with almost no free device memory is insufficient for the recommended profile. Prefer measured headroom, then tune teacher batch, attention batch, vision batch, loader workers and CPU threads without changing samples or loss.
5. Correctness and recovery: full regression suite in the pinned environment; actual four-rank unequal/empty-support reduction; complete recovery checkpoint, fresh-process continuation and comparison to uninterrupted training. Verify head weights change at nonzero auxiliary weight. Strict teacher-free export and native response replay, eviction, retry and reset on both datasets.
6. Dataset timing: representative complete-episode R2R and RxR fixtures, multiple warm updates, end-to-end update intervals including teacher/loader costs. Report sampling uncertainty and checkpoint/startup overhead separately. These are diagnostics, not epoch-completion or navigation-quality claims.
7. Finalize the source commit, runtime recommendation, revised plans and detailed evidence report. The user subsequently authorized a full one-epoch joint production run; launch it after these checks from an immutable source snapshot, with the separate one-epoch overlay (3,852 updates / 116 LR warmup; original auxiliary ramp 771).

## Evidence so far

- Online teacher/head/trainer/export targeted suite: 21 passed; actual processor/merger spatial-order test additionally passed.
- Regression after restoring Git metadata and enabling real processor tests: 350 passed, one CPU AdamW compiler mismatch. The matching isolated compiler was then installed; both CPU AdamW tests passed, including numerical and serialized-state continuation checks.
- Four-GPU mixed fixture: three updates passed, 17.78 s/update after startup, 56.64 GiB maximum reserved GPU memory, 76.44 GiB peak job RAM.
- Four-GPU longest RxR: three updates passed, 90.30 s/update, 76.85 GiB maximum reserved GPU memory, 225.18 GiB peak job RAM. Device usage reached 81,073 MiB; this profile has insufficient practical headroom for recommendation.
- Actual four-rank reduction passed unequal, empty and equal frame support with two accumulation slots (maximum gradient error 2.98e-8). CPU optimizer offload passed the longest RxR test with about 62 GiB active GPU allocation. All 23 four-GPU training diagnostics now passed. The selected resident teacher / CPUAdam profile uses teacher16, attention32, vision8, threads4, workers8; longestRxR peak active61.79GiB. Full-state fresh-process recovery passed (head exact, whole-state relativeL2 9.45e-9), both-dataset export/replay passed, and the final suite passed351 tests without skips. Joint warm timing27.54s/update projects29.46h per epoch, with planning uncertainty and checkpoint overhead. The authorized one-epoch production launcher uses a committed source snapshot and periodic recovery.

## Historical work

An initial offline diagnostic extracted 608 frames and fitted a train-only PCA before the user changed the design. Those artifacts are not used by this implementation. The offline extraction/cache modules were removed. Their costs must not be presented as online training time or current prerequisites.
