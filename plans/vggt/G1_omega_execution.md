# G1 Online Omega execution update

User changes supersede the initial execution plan below: ONLINE teacher, no pre-extraction/PCA, SpatialForcing2560->4096->2048 affine-LN projector, 4H100allocation4925. Authoritative revised spec: G1_current_alignment.md.

- [ ] Online teacher preprocessing/provenance/runtime tests; CPU-local projectorRNG.
- [ ] Functional layer23 checkpointed loss and globalGAS/rank reduction; no native path changes.
- [ ] Actual4GPU smoke with residentteacher/GPUoptimizer; longest627frameRxR and R2R, mixedthreshold,save/resume.
- [ ] Same-exposure runtime alternatives, teacher timings included, dataset-specific timings.
- [ ] Strict serving export/replay, fullregression, source/provenance/report finalization.

## Superseded initial offline plan (historical work record, not executable)

# G1 VGGT-Omega Implementation Plan

> **For agentic workers:** use superpowers:executing-plans, task by task, with tests before implementation.

**Goal:** implement and qualify G1 single-view Omega supervision and measure complete R2R/RxR training throughput on interactive 4924 (2 H100, 768 GiB).
**Architecture:** offline aggregator23 targets, fixed train-only PCA256 and position term; a checkpoint-local projector on decoder23 visual outputs; native Window8 serving.
**Tech Stack:** existing locked student environment, separate external Omega source, no dependency upgrades.
**Spec:** G1_current_alignment.md, teacher_single_view.md, common_infrastructure.md, resources_and_evaluation.md.

## Global constraints

Base 2430fd2. Only G1 implementation and diagnostic runs in this request. No V0/R0 training or modifications to existing jobs. Original native Window8, canonical action-text history, full BPTT, full episodes. Projector 2560->2048->256 with no-affine LN eps1e-6, GELU, Normal(.02), isolated seed429. Frame-mean cosine FP32 eps1e-6; global rank/GAS frame denominator; lambda .2 ramp771 updates. Base Qwen revision851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, nominal global8, two-epoch schedule7704/232. Production launch remains conditional on complete cache and baseline binding; never pass partial-cache smoke evidence as production qualification.

## Resolved design substitutions

User explicitly requests Omega instead of VGGT. Use existing official Omega source ad02778c052e636a623ef425377f342ee41c4fa6 and local vggt_omega_1b_512.pt (hash measured before extraction). Use official balanced512 preprocessing: for640x480,448x592, patch16,28x37 grid,17 specials,2048 channels. Assert actual outputs; slice returned offset. Independent B batches with S=1. Only cache layer23. No geometry heads or teacher resident in student jobs. Existing SpatialForcing wrapper uses images[None] and would mix views: do not reuse that batching for this single-view design.

Resolve physical trajectory via dataset, scene and ordered RGB path identities (root-normalized); actual RGB hashes deduplicate extraction. Repeated instructions sharing the physical image sequence share sampling. Record identity limitations for differently named duplicate trajectories. G1 extracts its sampled frames only; no G3/A1 union is necessary in this scope.

## Review focus

- Missing/corrupt/wrong-provenance cache records must fail, including on resume.
- Checkpoint recomputation must return a scalar aux loss exactly once; no hidden-state hooks or graph lists.
- Unequal rank/GAS support and zero counts must match a single global reference.
- Export strips exactly research_aux parameters and research config; recovery retains both.
- Full 199K-token episode must preserve gradients and fit; no truncation to improve timing.

### Task 1: teacher contract and cache
Files: research/vggt_sampling.py, data/vggt_targets.py, research/vggt_omega.py, scripts/vln/extract_vggt_single_view.py, tests/vln/test_vggt_targets.py.
Interfaces: sample_frames(key,T,seed)->tuple; physical_trajectory_key(ep)->str; TeacherCache.load(paths)->Tensor[F,300,256]; extraction manifest binds weight/source/preprocessing/PCA/hash.
- [ ] Write and run failing sampling, grid/order, cache corruption/provenance, PCA and detached-target tests.
- [ ] Implement deterministic sampling, strict bounded binary shards and Omega aggregator-only extraction; fit FP64 covariance from100000 train tokens over up to2000 trajectories.
- [ ] Run tests and actual teacher batch-isolation, preprocessing, throughput QC. Commit.

### Task 2: G1 functional forward and loss
Files: models/vggt_aux_heads.py, models/research_forward.py, research/vggt_g1.py, models/nav_model.py, data/episode_dataset.py, train/trainer.py.
Interfaces: CurrentSpatialProjector.forward(hidden,target)->frame-loss sum; research_text_forward(...)->(hidden,aux_sum); batches carry sampled visual positions, detached targets, frame count. Counts computed across ranks and GAS before forward.
- [ ] Write and run failing projector gradient/reduction/RNG, native-off forward, checkpoint gradient, zero/unequal-support tests.
- [ ] Add G1-only validation and native functional decoder loop; compute head inside selected layer checkpoint; use one explicit hidden tensor per checkpoint.
- [ ] Add CPU cache loading and parameter grouping at1e-4; log nav/aux separately. Run tests and commit.

### Task 3: recovery/export and real qualification
Files: train/train_episode.py, scripts/vln/export_vggt_g1.py, scripts/vln/profile_vggt_g1.py, configs/vggt/g1.yaml, tests/vln/test_vggt_export.py.
- [ ] Tests for exact namespace stripping, strict reload, target/config resume mismatch; implement minimal changes.
- [ ] Run full regression suite in existing environment and real R2R/RxR complete-episode tests; benchmark alternatives with same exposures and warmed cache.
- [ ] Qualify longest R2R/RxR, threshold boundaries, changing lengths, optimizer states and same-topology save/new-process resume. Export/replay.
- [ ] Report measured per-dataset times, uncertainty, extraction cost, memory, source/cache hashes and all incomplete gates. Fresh whole-branch review and fixes, commit.
