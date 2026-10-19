# G1 paired-future Omega implementation plan

> Agentic execution: use `superpowers:executing-plans` task by task, then a fresh whole-branch review before production launch.

**Goal:** Train original G1 against current-frame Omega features conditioned on the next observation, without future student inputs.

**Architecture:** Frozen Omega receives independent pairs `[I_t, I_(t+1)]` in `[B,2,3,448,592]`. Select slot zero at aggregator layer 23 after inter-frame attention, then apply unchanged G1 pooling, no-affine LN and positional target. Preserve original G1 student, projector and inference export.

**Tech stack:** Existing pinned Qwen3.5-4B/PyTorch/Transformers/FLA/Omega, four H100s, ZeRO2 CPUAdam and pinned activation offload.

**Spec:** User's paired-future teacher request, followed by explicit instruction to revert the G3 attempt and use G1. This document records the G1 interpretation; the original G3 worktrees were restored and no G3-future training was launched.

## Constraints

- Base `origin/c2_vggt_omega_g1@6833bee`; isolated `c2_vggt_omega_g1_future`.
- Native Window8 text, canonical action history, frozen visual encoder, full BPTT, decoder-layer-23 spatial projector 2560→4096→2048, existing cosine objective.
- Teacher weights/source/preprocessing unchanged; FP32 eval, inference mode, no teacher parameters in student/optimizer/checkpoint. No PCA.
- Preserve original deterministic sampled frames, dropping only sampled frames whose future observation does not exist. No clamping, wraparound, replacement frame or resampling. Navigation retains every frame/action.
- Teacher context is explicit opt-in; old G1 configs/checkpoints retain single-view behavior and identity. New identity includes context, offset, frame order, current-slot extraction and adapter hash.
- One joint R2R + English RxR15deg epoch from original base: 30,815 episodes, seed429, batch8, 3852 steps, warmup116, cosine floor0.1, backboneLR5e-6, headLR1e-4, lambda_max0.2, ramp771.
- Worker-2 only; allocation4925 currently idle except interactive shell. Separate output directories and frozen production source. No shared environment updates.

## Review focus

- Flattening pair dimension into batch would silently disable joint attention.
- Taking output slot1 or averaging both slots would misalign targets.
- Terminal frames and empty support must preserve global frame denominators and differentiable-zero auxiliary participation.
- Future pixels/teacher graphs must not enter student inputs, query paths, optimizer or inference metadata.
- Changed teacher semantics must invalidate checkpoint identity; restored state must preserve optimizer/scheduler/RNG and data sampling.

## Task 1 — Teacher boundary and temporal layout

- [x] Write and run failing tests in `tests/vln/test_vggt_future.py`: source/current selection, future sensitivity, batch independence, detached ordinary tensors, malformed input, terminal exclusion, unchanged serialized student inputs, invalid config.
- [x] Implement `future_frame_pairs(frames,length,offset)` and `OnlineFutureOmegaTeacher` in `src/qwen_vl/research/vggt_future.py`.
- [x] Add explicit context validation in `research/vggt_g1.py`; pair CPU pixels in `data/episode_dataset.py`; select the teacher in `train/train_episode.py`. Leave student/model/head code unchanged.
- [x] Run targeted CPU tests and commit.

## Task 2 — Actual teacher, causal gradients and configuration

- [x] Add H100 tests: direct Omega oracle, current-slot selection, future perturbation, batch isolation and microbatch numerical parity, frozen teacher, identity mismatch rejection.
- [x] Verify current-prefix outputs/gradients with future student pixels changed, checkpoint on/off, auxiliary gradients to projector/backbone, zero future/teacher gradients.
- [x] Extend diagnostic profiler with opt-in future offset and one-epoch policy overlay. Validate empty support rather than requiring every episode to have positive support.
- [x] Create `configs/vggt/g1_future_h4.yaml` with actual pinned identity and a separate production launcher. Compare resolved config against G1; document only intended changes.

## Task 3 — Four-H100 qualification and speed

- [x] Full repository regression on worker-2; distinguish CPU skips from actual GPU execution.
- [x] Complete mixed R2R/RxR updates; benchmark pair microbatches without changing student batch/GAS. Test representative and longest episodes in both datasets, repeated long updates and empty support.
- [x] Save reference checkpoint2, fresh-process resume to3; compare complete state/head/loss and validate all-rank inventories.
- [x] Teacher-free export/strict reload and streaming replay for both datasets; verify exact source/export outcomes, preserving established BF16 tolerances.
- [x] Independent whole-branch review; resolve material findings and rerun affected gates.

## Task 4 — Launch and report

- [x] Pin qualified source/config/data/teacher hashes; publish the isolated branch with a normal push.
- [x] Launch the verified G3+A1 continuation first, with G1-future queued automatically after its complete one-epoch checkpoint/export. Preserve owned supervisors, RAM/disk guards, periodic recovery and auto-resume.
- [x] Observe G3+A1 finite navigation/state/relation losses, preserved3852/116 schedule, four-rank continuation from2100 and complete recovery checkpoint2110. G1-future checkpoint10 monitoring belongs to its later automatic stage, not an already-completed claim.
- [x] Write detailed `reports/vggt_five/G1_future/implementation_report.md`, qualification evidence and measured ETA. Report running training honestly; do not present smoke validation as full navigation evaluation or guaranteed completion.

## Execution ledger

- 2026-10-19 UTC: G3 future experiment reverted locally/remotely, no training launched. G1 isolated branch starts at 6833bee.
- Task 1 RED: 10 new failures demonstrate missing paired teacher/layout/config behavior; 4 original G1 tests pass and CUDA RNG test skips on login CPU.
- Task 1 GREEN: 14 pass, 1 CUDA-only skip on login CPU after implementation.
- First worker-2 full regression: 362 passed, 115 dependency warnings, 49.87 s, including actual pinned Omega joint-attention test and CUDA RNG check. No skipped tests in this GPU suite.
- Ruling: terminal sampled frames are excluded from auxiliary support only. This is required for a genuinely future observation; wrapping/clamping would give a false future target. Navigation training keeps the terminal action. Cost: slightly fewer auxiliary frames for trajectories sampling the terminal observation.
- Ruling: retain original G1 lambda_max=0.2 and positional target, rather than carry G3 lambda=0.1/native-state head into the G1 pivot. Cost: this experiment compares against G1, not the abandoned G3 variant.
- Ruling: diagnostics use the one-epoch policy overlay, while bounded max-update smoke runs naturally have their own short diagnostic scheduler horizon. Only the full launch is claimed as a 3852-step scientific run.
- Qualification owns step4925.131 on worker-2; full training is gated on successful longest/recovery/export checks. No other training jobs were active before qualification.

- User follow-up: download `anhdao69/g3a1_training`, restore its complete G3+A1 checkpoint first, and queue G1-future after successful completion. User confirmed that HF repository after the G1A1/G3A1 naming clarification. G1-future must remain unstarted until the predecessor completes.
- GitHub fetched again: G1 base remains 6833bee. G3+A1 is not published on GitHub; its supplied HF bundle contains exact production source877815e and latest report commit260e6d8. Resume uses the exact archived production source rather than mixing unrelated GitHub code.
- Fresh whole-branch reviewer: no actionable correctness findings in6833bee..2ddcbce; remaining hardware gates still required.
- Mixed 4GPU diagnostics: B8 pairs20.49s/update, B4 pairs20.63s/update, both49.92GiB reserved and zeroOOM. Retain B8; observed0.7% difference is not a statistically established speed gain.
- Ruling: first terminal fixture had only one record and naturally ended each loader pass before GAS2. Its two updates were finite with noOOM, but exposure-count gate correctly failed. Retain failed evidence; rerun with eight terminal-only records. This changes only the disposable fixture, not training code or production data. Corrected terminal8 gate passes, zero auxiliary frames,16 exposures,28.24GiB reserved.
- Added restartable worker-resident G3+A1→G1-future queue. Tests first failed4/4 for missing implementation; then4/4 passed. It binds source/configs, uses launcher locks, requires complete checkpoint3852 plus final forward reference, and blocks after failed/early-stop children.

- Queue review found a material controller-only crash/restart risk: surviving per-run launcher could reject a duplicate start and strand the handoff. Fixed by waiting for the existing launcher's lock and rechecking completion before starting/resuming. Real subprocess regression: initially1 failed/4 passed; after fix5 passed. Both launchers hold the same per-run `launcher.lock` for their entire supervisor lifetime.
- G3+A1 resume wrapper keeps exact source877815e and original command/config/schedule, adds worker-2 restriction and one early recovery save at2110 (then ordinary100-step saves). This is an operational checkpoint-cadence change only.
- Full HF snapshot47043b9693d30f5f4b423e9fddf780f92a6f189e downloaded:66.46GiB. All source-manifest hashes pass. All-file checksum and checkpoint-internal validation in progress; immutable original download retained, prepared files use hardlinks plus atomic metadata replacement.

- Final qualification:371 G1 GPU tests +419 exact-source G3+A1 GPU tests +4 export-audit tests pass with no skips. Mixed-extreme3updates passes, noOOM. Source hashes for every diagnostic `src/` entry match immutable1fcc2af. Recovery and source/export/replay/reset pass. R2R smoke greedy output remains invalid after3updates; explicit diagnostic-only recording preserves strict default and unchanged production serving.
- Full migration audit passed all30815 original sampling keys, all3128624 image paths, explicit128pair lists/1024query sets, and all23 checkpoint-state hashes. Prepared metadata passes original trainer preflight.
- Queue launched2026-10-19 01:41UTC in4925.138 onworker-2. G3+A1 continuation first from2100, G1-future fresh base second. Queue completion gate requires full3852-step checkpoint/export, not process exit alone.

- Full continuation verified through2115/3852 at2026-10-19 01:46:56UTC:120 new unique exposures, no overlap with pre2100 samples, finite navigation/state/relation losses, zeroCUDAOOM. Newly saved checkpoint2110 independently validates all22 files, all four Adam shards/steps/LRs, scheduler, model heads/adapters and RNG. G1-future remains queued; no full-run completion/evaluation is claimed.
