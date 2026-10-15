# R1 online Omega lane-state implementation plan

> Execute inline with executing-plans and one fresh whole-branch review. The user's existing R1 technical plan and explicit request authorize implementation, qualification, speed trials and production launch.

Goal: train the registered four-lane Window8 text policy to retain delayed Omega visual targets in lane24, then run one full joint epoch on worker-2.
Architecture: retain the original lane initialization and serving path; insert read-only rows only into lane24's kernel during training. Obtain detached native query context during one native preparation; real native/lane outputs execute once. Reuse G3 sampling/reduction and full2048 region targets, with online Omega and a student-side MLP.
Tech stack: pinned PyTorch/Transformers/FLA/DeepSpeed; no environment upgrades.
Spec: R1_lane_state.md, updated online requirements, G3_native_state.md and the verified baseline registry.

## Global constraints
- Original base851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a; seed429; four lanes16/20/24/28 with original2key/4value heads128x128, step_end, read_only_zero, half-lives8/24/64/192, beta0.1, zero out_proj, lane seeds429+layer.
- One joint30815-episode pass; global8=4ranks*1episode*GAS2;3852updates,116warmup,cosine minimum0.1; backboneLR5e-6,lane/headLR1e-4; auxiliarymax0.1,ramp771.
- Online frozen FP32 Omega; no PCA/pre-extraction/positional augmentation;5x2048 regional targets; at most8pairs and40probes. Source/read lags8..32 and causal anchors unchanged.
- Full BPTT, native Window8, text action history, no episodic KV, no GeoAddress adapters. All installed lanes train; only lane24 gets direct recall loss.
- Worker-2 only, allocation4925 fourH100/60CPU/768GiB; preserve all prior artifacts. No new R0 training.
- Durable source/arm-bound qualification then immutable-source production; complete checkpoint resume, firstsave10/every100. Record the R0 schedule mismatch explicitly.

## Review focus
- Pre-action anchors must not read the current group's STEP_END write; repeated anchors and empty support remain legal.
- Empty support must retain the same autograd readiness order as supported episodes under four-rank ZeRO2 and accumulation2. Use zero-sized query/index-copy/decoder operations with zero additional sequence rows; a separate `lane(x) + head.empty_loss` shortcut is not safe. Validate actual collective bucket order with empty support in both accumulation slots, not only parameter-gradient presence or DDP normalization.
- No duplicate native/lane/residual computation; native convolution never sees probes.
- With zero lane out_proj, auxiliary gradients must reach earlier writer parameters but no future writer or detached query context.
- Research metadata and strict export preserve all four lane modules, init recipe and serving state; no trained R0 warm start.
- Checkpoint/offload closures bind per-layer arguments; complete4rank resume and source-bound launch gate reject stale evidence.

### Task 1: Registry, online target and lane-read head contracts
Files: research/vggt_r1.py, research/vggt_r1_teacher.py, models/vggt_g3_heads.py, research/vggt_g1.py; tests/vln/test_vggt_r1_contract.py.
Interfaces: validate_r1(config) validates exact registered lane spec/init/LR and canonical policy; r1_teacher_identity/OnlineR1Teacher produce G3-identical unpositioned regional targets using portable used-component identities; NativeStateHead(context_heads=32,heads=4) maps native4096 context into4x128 queries and decodes5x2048.
- [x] Write/run failing contracts: valid R1 accepted; changed layer/clock/init/LR/history/output rejected; H4 head shapes/parameter count/context detach/RNG; no-PCA online identity stable under unused-position drift.
- [x] Implement exact interfaces, preserve G1/G3/A0/A1 defaults and strict target checks.
- [x] Run targeted CPU suite and commit.

### Task 2: Functional lane probes and training wiring
Files: models/vggt_r1_forward.py, models/research_forward.py, models/nav_model.py, data/episode_dataset.py, train/train_episode.py; tests/vln/test_vggt_r1.py, test_vggt_r1_cuda.py.
Interfaces: lane_forward_with_queries(lane,x,roles,head,context,anchors,lags,targets) returns original lane output and recall sum; lane_state_decoder computes native once and lane once; research loop passes original roles to every installed lane. R1 shares G3 pair metadata and global loss denominator.
- [x] Write failing CPU behavior tests for no-probe parity, frozen-state pre-action read, direct writer gradients with W_out0, no future gradient, empty participation; actual pinned GPU test native/lane/probe parity and checkpoint on/off.
- [x] Implement pure prepared-kernel insertion, no synthetic role/token; bind all metadata inside checkpoint closure.
- [x] Run CPU tests, server CUDA tests and whole available suite; report actual environment failures separately. Commit.

### Task 3: Qualification, export and speed trials
Files: scripts/vln/profile_vggt_g1.py, scripts/vln/check_vggt_recovery.py, scripts/vln/check_vggt_export.py, scripts/vln/export_vggt_g1.py, scripts/vln/qualify_vggt_r1.py; tests/vln/test_vggt_r1_launch.py; configs/vggt/r1_omega_h4.yaml, r1_joint_one_epoch.yaml.
Interfaces: allowlistedR1 profile uses bound registry and teacher identity; strict export strips research_aux only and retains all4lanes; qualification requires tests/mixed/empty/longestR2R/RxR/threshold/extreme/recovery/export/timing and source manifest.
- [x] Add failing tests for R1 head/export contracts and incomplete/source-mismatched qualification rejection; extend reusable profilers without changing existing arms.
- [x] Start from G3 runtime: CPUAdam/ZeRO2, pinned arena,4threads,8workers,teacher8,vision8,attention32. Compare limited identical mixed fixtures with attention16/64, then validate chosen setup on representativeR2R/RxR and actual joint manifest.
- [x] Complete three-update extreme/longest memory tests, fresh-process save/resume, trained same-history export/reset/retry/eviction; validate existing R0 serving parity read-only. Record finite auxiliary/writer gradients, GPU/hostOOMs and measured times. No objective/truncation changes for speed.
- [x] Fresh independent whole-branch review; fix important findings with regressions. Commit.

### Task 4: Immutable production launch and report
Files: train/launch_r1_omega_joint_e1.sh, scripts/vln/queue_vggt_r1.py, reports/vggt_five/R1/implementation_report.md and evidence.
- [x] Revalidate bound R0 artifact identity, source/qualification seal, worker2 ownership/idle state and disk/RAM guards; durable detached launch inside4925 only.
- [x] Observe all4ranks real optimizer progress, intended3852/116schedule and positive finite recall loss; validate complete checkpoint10 before claiming successful start.
- [x] Update plan/report with changes, limitations, failed attempts, measured speed/memory, recovery instructions and exact source/job for publication on the new branch. Completion means launched and verified; full epoch remains running.

Progress 2026-10-15: the original production launch saved complete checkpoint10 but stalled at update19 on mixed empty-support backward. The exact four-H100 ZeRO2 replay proved a gradient-bucket size mismatch. Runtime bcb14ea uses the same zero-sized autograd graph for supported and empty episodes, with no added rows or synthetic targets. CPU regressions, exact real-FLA outputs/all-gradient parity, and both source-bound accumulation-order replays passed; every replay produced328 identical bucket records on allfourranks. The revised18-gate qualification passed, including two fresh434-test server suites. Ten entirely positive-support profiles were carried only under exact module-hash/gradient-parity and fixture/exposure/checkpoint provenance checks; eight other gates reran.

The production continuation from checkpoint10 replayed the exact failed episodes and completed update19 in5.06s. The controlled stop was honored at24; checkpoint24 passed the full validator (world4,22files,71,004,558,611bytes), supervisorPASS, noGPU/hostOOM. The stop request was archived, and the full one-epoch queue restarted from24. Post-restart verification passed: allfourranks advanced to26, then32/3852 at2026-10-15T16:40:36Z in Slurm4925.78. No pending stop request or OOM event remains; the full epoch is running. Both resumed schedule gates preserve3852updates/116warmup andglobal8. Initial remaining-time budget14–18hours, with length/checkpoint uncertainty. Runtime source is source-bcb14ea, queue attempt3, actual training output retained in attempt2/r1_joint_e1 via a controlled symlink. The source-bound seal and launcher locks remain mandatory. Report: reports/vggt_five/R1/implementation_report.md.
