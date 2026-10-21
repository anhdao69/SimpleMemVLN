# G3+M5 diagnostic implementation plan

> For agentic workers: use executing-plans or subagent-driven-development task by task. The user has supplied the scientific design and authorized parallel execution; implement that design without another approval cycle.

**Goal:** complete D-a through D-f, preserving the original R1–R5 decisions and every remaining gate.

**Architecture:** additive diagnostic modules consume immutable serving exports and frozen evaluation sources. Preparation, parity, measurements, statistics and reporting are separate stages with saved evidence; failed gates stop dependent execution. Output and scratch data live outside existing evaluation roots.

**Tech stack:** repository Python policy environment, Habitat 0.2.4 simulator environment, PyTorch/FLA, NumPy, scikit-learn and matplotlib where available.

**Spec:** TASK_SPEC.md plus SCOPE_UPDATE.md in this directory.

## Global constraints

- Code only under scripts/diag or new src/qwen_vl/diag files. No existing scientific-source changes.
- No navigation or original-head training; fitted ridge probes only. D-g removed.
- Models M5_E1, G3_M5, TRIPLE, G3, G1; no A0.
- Seed 20261019 for panels; 240 teacher-forced episodes, 600 closed-loop, 60 determinism subset per model. At least 8 per scene in the shared panel; prefer >=34 actions.
- Rendering <=1/255 per channel. G-1 bitwise logits. G-2 atol .75, rtol .03, RMS <=.15. G-3 relative state error <=.01. D-c relative output error <=.01. No relaxed gates.
- Episode-unit scene-stratified 2000 bootstrap resamples. Paired comparisons. All statistics retain denominators and undefined cases.

## Review focus

- Panel ties, sparse scene strata and short paths: deterministic count and no duplicate selection tests.
- Action/observation off-by-one and STOP handling: reference PNG labels plus renderer gate.
- State orientation, repeated key heads and chunk boundaries: native recurrence comparison at every checked anchor.
- Causal/GQA visibility and B0 replacement plans: explicit index and kernel reconstruction tests.
- Original absolute scene paths affect selector hashes: scene asset identity and baseline UID checks before closed-loop runs.

## 1. Inputs, panel and rendering

- [x] Verify copied artifact hashes, journal coverage and frozen sources against upstream evidence; save REMOTE_ARTIFACT_VERIFICATION.json.
- [x] Run original OnlineG3Teacher constructor with relocated verified files; save full identity evidence.
- [x] Add scripts/diag/panels.py and scripts/diag/test_panels.py. Test exact counts, uniqueness, per-scene minima, reproducibility, long-trajectory preference and held-out scene fold isolation before implementation.
- [x] Save panel_ids.json and de_panel_ids.json before any measurement. Record continuous GT path length, action count, scene/instruction identity, selection method and input hashes.
- [x] Add scripts/diag/render_panel.py using unchanged evaluation sensor/action settings. Extract three reference episodes from verified official archive and compare every observation-before-action image per channel; stop on failure.
- [x] Render selected episodes, retain original raw-frame layout, use prepare_r2r unchanged with an explicitly recorded source subset. Validate every resulting episode and serializer output. Record the subset-source hash alongside the original dataset hash. The canonical three-training-episode rendering-parity gate failed and remains disclosed; user-authorized val_unseen rendering continued without altering its threshold.
- [x] Add scripts/diag/cache_teacher.py; compute exact OnlineG3Teacher targets and save FP16 caches with identity and per-input hashes.

## 2. Model loading and numerical gates

- [x] Implement stored-config load_checkpoint, frozen eval mode and transparent capture hooks in the additive replay/instrumentation tools. G-1/G-2 passed for all five checkpoints.
- [x] Add scripts/diag/state.py and test_state.py. Capture pre-layer inputs, prepare_gdn_inputs, exact prefix/obs/response block64 recurrence, anchor snapshots and source keys/values; test synthetic recurrence and run G-3 against FLA at all checked anchors. G-3 passed at all checked anchors.
- [x] Save raw gate values, not just pass flags. Failed training-render parity is retained as a limitation; no required model replay/state/attention gate failed.

## 3. D-d and D-f

- [x] Add scripts/diag/selection.py for candidate CV, normalized cutoff margin, selected Gini, age/turnover/spatial/concentration, paired Jaccard and token-score Spearman. Tests cover empty/warmup bank and zero spread.
- [x] Add scripts/diag/probe_drift.py and test_probe_drift.py; real sample_pairs and anchors, dummy heads only after two-head bitwise logits/plan invariance. Save per-step/layer overlap and logits errors, then length-stratified summaries.

## 4. D-a, D-b and D-c

- [x] Add scripts/diag/recall.py: epsilon and source-baseline forgetting at fixed anchors/lags/regions; fixed Gaussian head-block projection seed0 to4096; current-view control and lag embedding16. Tests verify source baseline and normalization.
- [x] Add scripts/diag/probes.py: two scene-disjoint outer folds and three inner training-scene folds for ridge selection, train-only scaling; within-episode all-frame top1/MRR. Store folds, lambdas and held-out predictions.
- [x] Add scripts/diag/bank_ablation.py and test_bank_ablation.py: copied episodic forward with valid B0 plans below layer24, recent groups unchanged, upper banks retained; targeted/control removals matched by source-bank share. Recompute recall/decodability and action KL; original-head metrics unavailable without identity evidence.
- [x] Add scripts/diag/attention.py and test_attention.py: post-RoPE QK capture, exact causal visible keys/GQA, FP32 probabilities, anchor plus32 seeded visual queries; pass probability-times-V gate before category/per-token mass measurements. All five 240-episode captures are complete; paired summaries are pending.
- [x] Finish score-versus-eviction-error bridge using D-a source quantities.

## 5. D-e

- [x] Add scripts/diag/selector_eval.py: explicit random_hash_v1 seed429 diagnostic override outside frozen qualified evaluator; preserve R4/B1200, seed42, cap500 and original scene UID paths. Outputs diagnostic=true, qualified=false.
- [x] Verify restricted original baseline IDs/UIDs/instructions for the 600-episode panel; original journals remain immutable.
- [~] The M5_E1 random-selector two-episode smoke passed; four workers are active on its fixed 600-episode panel. See the timestamped live counts in `STATUS.md`. The 60-episode original-selector repeat and both G3_M5 conditions remain queued. The 600-episode score condition is the existing immutable full-run journal, restricted to the same panel IDs.
- [ ] Compute SR/SPL/OS/NE/forced STOP, paired deltas, exact McNemar and scene-cluster interaction CI.

## 6. Analysis and delivery

- [x] Add scripts/diag/statistics.py and report.py plus README run order. Test paired bootstrap on known synthetic contrasts and undefined metrics.
- [ ] Write raw per-pair/per-episode data, six diagnostic figure groups, complete results.json and seven-section report. Fill exact R1–R5 table; do not treat absent measurements as refutations.
- [ ] Verify every numeric report claim traces to machine-readable evidence, no frozen source changed and no model parameter updates occurred. Review all code and gate evidence.
- [ ] Commit and push only diag/g3m5-failure-20261019; report any remaining push authentication failure.

## Dataset-authorized continuation

The rendering gate failed on all three fixed training episodes. The user subsequently instructed us to use the confirmed datasets, authorizing continuation with this disclosed limitation; see SCOPE_UPDATE.md. Its threshold and failed evidence remain unchanged. G-1/G-2/G-3 passed for all five checkpoints. All five full 240-episode captures are complete. The first post-measurement run stopped before D-b interventions on a custom full-vs-truncated state comparison; that evidence is preserved in POST_MEASUREMENT_DEBUG.json. The corrected same-prefix D-b run is active under `/data14t/anhdh35/diag_g3m5_20261019_assets/post_measurements_retry1`.

## Implementation versus execution status

Checked implementation items indicate code exists; execution completion is tracked separately. Runtime and gate infrastructure lives in replay_gate.py, instrument_gate.py, gate_queue.py and measurement_queue.py rather than separate runtime.py/gates.py files. All five G-2 replays pass. G-1/G-3/D-c and full-panel D-f dummy-head invariance passed; the five 240-episode captures are complete. Probe fitting, D-b bank interventions, and offline capture/probe/ablation summaries are complete. D-e is live on the first M5_E1 random-selector arm; the score baseline is taken from the original full-run journals and the 60-episode original-selector reruns are the determinism check. The final selector aggregation, complete report/figures, and final requirement-by-requirement review remain pending. The latest full helper suite passed 143 tests with 4 skips; focused D-a export/report tests also pass.
