# L3 learned-ranker R2R evaluation

## Latest measurements — 2026-10-25T10:00:28.687170+07:00

| Model | Median ms | p95 ms | Mean ms | Actions/s | R2R successes / completed | SR | SPL | Navigation status |
|---|---:|---:|---:|---:|---|---:|---:|---|
| L3 | 153.127 | 168.287 | 155.638 | 6.425 | 593 / 1199 | 49.46% | 46.41% | Interrupted; partial / 1,839 |

Measured 2026-10-25T10:00:28.687170+07:00 on the same RTX PRO6000 Blackwell Max-Q, driver580.178.04 and 300W limit. This is a **separate post-reboot session** using the unchanged original benchmark worker: batch1, the same 80 preloaded RGB frames/instruction, three trials of16 warm-up +64 timed actions (192 samples/model), CUDA-synchronized observe wall time; simulator, transport and disk reads excluded. No detected foreign GPU process. Native BF16 backbone and embedded learned rankers retained (feature preparation FP32, ranker FP64, priorities FP32). Warm-up/initialization excluded; this is not a cold-start benchmark or a proof of cross-session statistical superiority.

Navigation results were independently recomputed from saved journals with contract-order and source/config identity checks. The workstation rebooted at **2026-10-25 08:35:33 +07:00**; both evaluation processes are absent. These SRs are **partial, interrupted results**, not completed evaluations. L2 and L3 cover different-sized ordered prefixes, so their SRs are not a matched full-panel comparison. No navigation evaluation was restarted by this latency measurement.

Latency p99: 169.551 ms; peak allocated GPU memory: 8.720 GiB. Three-trial token sequences identical: True. Raw result: `/data14t/anhdh35/latency_l_20261025/results/l3/result.json`.

The setup and launch sections below are historical evidence, not current process status.


Prepared 2026-10-24T23:32:35.212893+07:00; workstation VMO-ANHDH35-D.
Status: **INTERRUPTED after server reboot; 1199/1,839 completed, 593 successes, SR 49.46%, SPL 46.41%, zero recorded execution errors.** Evaluation is not running.

## Pinned identity

- Branch: `m5_gdn_l_sweep`; latest fetched commit `ba8a1ca9f3dda5356b31e7502b31d95ea054cad5`.
- Recorded training source: `022e526b279b6d7af0f0c5d861429a0d59a7f7e5`. `git diff` confirms identical `src/` and `configs/` between these revisions. The checkout is isolated and clean.
- Hub: `anhdao69/M5_L`, folder `L3`, revision `475ff8e22c556206ffc79c5cfb99d97aeac01249`.
- Weights SHA256: `f4da7f73640cc0b37c262573973d015e913b4462e0590ea505e3fcb4108516a6`.
- Navigation SHA256: `84113d9fb58d0444c980c32448447434527462ba73fac62ac226cc9c1f5ce5b0`.
- Ranker SHA256: `a8bd49daef8d0b9879b45b7c2924014687c384d3a1add57a537f215c7b766ecd`.
- Downloaded all 26 files; remote LFS SHA256/Git blob identities and 25 published SHA256 entries verified. All 724 backbone tensors finite; no teacher, step-lane or auxiliary training tensors.

## Implementation reviewed

Read the branch implementation report, relevant Phase-B plan sections, contract/parser, ranker, feature extractor, offline ranker-plan builder, streaming feature/scoring path and learned-arm tests.
L3 uses F1_v2 features from the three preceding GDN layers per full-attention layer (24 GDN layers total), flattened by layer, native value head and feature channel. Each of eight frozen MLPs is 576→64→1 with SiLU (295,944 parameters overall). Normalizers and weights are embedded in navigation metadata and match the published artifact and run identity.

F1_v2 includes log beta, log negative decay, log value norm, log residual norm, log **decayed-prior prediction norm**, and cosine error. It does not use the legacy committed-norm fifth channel, age, episode length or future information. Prefix, observation and response state carry are preserved. Each frame's 300 visual rows are scored once, the feature buffers are released, and fixed priorities drive token selection.

Backbone serving stays **BF16**. Feature extraction is FP32; the frozen ranker intentionally computes in FP64 on fixed 300-row groups and rounds priorities to FP32. No precision override, ranker refit, model change or silent fallback is applied.

Memory: R=4 including current, B=1200 visual tokens, guard=4784; native rotated K/V retains original logical positions. Matching training configuration, embedded artifact, serialized metadata and YAML file hashes are checked independently. The training `memory_sha256` hashes the arm YAML file, not canonical parsed JSON; the local preflight was corrected to reproduce that definition before launch.

## Validation protocol

- Focused CPU tests: **151 passed** (129 learned-arm/contract/GDN-math/selector tests plus 22 forward-path tests).
- Strict checkpoint reload and exact serializer/memory metadata.
- Real-feature NumPy FP64 reference for each of the eight embedded ranker heads.
- Saved 64-step, 128-step and previous-M5 128-step RGB/action replays; original numerical tolerances retained. Forced replay decoded-action agreement is a protocol check, not free-greedy equivalence; top-token agreement is recorded separately.
- Bitwise greedy reset; idempotent retries; finite priorities; 24 extractor states; feature release; all-layer row bounds; 520-step bounded-stream test.
- Two-episode Habitat smoke, strict journal validation, then all **1,839 R2R val_unseen episodes** in the same ordered list as the S campaign.
- Same data/scenes, RGB geometry, action mapping, text-action history, seed42 and 500-step cap. Evaluator/config code matches the prior S-campaign revision. Existing two-episode S0 consistency evidence is preserved by hash; it is reused, not claimed as a newly executed control.
- Model Python: `/storage/anhdh35/SimpleMemVLN/.venv/bin/python`.
- Habitat Python: `/storage/anhdh35/miniconda3/envs/spatialstack_dagger/bin/python`.
- Dataset: `/storage/anhdh35/SpatialForcing-VLN/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz`.
- Scenes: `/storage/anhdh35/JanusVLN/data/scene_datasets_full`.

## Evidence and monitoring

Root: `/data14t/anhdh35/m5_l3_epoch1/evaluation`.
- Campaign status (`/data14t/anhdh35/m5_l3_epoch1/evaluation/campaign_status.json`)
- Download manifest (`/data14t/anhdh35/m5_l3_epoch1/checkpoint/download_manifest.json`)
- Preflight (`/data14t/anhdh35/m5_l3_epoch1/evaluation/preflight.json`)
- CPU tests (`/data14t/anhdh35/m5_l3_epoch1/evaluation/tests.log`)
- Audit log (`/data14t/anhdh35/m5_l3_epoch1/evaluation/audit.log`)
- Full results (created after qualification/smoke): `results/full/L3_s429/episodes.jsonl`, `summary.json`, `evaluation_contract.json`, `runtime.json`.
- `verify_result.py full` recomputes metrics, checks episode order/coverage, action metadata, hashes and failures. Partial SR must remain labeled provisional.

The published GPU replay was on M5_E1 with the learned ranker, not on this completed L3 policy. This evaluation runs checkpoint-specific GPU checks before launching navigation. Offline capture@96 and training loss are not SR evidence.

## Completed checkpoint-specific GPU qualification

| Fixture | Max logit error | RMS error | Top-token agreement | Mean selected-bank overlap | Original tolerance gate |
|---|---:|---:|---:|---:|---|
| 64 steps | 0.625 | 0.04196 | 99.476% | 98.988% | PASS |
| 128 steps | 0.750 | 0.04261 | 98.956% | 98.633% | PASS |
| Previous M5 128 steps | 1.000 | 0.03478 | 99.478% | 98.536% | PASS |

The unchanged gate uses `torch.allclose(atol=0.75, rtol=0.03)` plus RMS ≤0.15; max error 1.000 can pass the relative-plus-absolute criterion. Bank IDs and some argmax tokens differ: this is tolerance-based qualification, not bit-identical execution or proof of universal correctness.

All eight ranker heads exactly matched the independent NumPy FP64 computation after FP32 priority rounding on the checked 300 real-feature rows per head. Greedy reset logits were bitwise equal. The 520-step bounded audit retained 1,200 episodic rows in every attention layer, peaked at 2,546 resident KV rows against guard4,784, and used 8.719 GiB peak allocated memory.

Two smoke episodes: 1 success / 2, zero execution errors; this is a setup check, not a meaningful performance estimate. The full evaluation runs detached under campaign PID562621, evaluator PID562803. The controller will independently validate complete coverage and recomputed metrics when all episodes finish. `full_validation.json` currently records a **partial** check (`complete:false`), not final validation.

Verified launch evidence (`/data14t/anhdh35/m5_l3_epoch1/evaluation/launch_verification.json`). Existing historical evaluations and checkpoint sources were not modified.
