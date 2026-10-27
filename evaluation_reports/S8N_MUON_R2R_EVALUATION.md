# S8N_MUON — R2R evaluation

Snapshot 2026-10-27T07:08:14.418301+07:00. Campaign phase: **complete**. Numerical qualification: **False** (null means pending).

## Checkpoint and training

- Hub: `anhdao69/M5_S/S8N_MUON`, pinned revision `e73fec141af206d93a2870f2b6bbf3c4a18be866`.
- Local checkpoint: `/data14t/anhdh35/m5_s8n_muon/checkpoint/S8N_MUON`.
- Weight SHA256: `ea908a0588395463faca47114cfc192087124381db478b68d928eff35c051d27`.
- Branch `m5_s8n_followup`, evaluation commit `11baeb1d7707aebbf3af030461392dfd942c3191`, training commit `a8bb0a2bfdd854c80221ba895100846801d359a3`; isolated clean checkout.

The published README, run identity, policy YAML, training/publication validation and follow-up controller were reviewed. This model was trained for one epoch from original Qwen3.5-4B, seed429, 3,852 updates on joint R2R/English RxR15deg at peak LR5e-6. Muon replaces AdamW on language-model hidden-layer 2D matrices; embeddings, LM head, norms, gates, conv filters and scalars retain AdamW. Muon uses momentum0.95, Nesterov, five Newton-Schulz steps and match_rms_adamw scaling. Published action-weighted loss0.10316 is training evidence, not navigation SR.

The exported navigation metadata equals original S8N except training.matrix_optimizer, which exactly matches the published policy YAML. Latest src/configs/scripts/vln are identical to recorded training source. S8N net-state-drift scoring, R4/B1200, frame300 selection, guard4784, calibration, action history and generation protocol are unchanged. BF16 inference uses the original scorer and attention without constructing an optimizer. No serving override or calibration refit.

## Integrity and validation

- All22 remote file identities and21 published SHA256 entries verified.
- All724 serving tensors finite and expected backbone keys only.
- Training config, exported metadata, arm/policy hashes and source revision verified.
- Fresh focused tests passed: 148 tests covering memory contract, selector, forward, math and Muon; tests.log. Earlier 138-test evidence is also referenced in download verification; it is not a new execution.
- Fresh checkpoint-specific GPU qualification: strict reload, independent FP64 S8N helper oracle, saved64/128/previous-M5-128 replay fixtures, frame/cache invariants, bitwise greedy reset and520-step bounded stream.
- Native numerical thresholds remain allclose(atol0.75, rtol0.03) and RMS≤0.15. Forced decoded-action agreement is a protocol check, not free-greedy equivalence. Any numerical failure is disclosed as unqualified diagnostic; core structural/nonfinite/RMS/protocol failures block the run.

## GPU replay evidence

All three native BF16 replays failed the unchanged numerical gate. Selected frame banks differ, with minimum overlap0.75. The independent helper check passes, but no MUON-specific FP32 or forced-selection experiment has established the root cause. Do not claim pure rounding or proven code equivalence. Any navigation SR from this run is an **unqualified diagnostic**.


| Fixture | Max error | RMS | Top-token agreement | Mean bank overlap | Gate |
|---|---:|---:|---:|---:|---|
| 64 | 1.8564 | 0.07482 | 98.953% | 95.208% | FAIL |
| 128 | 1.8564 | 0.06550 | 99.217% | 95.867% | FAIL |
| Previous M5 128 | 1.3145 | 0.05685 | 99.478% | 97.455% | FAIL |

Helper passed: True; bitwise reset: True; bounded-stream passed: True, peak KV2546 / guard4784, peak allocated8.681 GiB.

## R2R protocol and monitoring

Two Habitat smoke episodes must pass strict journal validation before all **1,839 R2R val_unseen episodes**. Same fixed order, seed42, max500 steps, RGB480×640, scenes/dataset and action mapping as previous S8N evaluations. Full validation recomputes SR/SPL, checks exact identities, episode order/coverage and execution errors. Partial validation is not final validation.

Original S8N reference:964/1,839 successes, SR52.419793%, SPL48.679311%. Incomplete SR should not be interpreted as a matched full-panel comparison. This report concerns only S8N_MUON; other evaluations are tracked separately. Shared GPU runtime timings are not controlled latency benchmarks.

Model Python `/storage/anhdh35/SimpleMemVLN/.venv/bin/python`; Habitat Python `/storage/anhdh35/miniconda3/envs/spatialstack_dagger/bin/python`.
Evidence root `/data14t/anhdh35/m5_s8n_muon/evaluation`. Verification (`/data14t/anhdh35/m5_s8n_muon/evaluation/download_verification.json`), campaign status (`/data14t/anhdh35/m5_s8n_muon/evaluation/campaign_status.json`), audit (`/data14t/anhdh35/m5_s8n_muon/evaluation/audit/S8N_MUON/`).
Full journals: `results/full/S8N_MUON_s429/episodes.jsonl`, `summary.json`, `runtime.json`, `evaluation_contract.json`.

smoke_validation.json: 2/2 episodes, failures0, passed=True, complete=True. This is a validation snapshot, not an automatically refreshed live progress count.

full_validation.json: 1839/1839 episodes, failures0, passed=True, complete=True. This is a validation snapshot, not an automatically refreshed live progress count.

## Migration and result collection

Execution moved to `anhdh35@10.148.182.137`, root `/media/vmo-perception/disk_2/anhdh35/SimpleMemVLN/evaluation/s8n_muon_migration_20261027`. Local model/evaluator processes are stopped. The sealed handoff preserved 1,301 completed episodes, 542 successes; the remaining episodes resumed and finished on the destination. Weight, navigation, source, dataset, base metadata and all 22 scene/navmesh hashes matched. Both reference smoke episodes matched all 169 RGB frames, action tokens, trajectories, metrics and model runtime. Simulator Python differs (local3.10, destination3.9), with Habitat0.2.4 and matching numpy/Pillow; rendering equivalence was checked. Only filesystem paths in the contract and scene prefixes in episode identities change.

The local collector synchronized journal/summary every 60 seconds, verified the sealed prefix and ordered unique episode IDs, and validated all 1,839 episodes at completion. Final evidence is saved in `../migration/final_gathered.json`; collection has finished. The native BF16 qualification failure remains unchanged.

## Final gathered result — 2026-10-27

779/1,839 successes; **SR42.359978%**, **SPL37.786090%**, zero execution errors. Remote execution finished at04:25:54+07; final gathering completed at04:27:35+07. `migration/final_gathered.json` confirms complete local validation. No evaluation remains running for this model. Qualification remains failed on all three BF16 fixtures.
