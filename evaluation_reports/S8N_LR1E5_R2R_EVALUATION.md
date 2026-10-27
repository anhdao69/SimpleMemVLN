# S8N_LR1E5 — R2R evaluation

Snapshot 2026-10-27T07:08:14.403816+07:00. Campaign phase: **complete**. Numerical qualification: **True** (null means pending).

## Checkpoint and training

- Hub: `anhdao69/M5_S/S8N_LR1E5`, pinned revision `7c4720d5c663479b2d8d661c1b7b7e11eaa696c4`.
- Local checkpoint: `/data14t/anhdh35/m5_s8n_lr1e5/checkpoint/S8N_LR1E5`.
- Weight SHA256: `382089ca9d84ad26f774ded861684500dd383dca73f3184c8439b33ed626c650`.
- Branch `m5_s8n_followup`, exact training/evaluation commit `a8bb0a2bfdd854c80221ba895100846801d359a3`; isolated clean checkout.

The published README, run identity, policy YAML, training/publication validation and follow-up controller were reviewed. This model was trained for **one epoch from original Qwen3.5-4B**, with peak backbone and classifier learning rates **1e-5**, rather than original S8N's5e-6. It is not an epoch2 continuation. Joint R2R/English RxR15deg, seed429, 3,852 updates, warmup/cosine schedule and training order are the matched S8N recipe. Published action-weighted loss0.12575 is training evidence, not navigation SR.

The exported navigation config was independently compared with original S8N: **only training.backbone_lr and training.classifier_lr differ**, both equal1e-5. S8N scoring, R4/B1200, frame300 selection, guard4784, calibration, action history and generation protocol are unchanged. No inference override or calibration refit. Backbone serves in BF16 with the original native scorer and attention.

## Integrity and validation

- All22 remote file identities and21 published SHA256 entries verified.
- All724 serving tensors finite and expected backbone keys only.
- Training config, exported metadata, arm/policy hashes and source revision verified.
- The identical source commit already passed138 focused CPU tests; evidence is reused from `/data14t/anhdh35/m5_s8n_epoch2/evaluation/tests.log`, not claimed as a new test execution.
- Fresh checkpoint-specific GPU qualification: strict reload, independent FP64 S8N helper oracle, saved64/128/previous-M5-128 replay fixtures, frame/cache invariants, bitwise greedy reset and520-step bounded stream.
- Native numerical thresholds remain allclose(atol0.75, rtol0.03) and RMS≤0.15. Forced decoded-action agreement is a protocol check, not free-greedy equivalence. Any numerical failure is disclosed as unqualified diagnostic; core structural/nonfinite/RMS/protocol failures block the run.

## GPU replay evidence

| Fixture | Max error | RMS | Top-token agreement | Mean bank overlap | Gate |
|---|---:|---:|---:|---:|---|
| 64 | 0.5625 | 0.02498 | 100.000% | 99.896% | PASS |
| 128 | 0.5625 | 0.02461 | 100.000% | 99.950% | PASS |
| Previous M5 128 | 0.8125 | 0.02505 | 99.739% | 97.883% | PASS |

Helper passed: True; bitwise reset: True; bounded-stream passed: True, peak KV2546 / guard4784, peak allocated8.681 GiB.

## R2R protocol and monitoring

Two Habitat smoke episodes must pass strict journal validation before all **1,839 R2R val_unseen episodes**. Same fixed order, seed42, max500 steps, RGB480×640, scenes/dataset and action mapping as previous S8N evaluations. Full validation recomputes SR/SPL, checks exact identities, episode order/coverage and execution errors. Partial validation is not final validation.

Original S8N reference:964/1,839 successes, SR52.419793%, SPL48.679311%. Incomplete SR should not be interpreted as a matched full-panel comparison. Epoch2 and the resumed budget sweep have also completed; see their separate reports. Shared GPU runtime timings are not controlled latency benchmarks.

Model Python `/storage/anhdh35/SimpleMemVLN/.venv/bin/python`; Habitat Python `/storage/anhdh35/miniconda3/envs/spatialstack_dagger/bin/python`.
Evidence root `/data14t/anhdh35/m5_s8n_lr1e5/evaluation`. Verification (`/data14t/anhdh35/m5_s8n_lr1e5/evaluation/download_verification.json`), campaign status (`/data14t/anhdh35/m5_s8n_lr1e5/evaluation/campaign_status.json`), audit (`/data14t/anhdh35/m5_s8n_lr1e5/evaluation/audit/S8N_LR1E5/`).
Full journals: `results/full/S8N_LR1E5_s429/episodes.jsonl`, `summary.json`, `runtime.json`, `evaluation_contract.json`.

smoke_validation.json: 2/2 episodes, failures0, passed=True, complete=True. This is a validation snapshot, not an automatically refreshed live progress count.

full_validation.json: 1839/1839 episodes, failures0, passed=True, complete=True. This is a validation snapshot, not an automatically refreshed live progress count.

Final result: 1839/1,839 episodes; SR 49.374660%; SPL 46.435271%; execution failures 0; complete=True.
