# S8N epoch 2 — R2R evaluation

Snapshot 2026-10-27T07:08:12.063755+07:00. **Complete: 912/1,839 successes; SR49.592170%; SPL44.939655%; zero execution errors.** Full navigation validation passed. Native BF16 qualification failed on one of three fixtures, so the run remains an **unqualified diagnostic**.

## Identity and training interpretation

Checkpoint `/data14t/anhdh35/m5_s8n_epoch2/checkpoint/S8N_E2` from `anhdao69/M5_S/S8N_E2`, pinned Hub revision `770041b7f1ef5add5354d6fc3a34246a9c2451b2`.
Weight SHA256 `d13132e49f7053031b5248a00e3fa6c52c86b01f912f359abfe3fd9468ae25de`.
Exact training/evaluation source: `m5_s8n_followup` commit `a8bb0a2bfdd854c80221ba895100846801d359a3`.

The published README, initialization/policy identity, source/completion evidence and follow-up controller were reviewed. This is one additional epoch initialized from S8N epoch1 weights with **fresh AdamW state**, the same warmup/cosine schedule, and sampler epoch offset1. It is not an uninterrupted optimizer-state continuation. The second stage has 3,852 updates; total exposure across the two stages is 7,704 updates. Published action-weighted epoch loss0.07427 is training evidence, not navigation SR.

Navigation metadata exactly matches epoch1: S8N net-state-drift frame scoring, R4/B1200, 300 tokens per retained frame, guard4784, same calibration, serializer, action history and inference configuration. Backbone BF16; native score computation/attention unchanged. No serving-budget or precision override.

## Verified before GPU execution

- All23 downloaded remote file identities and22 published checksums verified.
- All724 model tensors finite; only expected backbone serving tensors.
- Parent weight hash, arm YAML hash, policy hash, exported/training config and exact source revision verified.
- **138 focused CPU tests passed** (memory contract, GDN math, selectors and forward paths).
- Separate data/journals from epoch1 and every budget-sweep arm.

## GPU replay results

| Fixture | Max logit error | RMS | Top-token agreement | Mean bank overlap | Native gate |
|---|---:|---:|---:|---:|---|
| 64 steps | 0.5000 | 0.03469 | 100.000% | 97.396% | PASS |
| 128 steps | 1.7500 | 0.07651 | 99.739% | 96.144% | FAIL |
| Previous M5 128 steps | 0.5000 | 0.03448 | 100.000% | 99.975% | PASS |

Gate remains torch.allclose(atol0.75, rtol0.03) plus RMS≤0.15. No threshold was relaxed. Forced-action decoded agreement is a protocol check, not free-greedy equivalence. The failed replay and selected-bank differences are retained, and no claim that rounding alone explains them is made.

The independent S8N FP64 helper oracle, bitwise reset, all-layer frame/cache invariants and520-step bounded stream are checked separately. The established campaign permits a disclosed numerical-only failure only when structural, finite, reset, RMS and replay-protocol checks pass; otherwise it stops. Two Habitat smoke episodes and strict validation precede the full run.

## Requested evaluation

The user confirmed starting standard R2R evaluation after downloading. Target: the same1,839 val_unseen episodes, fixed order, seed42, maximum500 steps, RGB480×640 and original action mapping as epoch1. Model Python `/storage/anhdh35/SimpleMemVLN/.venv/bin/python`; Habitat Python `/storage/anhdh35/miniconda3/envs/spatialstack_dagger/bin/python`.

At epoch2 launch, the four budget runs were stopped at the user's request. They were later explicitly resumed and all four are now complete; L2/L3 remain interrupted. See S8N_R_B_SWEEP.md.

Epoch1 reference:964/1,839 successes, SR52.419793%, SPL48.679311%. Any incomplete epoch2 SR must be labeled partial; no comparison to the full reference should be interpreted as a paired final result before coverage completes.

## Evidence and monitoring

Root: `/data14t/anhdh35/m5_s8n_epoch2/evaluation`.
- Download verification (`/data14t/anhdh35/m5_s8n_epoch2/evaluation/download_verification.json`)
- Campaign status (`/data14t/anhdh35/m5_s8n_epoch2/evaluation/campaign_status.json`)
- GPU audit (`/data14t/anhdh35/m5_s8n_epoch2/evaluation/audit/S8N_E2/`)
- Stopped sweep evidence (`/data14t/anhdh35/s8n_rb_sweep_20261025/stop_evidence.json`)
- Full output: `results/full/S8N_E2_s429/episodes.jsonl`, `summary.json`, `runtime.json`, `evaluation_contract.json`.
- `verify_result.py full` recomputes metrics and checks hashes, episode order/coverage, failures and action metadata. A partial validation is never final validation.

## Historical launch verification

Download/provenance and138 CPU tests passed. Independent helper, bitwise reset, frame invariants and520-step bounded stream passed (peak2546 KV rows against guard4784;8.681 GiB peak allocated). Both Habitat smoke episodes completed with zero execution errors;0/2 successes is only a setup observation, not a meaningful performance estimate.

Historical launch PIDs: campaign21471, evaluator21922. Evaluation is now complete; full_validation.json confirms all1,839 episodes and recomputed metrics. **Numerical qualification remains failed (2/3 replay fixtures passed)**, so results are labeled unqualified diagnostic.

Launch verification (`/data14t/anhdh35/m5_s8n_epoch2/evaluation/launch_verification.json`).
