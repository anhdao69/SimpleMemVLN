# S8N inference-only bank and recent-context sweep

Snapshot: 2026-10-27T07:08:18.352007+07:00. Controller phase: **complete**. Full target: 1,839 R2R val_unseen episodes per setting. Incomplete SRs are provisional.

## Bank capacity at R4

| Setting | R | B tokens (frames) | Completed / 1,839 | Successes | SR | SPL | Baseline SR on same episodes | Δ SR (pp) | Execution errors | Status |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| S8N_R4_B600 | 4 | 600 (2) | 1839 | 947 | 51.50% | 47.81% | 52.42% | -0.92 | 0 | complete |
| S8N_R4_B1200 | 4 | 1200 (4) | 1839 | 964 | 52.42% | 48.68% | 52.42% | +0.00 | 0 | baseline_reused |
| S8N_R4_B2400 | 4 | 2400 (8) | 1839 | 947 | 51.50% | 47.92% | 52.42% | -0.92 | 0 | complete; UNQUALIFIED diagnostic |

## Recent context at B1200

| Setting | R | B tokens (frames) | Completed / 1,839 | Successes | SR | SPL | Baseline SR on same episodes | Δ SR (pp) | Execution errors | Status |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| S8N_R2_B1200 | 2 | 1200 (4) | 1839 | 915 | 49.76% | 46.15% | 52.42% | -2.66 | 0 | complete; UNQUALIFIED diagnostic |
| S8N_R4_B1200 | 4 | 1200 (4) | 1839 | 964 | 52.42% | 48.68% | 52.42% | +0.00 | 0 | baseline_reused |
| S8N_R8_B1200 | 8 | 1200 (4) | 1839 | 931 | 50.63% | 47.09% | 52.42% | -1.79 | 0 | complete |

R4/B1200 is shared between the two comparisons, not a sixth run. Its verified complete result is reused (964/1,839, SR52.42%, SPL48.68%). Fresh two-episode baseline control passed: **True**. Partial arms are compared to baseline on **exactly the same completed episode IDs**, not to its full-panel SR. Final whole-panel deltas become comparable when all runs cover 1,839 episodes. These tests measure serving-budget sensitivity of one trained S8N checkpoint, not retrained policies.

## Frozen protocol and changes

- Original checkpoint: `/data14t/anhdh35/m5_s8n_epoch1/checkpoint/S8N`; weight SHA256 `c89de946f3cb05aa91a5cc498fd63991c901852ba0c05e6546cbaddf23446a2b`.
- Frozen source commit `a2d075fca2a3a206f7b9bc45cb9d6cdf772f0aa7`; source `/data14t/anhdh35/m5_s8n_epoch1/source`.
- BF16 backbone and original S8N scoring/attention code. Frozen drift-denominator floors and calibration hashes unchanged; no refit or training.
- Only recent_groups_including_current, episodic_budget_tokens and the derived kv_guard_tokens change. R includes the current observation. B is visual-token capacity; S8N admits whole300-token frames.
- Guard =2048 +384×R +B, from the checkpoint's maximum prefix/group lengths; a safety bound, not a target resident length.
- Identical dataset, scenes, RGB480×640, action mapping, appended text history, 500-step cap, seed42 and episode order as the original S8N evaluation.
- Original weight/tokenizer files are shared read-only by symlink; derived navigation metadata lives in separate per-arm directories and is explicitly marked inference-only. Original journals/checkpoint/source are preserved.
- Each changed setting receives strict model reload, independent FP64 S8N helper check, three native BF16 offline/streaming replays, all-layer frame/cache invariants, bitwise reset, 520-step bounded stream and two-episode smoke before full navigation.
- Existing numerical thresholds are unchanged. A native BF16 numerical-only failure is recorded as unqualified diagnostic, never rewritten as a pass; structural/finite/RMS/protocol failures block that arm. Qualification on altered budgets does not imply identical training/serving selections.
- Qualification runs sequentially; full jobs run concurrently. Observed wall/model timings under shared GPU load are not controlled latency benchmarks.
- The evaluator resumes saved completed episodes only when its full identity contract matches. To resume after interruption: run `/data14t/anhdh35/s8n_rb_sweep_20261025/campaign.py` with the model Python after confirming an old controller is absent. Completed qualification evidence is reused; the baseline smoke is rechecked.

## Per-arm numerical qualification

| Arm | Passed original replay thresholds | Phase | Error |
|---|---|---|---|
| S8N_R4_B1200 | original completed evaluation; fresh control | baseline_reused | — |
| S8N_R4_B600 | True | complete | — |
| S8N_R4_B2400 | False | complete | — |
| S8N_R2_B1200 | False | complete | — |
| S8N_R8_B1200 | True | complete | — |

### Checkpoint-specific replay details

Native tolerance: `atol=0.75, rtol=0.03` and RMS≤0.15. Forced-action decoding is a replay protocol check, not free-greedy agreement.

| Setting | Fixture | Max logit error | RMS | Top-token agreement | Mean bank overlap | Gate |
|---|---|---:|---:|---:|---:|---|
| S8N_R4_B600 | 64 | 0.8125 | 0.03257 | 100.000% | 100.000% | PASS |
| S8N_R4_B600 | 128 | 0.8125 | 0.03120 | 99.739% | 100.000% | PASS |
| S8N_R4_B600 | previous M5 128 | 0.5000 | 0.03165 | 98.956% | 100.000% | PASS |
| S8N_R4_B2400 | 64 | 0.8750 | 0.04697 | 99.476% | 99.193% | FAIL |
| S8N_R4_B2400 | 128 | 1.6250 | 0.05043 | 99.478% | 98.677% | FAIL |
| S8N_R4_B2400 | previous M5 128 | 0.5000 | 0.03390 | 99.478% | 97.896% | PASS |
| S8N_R2_B1200 | 64 | 1.3906 | 0.05439 | 98.429% | 99.597% | FAIL |
| S8N_R2_B1200 | 128 | 5.2500 | 0.10570 | 98.956% | 97.966% | FAIL |
| S8N_R2_B1200 | previous M5 128 | 0.5625 | 0.03430 | 100.000% | 100.000% | PASS |
| S8N_R8_B1200 | 64 | 0.4375 | 0.03393 | 100.000% | 100.000% | PASS |
| S8N_R8_B1200 | 128 | 0.5000 | 0.03258 | 100.000% | 100.000% | PASS |
| S8N_R8_B1200 | previous M5 128 | 0.5625 | 0.03225 | 99.739% | 99.974% | PASS |

## Evidence

Root: `/data14t/anhdh35/s8n_rb_sweep_20261025`. Manifest (`/data14t/anhdh35/s8n_rb_sweep_20261025/manifest.json`), live status (`/data14t/anhdh35/s8n_rb_sweep_20261025/status.json`), baseline control (`/data14t/anhdh35/s8n_rb_sweep_20261025/baseline_control.json`), machine-readable snapshot (`/data14t/anhdh35/s8n_rb_sweep_20261025/results_snapshot.json`). Each arm has `checkpoint/override_manifest.json`, `evaluation/audit/S8N/`, smoke/full validation and immutable contract-bound journals under `evaluation/results/`.
