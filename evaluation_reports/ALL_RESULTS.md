# Evaluation results

## Trained R/B ablations — final SR

Updated: **2026-11-13**. Final observed success rates for eight trained R/B configurations, including the original S8N baseline. Every configuration has complete R2R and RxR results; R12B0's resumed R2R evaluation has finished.

| Model | SR R2R (%) | SR RxR (%) |
|---|---:|---:|
| R2B1200 | 49.21 | 54.16 |
| R4B0 | 48.72 | 52.93 |
| R4B600 | 50.84 | 56.26 |
| R4B1200 — S8N baseline | 52.42 | 54.35 |
| R4B2400 | 51.33 | 55.19 |
| R8B0 | 48.99 | 53.20 |
| R8B1200 | 50.95 | 53.97 |
| R12B0 | 51.22 | 54.65 |

R2R uses **1,839 val_unseen episodes**. RxR uses **3,669 English-guide val_unseen episodes** (en-US and en-IN) with **15° turns**. All episodes remain in the denominators; failed episodes receive zero success credit. SR is shown as a percentage rounded to two decimal places.

R counts recent observation groups including the current observation. B is the episodic visual-token budget. R4B1200 is the original S8N baseline. The earlier [inference-only budget sweep](S8N_R_B_SWEEP.md) has its own results and evidence.

[SR table CSV](evidence/S8N_R_B_ABLATION_SR_20261113.csv) · [Full-precision counts and source evidence](evidence/S8N_R_B_ABLATION_SR_20261113.json). The evidence index records success counts, full episode counts, journal digests, source-result paths, and the completed R12B0 recovery certificate.

[Dedicated trained-ablation report](S8N_R_B_ABLATION.md). [Sync verification](evidence/S8N_R_B_ABLATION_SYNC_VERIFICATION_20261113.json) records source hashes, fresh journal checks and the certificate-only verification limits.

## Historical R2R index through 2026-11-01

Publication check: [fresh read-only verification](evidence/WS_REPORT_SYNC_VERIFICATION_20261026.json) covers 43 navigation journals and the 35 existing latency configurations. [Sync manifest](evidence/WS_REPORT_SYNC_MANIFEST_20261026.json) distinguishes the server snapshot from these merged reports.

Verified snapshot: **2026-10-27T07:08:12.063755+07:00**. Metrics independently recomputed from saved episode journals; unique IDs, panel membership, available contract order and action metadata checked. This snapshot indexed **43 navigation runs**: 40 full-panel attempts and three partial runs (L2, L3, G3_M5). Of the full-panel attempts, S7 contains one execution failure and failed final navigation validation. Numerical qualification is separate from navigation validation. No evaluation was restarted by this report update.

**2026-11-01 update — S8N stage 2.** Three full-panel runs were added: S8N_DAGGER_SF, S8N_DAGGER_SF_REMAINING and S8N_DAGGER_SF_FULL. Each covers all 1,839 episodes with zero execution errors and passed final verification. The index now has **46 navigation runs**: 43 full-panel attempts and three partial runs. The 2026-10-27 snapshot, the publication check and the JSON/CSV links below do not cover these three rows. They are verified by their own [evidence](evidence/S8N_STAGE2_SF_R2R_20261101.json), [per-episode CSV](evidence/S8N_STAGE2_SF_R2R_20261101_EPISODES.csv) and [verifier](evidence/verify_s8n_stage2_sf_20261101.py), described in the [stage-2 report](S8N_STAGE2_STOPFIX_R2R_EVALUATION.md). No earlier row or verification claim was changed.

S8N_MUON, L0/L1, both S8N training follow-ups and all four resumed budget arms are complete. L2/L3 remain interrupted; G3_M5 is archived partial. S8N_MUON finished on `10.148.182.137`, and all 1,839 episodes were gathered and validated locally. Its migration preserved 1,301 episodes and verified weights/source/assets plus 169 matching smoke frames; only path prefixes changed.

S8N_DAGGER_SF_REMAINING (stage 2) has the highest completed SR in this index: **1,171/1,839 (63.675911%)**. Among first-stage models, S8N epoch1 remains highest at **964/1,839 (52.419793%)**. Comparisons below are descriptive, not significance claims. The exception is the stage-2 section, whose paired McNemar tests and bootstrap intervals are in its report. Partial rows must not be ranked as full-panel results. S6, S8N epoch2, S8N_MUON, R4/B2400 and R2/B1200 retain explicit unqualified-diagnostic labels. S1 BF16, S4, S5 and M5_E1 R4/B1200 also retain native numerical-replay failures despite navigation validation passing.

[Machine-readable evidence](evidence/ALL_RESULTS_20261027.json) · [CSV](evidence/ALL_RESULTS_20261027.csv) · [Latency and SR](evidence/ALL_LATENCY_SR_20261027.csv).

### S campaign and matched controls

| Model | Episodes / 1,839 | Successes | SR (%) | SPL (%) | Errors | Navigation status | Numerical replay |
|---|---:|---:|---:|---:|---:|---|---|
| M5_E1 R4/B1200 | 1,839 | 950 | 51.658510 | 48.401034 | 0 | FINAL — navigation validated | FAIL (1/3 passed) |
| M5_E1 served B0 | 1,839 | 919 | 49.972811 | 45.991456 | 0 | FINAL — navigation validated | PASS (3/3 passed) |
| R4B0 trained | 1,839 | 896 | 48.722132 | 44.964877 | 0 | FINAL — navigation validated | PASS (3/3 passed) |
| S1 BF16 | 1,839 | 921 | 50.081566 | 46.539055 | 0 | FINAL — navigation validated | FAIL (1/3 passed) |
| S1 FP32 | 1,839 | 917 | 49.864057 | 46.432850 | 0 | FINAL — navigation validated | PASS (saved FP32 diagnosis) |
| S2 | 1,839 | 926 | 50.353453 | 46.896355 | 0 | FINAL — navigation validated | PASS (3/3 passed) |
| S3 | 1,839 | 928 | 50.462208 | 47.152476 | 0 | FINAL — navigation validated | PASS (3/3 passed) |
| S4 | 1,839 | 917 | 49.864057 | 46.671319 | 0 | FINAL — navigation validated | FAIL (1/3 passed) |
| S5 | 1,839 | 901 | 48.994018 | 45.378002 | 0 | FINAL — navigation validated | FAIL (2/3 passed) |
| S6 | 1,839 | 929 | 50.516585 | 47.013465 | 0 | FINAL — unqualified diagnostic | FAIL; diagnostic |
| S7 | 1,839 | 907 | 49.320283 | 45.529253 | 1 | ALL ATTEMPTED — final validation FAILED | PASS (3/3 passed) |
| S8N | 1,839 | 964 | 52.419793 | 48.679311 | 0 | FINAL — navigation validated | PASS (3/3 passed) |
| S8W | 1,839 | 948 | 51.549755 | 48.690482 | 0 | FINAL — navigation validated | PASS (3/3 passed) |
| S9 | 1,839 | 932 | 50.679717 | 47.439929 | 0 | FINAL — navigation validated | PASS 2/2 (migrated fixtures) |

### Learned rankers

| Model | Episodes / 1,839 | Successes | SR (%) | SPL (%) | Errors | Navigation status | Numerical replay |
|---|---:|---:|---:|---:|---:|---|---|
| L0 | 1,839 | 951 | 51.712887 | 48.202080 | 0 | FINAL — navigation validated | PASS (3/3 passed) |
| L1 | 1,839 | 932 | 50.679717 | 47.172627 | 0 | FINAL — navigation validated | PASS (3/3 passed) |
| L2 | 1,384 | 678 | 48.988439 | 45.418499 | 0 | PARTIAL — interrupted; not running | PASS (3/3 passed) |
| L3 | 1,199 | 593 | 49.457882 | 46.412354 | 0 | PARTIAL — interrupted; not running | PASS (3/3 passed) |

### S8N training follow-ups

| Model | Episodes / 1,839 | Successes | SR (%) | SPL (%) | Errors | Navigation status | Numerical replay |
|---|---:|---:|---:|---:|---:|---|---|
| S8N epoch 2 | 1,839 | 912 | 49.592170 | 44.939655 | 0 | FINAL — unqualified diagnostic | FAIL (2/3 passed) |
| S8N LR1E5 | 1,839 | 908 | 49.374660 | 46.435271 | 0 | FINAL — navigation validated | PASS (3/3 passed) |
| S8N_MUON | 1,839 | 779 | 42.359978 | 37.786090 | 0 | FINAL — unqualified diagnostic | FAIL (0/3 passed) |

### S8N stage 2: DAgger with the STOP-label fix

| Model | Episodes / 1,839 | Successes | SR (%) | SPL (%) | Errors | Navigation status | Numerical replay |
|---|---:|---:|---:|---:|---:|---|---|
| S8N_DAGGER_SF | 1,839 | 1,117 | 60.739532 | 48.838608 | 0 | FINAL — navigation validated | PASS (serving audit 2/2) |
| S8N_DAGGER_SF_REMAINING | 1,839 | 1,171 | 63.675911 | 56.654234 | 0 | FINAL — navigation validated | PASS (serving audit 2/2) |
| S8N_DAGGER_SF_FULL | 1,839 | 1,111 | 60.413268 | 51.854865 | 0 | FINAL — navigation validated | PASS (serving audit 2/2) |

- **Training:** each model is one epoch starting from S8N epoch1's weights, with fresh AdamW, on oracle-relabeled DAgger data with the STOP-label fix.
  - SF trains on that data alone.
  - SF_REMAINING adds the 16,620 original R2R/RxR15deg episodes whose instructions DAgger did not re-collect.
  - SF_FULL adds all 30,815 original episodes.
- **Inference:** the same R4/B1200 configuration as S8N.
- **Navigation status:** from each campaign's `finish_verify.py`, which checks order, uniqueness, shard/merged equality and the recomputed summary.
- **"Serving audit 2/2":** the stage-2 qualification. It requires 64- and 128-step forced replays within allclose(0.75, 0.03) with RMS ≤ 0.15, a bitwise reset and a 520-step bounded stream. It is not the first-stage three-fixture native gate.
- **Significance:** against S8N on the same episodes, all three SR gains remain significant after Bonferroni correction over six comparisons. See the [stage-2 report](S8N_STAGE2_STOPFIX_R2R_EVALUATION.md).

### S8N inference-budget sweep

| Model | Episodes / 1,839 | Successes | SR (%) | SPL (%) | Errors | Navigation status | Numerical replay |
|---|---:|---:|---:|---:|---:|---|---|
| S8N_R4_B600 | 1,839 | 947 | 51.495378 | 47.809805 | 0 | FINAL — navigation validated | PASS (3/3 passed) |
| S8N_R4_B2400 | 1,839 | 947 | 51.495378 | 47.924136 | 0 | FINAL — unqualified diagnostic | FAIL (1/3 passed) |
| S8N_R2_B1200 | 1,839 | 915 | 49.755302 | 46.151563 | 0 | FINAL — unqualified diagnostic | FAIL (1/3 passed) |
| S8N_R8_B1200 | 1,839 | 931 | 50.625340 | 47.094037 | 0 | FINAL — navigation validated | PASS (3/3 passed) |

### Earlier episodic models and baselines

| Model | Episodes / 1,839 | Successes | SR (%) | SPL (%) | Errors | Navigation status | Numerical replay |
|---|---:|---:|---:|---:|---:|---|---|
| V0 | 1,839 | 852 | 46.329527 | 41.756802 | 0 | COMPLETE — recomputed | FAIL (0/3 passed) |
| Window8-B-E1 | 1,839 | 907 | 49.320283 | 45.561094 | 0 | COMPLETE — recomputed | See historical evidence |
| M1 | 1,839 | 799 | 43.447526 | 40.204753 | 0 | COMPLETE — recomputed | PASS (2/2 passed) |
| M2 | 1,839 | 906 | 49.265905 | 45.877272 | 0 | COMPLETE — recomputed | FAIL (0/3 passed) |
| M2c | 1,839 | 848 | 46.112017 | 43.365987 | 0 | COMPLETE — recomputed | FAIL (0/3 passed) |
| M5 | 1,839 | 938 | 51.005982 | 47.438938 | 0 | COMPLETE — recomputed | PASS (4/4 passed) |
| M5c | 1,839 | 877 | 47.688961 | 43.944998 | 0 | COMPLETE — recomputed | PASS (2/2 passed) |
| M5 E1 R2/B1800 seed431 | 1,839 | 944 | 51.332246 | 47.525482 | 0 | FINAL — navigation validated | PASS (2/2 passed) |
| M5 E1 R4/B1200 seed431 | 1,839 | 884 | 48.069603 | 44.099314 | 0 | FINAL — navigation validated | FAIL (1/2 passed) |

### VGGT campaign

| Model | Episodes / 1,839 | Successes | SR (%) | SPL (%) | Errors | Navigation status | Numerical replay |
|---|---:|---:|---:|---:|---:|---|---|
| G1 | 1,839 | 914 | 49.700924 | 46.072125 | 0 | COMPLETE — recomputed | See historical evidence |
| G3 | 1,839 | 956 | 51.984774 | 48.095941 | 0 | COMPLETE — recomputed | PASS (4/4 passed) |
| G3_fixed01_seed430 | 1,839 | 909 | 49.429038 | 45.655180 | 0 | COMPLETE — recomputed | See historical evidence |
| G3 DINOv3 ViTB16 | 1,839 | 879 | 47.797716 | 44.574182 | 0 | COMPLETE — recomputed | See historical evidence |
| G3A1 | 1,839 | 877 | 47.688961 | 43.548813 | 0 | COMPLETE — recomputed | See historical evidence |
| G3_M5 | 135 | 64 | 47.407407 | 42.989009 | 0 | PARTIAL — archived | PASS (2/2 passed) |
| A0 | 1,839 | 910 | 49.483415 | 45.394182 | 0 | COMPLETE — recomputed | See historical evidence |
| A1 | 1,839 | 929 | 50.516585 | 46.873430 | 0 | COMPLETE — recomputed | See historical evidence |
| R1 | 1,839 | 907 | 49.320283 | 45.825211 | 0 | COMPLETE — recomputed | See historical evidence |

### Interpretation and provenance

R2R val_unseen has1,839 episodes; the standard protocol uses seed 42, a 500-step cap, 0.25m moves, 15-degree turns and a 3m success radius. Exact per-run contracts, checkpoints, source hashes and runtime metadata are indexed in the JSON evidence. Historical configurations differ; the model names alone do not establish matched training or inference settings.

S7's failure is episode 672 (`MOVE`, invalid navigation response); denominator remains1,839 and the failed episode has zero SR/SPL credit. No failed episode was removed or replaced. M5_E1 served B0 reuses its trained B1200 checkpoint, whereas R4B0 trained is a separately trained model. S1 FP32 also changes attention implementation and TF32 settings, so it is not an isolated dtype comparison.

S8N epoch2 is a weights-only extra epoch with fresh AdamW state, not an uninterrupted optimizer continuation. LR1E5 starts from the original base with higher training LR. MUON starts from the original base with a different matrix optimizer; its inference uses the same S8N R4/B1200 configuration. Native BF16 parity failed on all three MUON fixtures; the successful migration does not remove that warning.

The budget sweep uses one original S8N checkpoint with inference-only R/B overrides and unchanged calibration. R includes the current observation; B counts visual tokens in complete 300-token frames. R4/B1200 is the shared baseline, not an additional trained model.

The protected M5_E1 R4/B1200 seed431 report was not edited; its journal is read only for this comprehensive index. Controlled latency has35 measured configurations; new configurations without controlled measurements are marked unmeasured. No new latency benchmark was run.

The S8N stage-2 arms train one epoch from S8N epoch1's final weights on different data, described in the [stage-2 report](S8N_STAGE2_STOPFIX_R2R_EVALUATION.md). They have no controlled latency measurement. Their navigation configuration is unchanged from S8N, R4/B1200 with guard 4784.
