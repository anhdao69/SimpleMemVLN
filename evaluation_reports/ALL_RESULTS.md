# All R2R evaluation results

Publication check: [fresh read-only verification](evidence/WS_REPORT_SYNC_VERIFICATION_20261026.json) covers 43 navigation journals and the 35 existing latency configurations. [Sync manifest](evidence/WS_REPORT_SYNC_MANIFEST_20261026.json) distinguishes the server snapshot from these merged reports.

Verified snapshot: **2026-10-27T07:08:12.063755+07:00**. Metrics independently recomputed from saved episode journals; unique IDs, panel membership, available contract order and action metadata checked. There are **43 navigation runs** in this index: 40 full-panel attempts and three partial runs (L2, L3, G3_M5). Of the full-panel attempts, S7 contains one execution failure and failed final navigation validation. Numerical qualification is separate from navigation validation. No evaluation was restarted by this report update.

S8N_MUON, L0/L1, both S8N training follow-ups and all four resumed budget arms are complete. L2/L3 remain interrupted; G3_M5 is archived partial. S8N_MUON finished on `10.148.182.137`, and all 1,839 episodes were gathered and validated locally. Its migration preserved 1,301 episodes and verified weights/source/assets plus 169 matching smoke frames; only path prefixes changed.

S8N epoch1 has the highest completed SR in this index: **964/1,839 (52.419793%)**. Comparisons below are descriptive, not significance claims. Partial rows must not be ranked as full-panel results. S6, S8N epoch2, S8N_MUON, R4/B2400 and R2/B1200 retain explicit unqualified-diagnostic labels. S1 BF16, S4, S5 and M5_E1 R4/B1200 also retain native numerical-replay failures despite navigation validation passing.

[Machine-readable evidence](evidence/ALL_RESULTS_20261027.json) · [CSV](evidence/ALL_RESULTS_20261027.csv) · [Latency and SR](evidence/ALL_LATENCY_SR_20261027.csv).

## S campaign and matched controls

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

## Learned rankers

| Model | Episodes / 1,839 | Successes | SR (%) | SPL (%) | Errors | Navigation status | Numerical replay |
|---|---:|---:|---:|---:|---:|---|---|
| L0 | 1,839 | 951 | 51.712887 | 48.202080 | 0 | FINAL — navigation validated | PASS (3/3 passed) |
| L1 | 1,839 | 932 | 50.679717 | 47.172627 | 0 | FINAL — navigation validated | PASS (3/3 passed) |
| L2 | 1,384 | 678 | 48.988439 | 45.418499 | 0 | PARTIAL — interrupted; not running | PASS (3/3 passed) |
| L3 | 1,199 | 593 | 49.457882 | 46.412354 | 0 | PARTIAL — interrupted; not running | PASS (3/3 passed) |

## S8N training follow-ups

| Model | Episodes / 1,839 | Successes | SR (%) | SPL (%) | Errors | Navigation status | Numerical replay |
|---|---:|---:|---:|---:|---:|---|---|
| S8N epoch 2 | 1,839 | 912 | 49.592170 | 44.939655 | 0 | FINAL — unqualified diagnostic | FAIL (2/3 passed) |
| S8N LR1E5 | 1,839 | 908 | 49.374660 | 46.435271 | 0 | FINAL — navigation validated | PASS (3/3 passed) |
| S8N_MUON | 1,839 | 779 | 42.359978 | 37.786090 | 0 | FINAL — unqualified diagnostic | FAIL (0/3 passed) |

## S8N inference-budget sweep

| Model | Episodes / 1,839 | Successes | SR (%) | SPL (%) | Errors | Navigation status | Numerical replay |
|---|---:|---:|---:|---:|---:|---|---|
| S8N_R4_B600 | 1,839 | 947 | 51.495378 | 47.809805 | 0 | FINAL — navigation validated | PASS (3/3 passed) |
| S8N_R4_B2400 | 1,839 | 947 | 51.495378 | 47.924136 | 0 | FINAL — unqualified diagnostic | FAIL (1/3 passed) |
| S8N_R2_B1200 | 1,839 | 915 | 49.755302 | 46.151563 | 0 | FINAL — unqualified diagnostic | FAIL (1/3 passed) |
| S8N_R8_B1200 | 1,839 | 931 | 50.625340 | 47.094037 | 0 | FINAL — navigation validated | PASS (3/3 passed) |

## Earlier episodic models and baselines

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

## VGGT campaign

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

## Interpretation and provenance

R2R val_unseen has1,839 episodes; the standard protocol uses seed 42, a 500-step cap, 0.25m moves, 15-degree turns and a 3m success radius. Exact per-run contracts, checkpoints, source hashes and runtime metadata are indexed in the JSON evidence. Historical configurations differ; the model names alone do not establish matched training or inference settings.

S7's failure is episode 672 (`MOVE`, invalid navigation response); denominator remains1,839 and the failed episode has zero SR/SPL credit. No failed episode was removed or replaced. M5_E1 served B0 reuses its trained B1200 checkpoint, whereas R4B0 trained is a separately trained model. S1 FP32 also changes attention implementation and TF32 settings, so it is not an isolated dtype comparison.

S8N epoch2 is a weights-only extra epoch with fresh AdamW state, not an uninterrupted optimizer continuation. LR1E5 starts from the original base with higher training LR. MUON starts from the original base with a different matrix optimizer; its inference uses the same S8N R4/B1200 configuration. Native BF16 parity failed on all three MUON fixtures; the successful migration does not remove that warning.

The budget sweep uses one original S8N checkpoint with inference-only R/B overrides and unchanged calibration. R includes the current observation; B counts visual tokens in complete 300-token frames. R4/B1200 is the shared baseline, not an additional trained model.

The protected M5_E1 R4/B1200 seed431 report was not edited; its journal is read only for this comprehensive index. Controlled latency has35 measured configurations; new configurations without controlled measurements are marked unmeasured. No new latency benchmark was run.
