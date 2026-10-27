# M5 GDN S campaign — R2R evaluation results

## Final follow-up update — 2026-10-27

| Model | Episodes / 1,839 | Successes | SR (%) | SPL (%) | Errors | Navigation status | Numerical replay |
|---|---:|---:|---:|---:|---:|---|---|
| S8N epoch 2 | 1,839 | 912 | 49.592170 | 44.939655 | 0 | FINAL — unqualified diagnostic | FAIL (2/3 passed) |
| S8N LR1E5 | 1,839 | 908 | 49.374660 | 46.435271 | 0 | FINAL — navigation validated | PASS (3/3 passed) |
| S8N_MUON | 1,839 | 779 | 42.359978 | 37.786090 | 0 | FINAL — unqualified diagnostic | FAIL (0/3 passed) |

### Completed inference-budget sweep

| Model | Episodes / 1,839 | Successes | SR (%) | SPL (%) | Errors | Navigation status | Numerical replay |
|---|---:|---:|---:|---:|---:|---|---|
| S8N_R4_B600 | 1,839 | 947 | 51.495378 | 47.809805 | 0 | FINAL — navigation validated | PASS (3/3 passed) |
| S8N_R4_B2400 | 1,839 | 947 | 51.495378 | 47.924136 | 0 | FINAL — unqualified diagnostic | FAIL (1/3 passed) |
| S8N_R2_B1200 | 1,839 | 915 | 49.755302 | 46.151563 | 0 | FINAL — unqualified diagnostic | FAIL (1/3 passed) |
| S8N_R8_B1200 | 1,839 | 931 | 50.625340 | 47.094037 | 0 | FINAL — navigation validated | PASS (3/3 passed) |

All rows above cover 1,839 episodes with zero execution errors; qualification warnings remain. The original 14-run campaign below is preserved with its dated evidence. For all 43 navigation runs, see [the consolidated report](ALL_RESULTS.md).


Verified snapshot: **2026-10-24T21:21:47.211620+07:00** (Asia/Ho_Chi_Minh).
This supersedes the October 22 partial snapshot and baseline launch-only entries.
All 14 listed runs have attempted the full 1,839-episode panel. Thirteen passed
stored final navigation validation; S7 failed final validation due to one invalid
response. Numerical qualification is a separate axis and is retained below.

## Results

| Model | Attempted / 1,839 | Successes | SR (%) | SPL (%) | Errors | Navigation status |
|---|---:|---:|---:|---:|---:|---|
| M5_E1 R4/B1200 | 1,839 | 950 | 51.658510 | 48.401034 | 0 | FINAL — navigation validated |
| M5_E1 served B0 | 1,839 | 919 | 49.972811 | 45.991456 | 0 | FINAL — navigation validated |
| R4B0 trained | 1,839 | 896 | 48.722132 | 44.964877 | 0 | FINAL — navigation validated |
| S1 BF16 | 1,839 | 921 | 50.081566 | 46.539055 | 0 | FINAL — navigation validated |
| S1 FP32 | 1,839 | 917 | 49.864057 | 46.432850 | 0 | FINAL — navigation validated |
| S2 | 1,839 | 926 | 50.353453 | 46.896355 | 0 | FINAL — navigation validated |
| S3 | 1,839 | 928 | 50.462208 | 47.152476 | 0 | FINAL — navigation validated |
| S4 | 1,839 | 917 | 49.864057 | 46.671319 | 0 | FINAL — navigation validated |
| S5 | 1,839 | 901 | 48.994018 | 45.378002 | 0 | FINAL — navigation validated |
| S6 | 1,839 | 929 | 50.516585 | 47.013465 | 0 | FINAL — unqualified diagnostic |
| S7 | 1,839 | 907 | 49.320283 | 45.529253 | 1 | ALL ATTEMPTED — final validation FAILED |
| S8N | 1,839 | 964 | 52.419793 | 48.679311 | 0 | FINAL — navigation validated |
| S8W | 1,839 | 948 | 51.549755 | 48.690482 | 0 | FINAL — navigation validated |
| S9 | 1,839 | 932 | 50.679717 | 47.439929 | 0 | FINAL — navigation validated |

S6 is an **unqualified diagnostic**, even though its merged navigation journal
passed coverage, provenance and metric validation. S7's SR and SPL retain all
1,839 attempted episodes, counting its failed episode as zero. These are observed
failure-adjusted metrics, not a clean validated navigation result. No episode
was silently dropped, rerun, repaired or replaced in this report update.

## S7 failure and validation state

Episode **672** failed with `ValueError: ValueError: Invalid navigation response: 'MOVE'`.
The scene is `zsNo4HB9uLZ`; the full campaign ended with `phase=failed` when
`verify_result.py` rejected the recorded failure (the actual
validator log is `verify_full.log`). All 1,839 episode IDs are present exactly
once. There are **907 successes, one execution failure and one invalid response**.
The full observed SR is **49.320283%**, SPL **45.529253%**.

The retained `full_validation.json` contains an earlier **three-episode partial
check** (`complete=false`), not a successful final validation. The current report
uses the complete journal, summary and failed campaign/validator evidence. The
three numerical replay gates passed, illustrating that finite-fixture parity
qualification does not guarantee valid responses in every navigation episode.
Failure evidence: S7 run report (`/data14t/anhdh35/m5_s7_epoch1/evaluation/REPORT.md`),
validator log (`/data14t/anhdh35/m5_s7_epoch1/evaluation/verify_full.log`).

## Descriptive comparisons

S8N has the highest observed SR among these S-arm checkpoints: **964/1,839,
52.419793%**, 14 successes (+0.761283 percentage points) above the matched
M5_E1 R4/B1200 baseline. S8W has 948 successes (**51.549755%**), 16 fewer
than S8N, while their SPLs are close (48.690482% versus 48.679311%).
These are descriptive differences, not claims of statistical superiority.

The M5_E1 B1200 and served-B0 runs use the **same checkpoint**. Removing the
episodic bank lowers observed SR by 31 successes (**1.685699 points**).
Separately trained R4B0 has 896 successes, 23 fewer (**1.250680 points**) than
M5_E1 served with B0. The separately trained checkpoint must not be conflated
with the serving-only ablation. B0 retains native recurrent state, the prefix,
recent observations and action text; it is not a memory-free policy.

S1 FP32 has four fewer successes than S1 BF16 (917 versus 921). FP32 also changes
text attention to exact reference attention, vision attention to SDPA, disables
TF32 and uses Triton IEEE products. This is a serving-configuration comparison,
not an isolated causal dtype experiment. Similar aggregate SR does not imply
identical episode outcomes or that numerical differences are generally harmless.

| Comparison A vs B | Both succeed | A only | B only | Both fail |
|---|---:|---:|---:|---:|
| S1 BF16 vs S1 FP32 | 784 | 137 | 133 | 785 |
| M5_E1 R4/B1200 vs M5_E1 served B0 | 748 | 202 | 171 | 718 |
| M5_E1 served B0 vs R4B0 trained | 619 | 300 | 277 | 643 |
| S8N vs S8W | 675 | 289 | 273 | 602 |
| S8N vs M5_E1 R4/B1200 | 671 | 293 | 279 | 596 |

## Additional navigation and memory measurements

| Model | NE (m) | Oracle success (%) | Trajectory (m) | Mean actions | Forced STOPs | Peak KV rows | Peak GPU GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| M5_E1 R4/B1200 | 5.791268 | 58.020663 | 9.389865 | 93.5786 | 137 | 2646 | 8.6935 |
| M5_E1 served B0 | 6.015965 | 59.543230 | 10.332400 | 107.7243 | 179 | 1446 | 8.6316 |
| R4B0 trained | 6.162409 | 57.205003 | 9.859953 | 98.9880 | 153 | 1446 | 8.6329 |
| S1 BF16 | 5.774322 | 57.531267 | 9.653281 | 86.9576 | 108 | 2646 | 8.6909 |
| S1 FP32 | 5.921898 | 56.443719 | 9.523072 | 89.2474 | 122 | 2646 | 17.3715 |
| S2 | 5.797960 | 56.552474 | 9.309996 | 94.9337 | 145 | 2646 | 8.6781 |
| S3 | 5.809722 | 57.150625 | 9.351083 | 92.9277 | 138 | 2646 | 8.6781 |
| S4 | 5.635993 | 55.682436 | 9.258985 | 92.5247 | 137 | 2646 | 8.6909 |
| S5 | 6.094529 | 55.791191 | 9.466793 | 92.2953 | 132 | 2646 | 8.6909 |
| S6 | 5.767480 | 56.280587 | 9.381719 | 94.6525 | 143 | 2646 | 8.6710 |
| S7 | 5.980665 | 56.389342 | 9.857001 | 98.1419 | 145 | 2646 | 8.6935 |
| S8N | 5.710046 | 58.020663 | 9.401666 | 91.8793 | 130 | 2646 | 8.6949 |
| S8W | 5.457755 | 57.585644 | 9.189095 | 88.5829 | 122 | 2646 | 8.6948 |
| S9 | 5.689049 | 56.933116 | 9.415417 | 91.9744 | 128 | 2646 | 8.6706 |

nDTW is unavailable (zero recorded coverage), not zero performance. Latency and
peak allocations were measured with different concurrent workloads and runtimes;
they are not controlled throughput comparisons. S7 includes a truncated failed
episode in its observed trajectory/error statistics, with zero SR/SPL credit.

## Model definitions

All S arms use R4 including current and B1200. Token arms select individual
visual rows; frame arms retain four complete 300-row visual frames. Priority is
fixed at birth, original logical positions are retained, and native GDN state
persists. All listed S checkpoints are epoch 1 / seed 429; FP32 reuses S1 weights.

| Arm | Priority | Selection |
|---|---|---|
| S1 | Mean unscaled residual norm | Token |
| S2 | Mean beta × value norm | Token |
| S3 | Mean value norm | Token |
| S4 | Minimum cosine prediction error over heads | Token |
| S5 | Mean cosine prediction error over heads | Token |
| S6 | Mean committed-write norm over frozen slow heads | Token |
| S7 | Exact math.fsum of 300 S0 visual priorities / 300 | Whole frame |
| S8N | Net state change norm / max(prior-state norm, calibrated floor) | Whole frame |
| S8W | Decay-transported write norm / same denominator | Whole frame |
| S9 | Mean (1 − beta) × residual norm | Token |

S6's slow-head mask and S8's per-layer floors are frozen training artifacts.
S7 has no S8 drift calibration. The S8W write is accumulated with native decay,
not computed by subtracting large boundary states. M5_E1 uses the original
committed-write score; R4B0 disables the bank and scorer with guard3584.

## Numerical qualification

| Model | Saved numerical result |
|---|---|
| M5_E1 R4/B1200 | FAIL (2/3) |
| M5_E1 served B0 | PASS (3/3) |
| R4B0 trained | PASS (3/3) |
| S1 BF16 | FAIL (2/3) |
| S1 FP32 | PASS (saved FP32 diagnosis) |
| S2 | PASS (3/3) |
| S3 | PASS (3/3) |
| S4 | FAIL (2/3) |
| S5 | FAIL (1/3) |
| S6 | FAIL (1/2); unqualified diagnostic |
| S7 | PASS (3/3) |
| S8N | PASS (3/3) |
| S8W | PASS (3/3) |
| S9 | PASS (2/2; migrated fixture set) |

Unchanged native BF16 gate: elementwise absolute tolerance 0.75 plus relative
0.03 × abs(reference), RMS <= 0.15. An absolute maximum above 0.75 can pass via
the relative term. Forced canonical histories and decoding are not an independent
greedy-policy equivalence check. Selection overlap and top-token agreement remain
reported even when the numerical gate passes. Continued numerical-only failures
follow the existing user authorization; thresholds/model/weights were not changed.

| Model | Fixture | Max logit error | RMS | Top-token agreement (%) | Mean selected overlap (%) | Gate |
|---|---|---:|---:|---:|---:|---|
| M5_E1 R4/B1200 | 128 | 2.562500 | 0.037038 | 98.6945 | 98.9945 | FAIL |
| M5_E1 R4/B1200 | 64 | 0.906250 | 0.036861 | 98.4293 | 99.2382 | FAIL |
| M5_E1 R4/B1200 | previous_m5_128 | 0.656250 | 0.034792 | 100.0000 | 98.9615 | PASS |
| M5_E1 served B0 | 128 | 0.500000 | 0.033194 | 99.4778 | N/A | PASS |
| M5_E1 served B0 | 64 | 0.500000 | 0.034848 | 100.0000 | N/A | PASS |
| M5_E1 served B0 | previous_m5_128 | 0.406250 | 0.030537 | 99.4778 | N/A | PASS |
| R4B0 trained | 128 | 0.515625 | 0.040017 | 99.2167 | N/A | PASS |
| R4B0 trained | 64 | 0.515625 | 0.042083 | 99.4764 | N/A | PASS |
| R4B0 trained | previous_m5_128 | 0.625000 | 0.034461 | 99.4778 | N/A | PASS |
| S1 BF16 | 128 | 1.500000 | 0.056971 | 98.6945 | 97.7609 | FAIL |
| S1 BF16 | 64 | 1.156250 | 0.058749 | 98.9529 | 98.3917 | FAIL |
| S1 BF16 | previous_m5_128 | 1.062500 | 0.046826 | 99.2167 | 97.7476 | PASS |
| S2 | 128 | 0.937500 | 0.039204 | 99.2167 | 99.0150 | PASS |
| S2 | 64 | 0.718750 | 0.038884 | 99.4764 | 99.2474 | PASS |
| S2 | previous_m5_128 | 0.750000 | 0.036911 | 99.7389 | 98.9595 | PASS |
| S3 | 128 | 0.875000 | 0.046616 | 98.6945 | 98.0911 | PASS |
| S3 | 64 | 0.875000 | 0.045260 | 97.9058 | 98.6597 | PASS |
| S3 | previous_m5_128 | 0.765625 | 0.042803 | 100.0000 | 98.0994 | PASS |
| S4 | 128 | 1.093750 | 0.058886 | 99.4778 | 95.6093 | FAIL |
| S4 | 64 | 0.750000 | 0.048977 | 99.4764 | 96.8613 | PASS |
| S4 | previous_m5_128 | 1.023438 | 0.047561 | 99.4778 | 95.6780 | FAIL |
| S5 | 128 | 0.750000 | 0.043282 | 99.4778 | 98.0808 | PASS |
| S5 | 64 | 0.562500 | 0.041939 | 99.4764 | 98.6179 | PASS |
| S5 | previous_m5_128 | 0.875000 | 0.043358 | 100.0000 | 98.1618 | FAIL |
| S6 | migrated audit 64 | 0.546875 | 0.036578 | 98.9529 | 99.0167 | PASS |
| S6 | migrated audit 128 | 0.945312 | 0.034869 | 99.2167 | 98.8096 | FAIL |
| S7 | 128 | 0.500000 | 0.034808 | 98.9556 | 100.0000 | PASS |
| S7 | 64 | 0.500000 | 0.037030 | 98.9529 | 100.0000 | PASS |
| S7 | previous_m5_128 | 0.562500 | 0.031851 | 99.7389 | 100.0000 | PASS |
| S8N | 128 | 0.750000 | 0.033828 | 99.4778 | 99.7228 | PASS |
| S8N | 64 | 0.750000 | 0.035574 | 99.4764 | 99.4271 | PASS |
| S8N | previous_m5_128 | 0.562500 | 0.032977 | 100.0000 | 99.5464 | PASS |
| S8W | 128 | 0.625000 | 0.034789 | 98.9556 | 99.3448 | PASS |
| S8W | 64 | 0.500000 | 0.032856 | 99.4764 | 99.6875 | PASS |
| S8W | previous_m5_128 | 1.250000 | 0.054656 | 98.9556 | 95.0605 | PASS |
| S9 | migrated audit 64 | 0.500000 | 0.040891 | 98.9529 | 98.1960 | PASS |
| S9 | migrated audit 128 | 1.000000 | 0.038723 | 99.4778 | 97.5870 | PASS |

S6/S9 used their separately preserved two-fixture audit, not the three-fixture
workstation set. Their original journals (S6:1,152; S9:977 episodes) were preserved
at migration to ws_anhdh35 / 10.148.183.119. This refresh verified original journal
hashes, exact merged records, uniqueness and full membership. S6's numerical
failure remains an explicit unqualified-diagnostic designation.

S1 FP32's saved precision diagnosis is at
`/data14t/anhdh35/m5_s1_replay_diagnosis_20261021/precision/`: maxima
0.140746 / 0.143094 / 0.079043 and 100% top-token agreement on its three fixtures.
Those are saved diagnostic results, not fresh reruns during this documentation update.
They do not prove universal logic correctness or bit-identical selection.

Checkpoint-specific scorer, reset, bounded-cache and smoke evidence remains in
individual run reports. S7's actual-hidden FP64 committed-write oracle and exact
frame-mean checks passed; its later invalid response is retained independently.
New S4/S7/S8N/S8W launches ran 138 targeted source tests. Earlier campaigns have
their own test counts/references; no new model tests or evaluations were run for
this documentation refresh.

## Evaluation protocol and provenance

R2R VLN-CE val_unseen, 1,839 IDs, Habitat-Lab/Sim0.2.4, seed 42, max500 actions,
3m success radius, RGB640×480/HFOV79°, forward0.25m/turn15°, greedy canonical
text actions. Model and simulator use separate Python environments. Dataset hash
`1767a407e2c8a011fbb7abece76cd64c5b39ff9fa0e9e340ebdce5a490d167c3`;
standard Habitat config hash `dd20dd0e43e6dff3e552b63e9bc70c74062170c39352564758835c313bcf468f`.
Migration-specific scene paths, runtime and immutable evidence remain in the
S6/S9 migration folder; do not substitute paths when reproducing their episode UIDs.

| Model | HF revision | Weights SHA256 | Evaluation source |
|---|---|---|---|
| M5_E1 R4/B1200 | `95249f11a51e6d54d0e0f969ebcbac74db6464cd` | `92089fa16096f5fb581a7fb85a670b539a887278d7f66bacd20b4d0424d70ea6` | `fc017d9a5bbf1e80ae2c13151eb1aaaf38f0a7a6` |
| M5_E1 served B0 | `95249f11a51e6d54d0e0f969ebcbac74db6464cd` | `92089fa16096f5fb581a7fb85a670b539a887278d7f66bacd20b4d0424d70ea6` | `fc017d9a5bbf1e80ae2c13151eb1aaaf38f0a7a6` |
| R4B0 trained | `f2d87d8760f0c2a4a75a7ae5640a026ad48fa8c6` | `505fec28668b1bc950c2993cefccc86c939e66c450e00583e8f9bb2c563fe360` | `a2d075fca2a3a206f7b9bc45cb9d6cdf772f0aa7` |
| S1 BF16 | `0a481ea1d3b84907981d03c028c543aa2d8317e8` | `dd5876f50647fac8bfeb09bb4c7f6c8590dc796aa96db96642ec4b7af4e228df` | `fc017d9a5bbf1e80ae2c13151eb1aaaf38f0a7a6` |
| S1 FP32 | `0a481ea1d3b84907981d03c028c543aa2d8317e8` | `dd5876f50647fac8bfeb09bb4c7f6c8590dc796aa96db96642ec4b7af4e228df` | `fc017d9a5bbf1e80ae2c13151eb1aaaf38f0a7a6` |
| S2 | `54988913b2fc85de3dbac1c8df8398dabb4b916b` | `88dba8325a014868ca0633634ff0725dbb05b255e36b3d8fb9c78acefeab3044` | `fc017d9a5bbf1e80ae2c13151eb1aaaf38f0a7a6` |
| S3 | `a910d9429eb2e9d4bf7271f68d02c99abf4c4a4e` | `172b541a79a42d493c5acdb9c8c5478f8f540669577893cbf7163f94b0b0e9d1` | `fc017d9a5bbf1e80ae2c13151eb1aaaf38f0a7a6` |
| S4 | `c950a7af3c34985208ea827029cb74f80fbcff68` | `38b8b5aa4aa1cb7682df9678db6923e9a45286b015f505e3e53f6facbe416bcc` | `a2d075fca2a3a206f7b9bc45cb9d6cdf772f0aa7` |
| S5 | `2f4883da8a96a41b341a55114aeab531449bdee7` | `6d4c8e68101ec4dca67b6a8a771108ebdf2877a68fc5abd7dd10955162426f6b` | `fc017d9a5bbf1e80ae2c13151eb1aaaf38f0a7a6` |
| S6 | `ca57b99c9854502f1730c7551e4594aad5629c82` | `69d3f030a46b5ee4f05d75e7951ca1cee608fab0253ad788d5b571510e60a228` | `a2d075fca2a3a206f7b9bc45cb9d6cdf772f0aa7` |
| S7 | `e99a0d8c93299c833e66ba27a07ad12ff13718f0` | `b927f44f6920ecfeef183851c3135cc503d34ffac53ebfc67515968c292daaeb` | `a2d075fca2a3a206f7b9bc45cb9d6cdf772f0aa7` |
| S8N | `b6a1ac8099568fa5f740e45488d36d12e4e44337` | `c89de946f3cb05aa91a5cc498fd63991c901852ba0c05e6546cbaddf23446a2b` | `a2d075fca2a3a206f7b9bc45cb9d6cdf772f0aa7` |
| S8W | `5ddc8042528c9361fc295ec6891606f57e239807` | `113482a5ce90b4d6819fb2669f94cc18e91a293d6390daf0cc4c5a9a1fe8b755` | `a2d075fca2a3a206f7b9bc45cb9d6cdf772f0aa7` |
| S9 | `ca57b99c9854502f1730c7551e4594aad5629c82` | `1b4602baed643b791332c4329fd18a37ed01142db0a3fd6bc927f7cfce15dfc1` | `a2d075fca2a3a206f7b9bc45cb9d6cdf772f0aa7` |

## Evidence locations

| Model | Report | Full journal |
|---|---|---|
| M5_E1 R4/B1200 | Report (`/data14t/anhdh35/m5_e1_s_campaign_epoch1/evaluation/REPORT.md`) | episodes.jsonl (`/data14t/anhdh35/m5_e1_s_campaign_epoch1/evaluation/results/full/M5_E1_R4_B1200/episodes.jsonl`) |
| M5_E1 served B0 | Report (`/data14t/anhdh35/m5_e1_s_campaign_b0_epoch1/evaluation/REPORT.md`) | episodes.jsonl (`/data14t/anhdh35/m5_e1_s_campaign_b0_epoch1/evaluation/results/full/M5_E1_R4_B0/episodes.jsonl`) |
| R4B0 trained | Report (`/data14t/anhdh35/m5_r4b0_trained_epoch1/evaluation/REPORT.md`) | episodes.jsonl (`/data14t/anhdh35/m5_r4b0_trained_epoch1/evaluation/results/full/R4B0_trained_s429/episodes.jsonl`) |
| S1 BF16 | Report (`/data14t/anhdh35/m5_s1_epoch1/evaluation/REPORT.md`) | episodes.jsonl (`/data14t/anhdh35/m5_s1_epoch1/evaluation/results/full/S1_s429/episodes.jsonl`) |
| S1 FP32 | Report (`/data14t/anhdh35/m5_s1_fp32_epoch1/evaluation/REPORT.md`) | episodes.jsonl (`/data14t/anhdh35/m5_s1_fp32_epoch1/evaluation/results/full/S1_s429_FP32/episodes.jsonl`) |
| S2 | Report (`/data14t/anhdh35/m5_s2_epoch1/evaluation/REPORT.md`) | episodes.jsonl (`/data14t/anhdh35/m5_s2_epoch1/evaluation/results/full/S2_s429/episodes.jsonl`) |
| S3 | Report (`/data14t/anhdh35/m5_s3_epoch1/evaluation/REPORT.md`) | episodes.jsonl (`/data14t/anhdh35/m5_s3_epoch1/evaluation/results/full/S3_s429/episodes.jsonl`) |
| S4 | Report (`/data14t/anhdh35/m5_s4_epoch1/evaluation/REPORT.md`) | episodes.jsonl (`/data14t/anhdh35/m5_s4_epoch1/evaluation/results/full/S4_s429/episodes.jsonl`) |
| S5 | Report (`/data14t/anhdh35/m5_s5_epoch1/evaluation/REPORT.md`) | episodes.jsonl (`/data14t/anhdh35/m5_s5_epoch1/evaluation/results/full/S5_s429/episodes.jsonl`) |
| S6 | Report (`/data14t/anhdh35/m5_s6_s9_20261023/continuation/S6/REPORT.md`) | episodes.jsonl (`/data14t/anhdh35/m5_s6_s9_20261023/continuation/S6/episodes.jsonl`) |
| S7 | Report (`/data14t/anhdh35/m5_s7_epoch1/evaluation/REPORT.md`) | episodes.jsonl (`/data14t/anhdh35/m5_s7_epoch1/evaluation/results/full/S7_s429/episodes.jsonl`) |
| S8N | Report (`/data14t/anhdh35/m5_s8n_epoch1/evaluation/REPORT.md`) | episodes.jsonl (`/data14t/anhdh35/m5_s8n_epoch1/evaluation/results/full/S8N_s429/episodes.jsonl`) |
| S8W | Report (`/data14t/anhdh35/m5_s8w_epoch1/evaluation/REPORT.md`) | episodes.jsonl (`/data14t/anhdh35/m5_s8w_epoch1/evaluation/results/full/S8W_s429/episodes.jsonl`) |
| S9 | Report (`/data14t/anhdh35/m5_s6_s9_20261023/continuation/S9/REPORT.md`) | episodes.jsonl (`/data14t/anhdh35/m5_s6_s9_20261023/continuation/S9/episodes.jsonl`) |

[Machine-readable snapshot](evidence/ALL_RESULTS_20261024.json) contains recomputed metrics, journal hashes,
coverage/action checks, source/runtime/contract metadata, saved qualification and
validation evidence, and paired outcomes. The previous October22 snapshot remains
available for history. No journals or original validator outcomes were rewritten.
See [episodic-memory results](EPISODIC_MEMORY_EVALUATION.md) and
[VGGT results](VGGT_EVALUATION.md) for the older campaigns. M5 R4/B1200 seed431
is excluded from this refresh per the user's instruction.
