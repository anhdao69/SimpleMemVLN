# Controlled inference latency — all available evaluated models

Publication check: [fresh read-only verification](evidence/WS_REPORT_SYNC_VERIFICATION_20261024.json) covers 33 navigation journals and 35 latency configurations. [Sync manifest](evidence/WS_REPORT_SYNC_MANIFEST_20261024.json) distinguishes original server snapshots from the merged reports.

## L2/L3 latency and SR update — 2026-10-25

| Model | Median ms | p95 ms | Mean ms | Actions/s | R2R successes / completed | SR | SPL | Navigation status |
|---|---:|---:|---:|---:|---|---:|---:|---|
| L2 | 109.716 | 115.079 | 110.094 | 9.083 | 678 / 1384 | 48.99% | 45.42% | Interrupted; partial / 1,839 |
| L3 | 153.127 | 168.287 | 155.638 | 6.425 | 593 / 1199 | 49.46% | 46.41% | Interrupted; partial / 1,839 |

Measured 2026-10-25T10:00:28.687170+07:00 on the same RTX PRO6000 Blackwell Max-Q, driver580.178.04 and 300W limit. This is a **separate post-reboot session** using the unchanged original benchmark worker: batch1, the same 80 preloaded RGB frames/instruction, three trials of16 warm-up +64 timed actions (192 samples/model), CUDA-synchronized observe wall time; simulator, transport and disk reads excluded. No detected foreign GPU process. Native BF16 backbone and embedded learned rankers retained (feature preparation FP32, ranker FP64, priorities FP32). Warm-up/initialization excluded; this is not a cold-start benchmark or a proof of cross-session statistical superiority.

Navigation results were independently recomputed from saved journals with contract-order and source/config identity checks. The workstation rebooted at **2026-10-25 08:35:33 +07:00**; both evaluation processes are absent. These SRs are **partial, interrupted results**, not completed evaluations. L2 and L3 cover different-sized ordered prefixes, so their SRs are not a matched full-panel comparison. No navigation evaluation was restarted by this latency measurement.

Raw benchmark evidence (`/data14t/anhdh35/latency_l_20261025/manifest.json`) · [Verification](evidence/L_LATENCY_SR_20261025_VERIFICATION.json) · [All 35 configurations: latency and SR CSV](evidence/ALL_LATENCY_SR_20261025.csv).

The original 33-configuration study below is preserved as its dated snapshot; with the two additions above, **35 configurations and 6,720 timed actions** are available.


Snapshot: **2026-10-24T22:16:41.341083+07:00**.
**33/33 configurations have complete measurements with no detected GPU contention**, totaling **6,336 timed actions**.
Raw observations and reproducible scripts: `/data14t/anhdh35/latency_20261024`.
This is a latency benchmark, not a new navigation evaluation or qualification pass.

## Scope and method

All locally available evaluated configurations are included: historical V0/M1/M2/M2c/M5/M5c,
Window8-B, VGGT variants (including the preserved G3_M5 TRIPLE checkpoint), the complete S campaign, M5_E1 B1200/B0, separately
trained R4B0, and both seed431 R2/B1800 and R4/B1200 checkpoints. The latter is
measured only in this new latency report; its existing navigation report is untouched.
S1 FP32 is a separate serving configuration of S1, not a separate checkpoint.

- Batch size **1**, one sequential episode stream and one model process at a time on the same GPU; concurrent compute PIDs are monitored.
- Each model uses its frozen evaluation source and original checkpoint configuration.
  Checkpoint SHA256, navigation metadata and applicable source hashes are verified.
- Identical 80 real Habitat RGB frames (640×480) and one instruction from the
  saved current-workstation 128-step fixture. Images are loaded before timing.
  Fixture source: `/data14t/anhdh35/m5_s3_epoch1/evaluation/replay_128.json`.
  Instruction: “Walk into the living room and keep walking straight past the living room. Then walk into the entrance under the balcony. Wait in the entrance to the other room.”
- Three resets/trials per model; each trial uses 16 warm-up actions followed by
  64 timed actions (steps16–79): **192 headline samples/model**. Each trial uses
  the same episode UID, frame order and instruction; greedy responses are not forced.
- STOP does not terminate the open-loop benchmark. Predicted actions do not control
  the subsequent frames. It measures inference on matched inputs, not closed-loop SR.
- Wall time is synchronized with CUDA immediately before and after `observe`.
  It includes CPU preprocessing, vision encoding, memory selection/update, streaming
  language model, autoregressive action generation and response bookkeeping.
- Excludes disk image reads, checkpoint loading, episode reset, warm-up, simulator
  stepping/rendering, interprocess image transport and network communication.
- CPU intra-op threads4, OMP/MKL4. Native BF16 and frozen attention code are retained.
  S1 FP32 keeps its saved exact text-attention/SDPA-vision patch, TF32 disabled and
  TRITON_F32_DEFAULT=ieee. Runtime flags are recorded per process.
- S3 is first to validate the harness; remaining model order is shuffled with seed429.
  TRIPLE was discovered later in the preserved artifact inventory and runs last.
  No GPU clock/power limit changes or model optimizations are made. CPU affinity is not pinned.

Median, p95 and p99 use all 192 measured samples; quantiles use linear interpolation.
Actions/s = sample count / total measured seconds. Per-trial means show variability;
no confidence interval or statistical significance is claimed for these correlated samples.
This 80-step workload does not establish worst-case 500-step latency or performance
on all scenes/instruction lengths. Learned action history can differ across models.

## Controlled per-action latency

Sorted by median. Only complete runs with no detected GPU contention appear here.

| Model/configuration | Samples | Mean ms | Median ms | p95 ms | p99 ms | Actions/s | Tokens/action | Peak GPU GiB | Peak KV rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G1 | 192 | 85.726 | 84.669 | 92.168 | 93.034 | 11.665 | 3.000 | 8.649 | 2618 |
| G3_fixed01_seed430 | 192 | 85.690 | 84.841 | 92.211 | 93.667 | 11.670 | 3.000 | 8.649 | 2618 |
| G3 DINOv3 ViTB16 | 192 | 86.539 | 84.872 | 92.027 | 93.804 | 11.556 | 3.000 | 8.649 | 2618 |
| Window8-B-E1 | 192 | 85.608 | 84.875 | 92.036 | 92.417 | 11.681 | 3.000 | 8.649 | 2618 |
| R4B0 trained | 192 | 85.170 | 84.878 | 85.838 | 91.961 | 11.741 | 3.000 | 8.610 | 1346 |
| M1 | 192 | 86.888 | 85.139 | 92.894 | 96.032 | 11.509 | 3.000 | 8.659 | 2936 |
| M5_E1 served B0 | 192 | 85.186 | 85.224 | 85.754 | 86.120 | 11.739 | 3.000 | 8.610 | 1346 |
| G3 | 192 | 87.332 | 85.326 | 92.807 | 94.685 | 11.451 | 3.000 | 8.649 | 2618 |
| A1 | 192 | 86.973 | 85.789 | 92.977 | 93.631 | 11.498 | 3.000 | 8.649 | 2618 |
| V0 | 192 | 85.894 | 85.855 | 86.581 | 87.082 | 11.642 | 3.000 | 8.649 | 2618 |
| M2c | 192 | 88.892 | 86.083 | 93.897 | 94.811 | 11.250 | 3.000 | 8.658 | 2918 |
| A0 | 192 | 86.128 | 86.088 | 86.618 | 86.896 | 11.611 | 3.000 | 8.649 | 2618 |
| G3A1 | 192 | 87.472 | 86.299 | 93.400 | 94.119 | 11.432 | 3.000 | 8.649 | 2618 |
| M5c | 192 | 89.077 | 87.341 | 95.290 | 95.852 | 11.226 | 3.000 | 8.649 | 2546 |
| S2 | 192 | 99.334 | 95.522 | 105.586 | 107.286 | 10.067 | 3.000 | 8.663 | 2546 |
| S3 | 192 | 97.756 | 96.685 | 104.639 | 105.938 | 10.230 | 3.000 | 8.663 | 2546 |
| R1 | 192 | 100.908 | 99.572 | 109.207 | 111.894 | 9.910 | 3.000 | 8.689 | 2618 |
| S6 | 192 | 103.307 | 102.524 | 111.776 | 112.239 | 9.680 | 3.000 | 8.676 | 2546 |
| M2 | 192 | 102.639 | 102.586 | 103.582 | 104.222 | 9.743 | 3.000 | 8.690 | 2918 |
| S7 | 192 | 105.104 | 102.919 | 112.544 | 113.754 | 9.514 | 3.000 | 8.679 | 2546 |
| M5 | 192 | 105.586 | 103.041 | 112.454 | 116.571 | 9.471 | 3.000 | 8.679 | 2546 |
| M5 E1 R4/B1200 seed431 | 192 | 103.324 | 103.273 | 103.660 | 104.113 | 9.678 | 3.000 | 8.679 | 2546 |
| M5_E1 R4/B1200 | 192 | 105.414 | 103.345 | 112.633 | 112.903 | 9.486 | 3.000 | 8.679 | 2546 |
| G3_M5 | 192 | 106.967 | 103.505 | 113.573 | 116.221 | 9.349 | 3.000 | 8.679 | 2546 |
| S8N | 192 | 103.918 | 103.511 | 104.688 | 112.558 | 9.623 | 3.000 | 8.681 | 2546 |
| G3_M5 L8/L16/L24 TRIPLE | 192 | 106.743 | 103.723 | 112.798 | 115.155 | 9.368 | 3.000 | 8.679 | 2546 |
| M5 E1 R2/B1800 seed431 | 192 | 105.618 | 103.883 | 113.918 | 117.461 | 9.468 | 3.000 | 8.679 | 2510 |
| S8W | 192 | 105.603 | 104.223 | 113.495 | 114.141 | 9.469 | 3.000 | 8.681 | 2546 |
| S1 BF16 | 192 | 109.063 | 106.164 | 116.074 | 117.340 | 9.169 | 3.000 | 8.676 | 2546 |
| S9 | 192 | 110.839 | 107.301 | 117.119 | 118.516 | 9.022 | 3.000 | 8.676 | 2546 |
| S4 | 192 | 110.179 | 107.778 | 118.311 | 119.429 | 9.076 | 3.000 | 8.676 | 2546 |
| S5 | 192 | 108.707 | 108.048 | 117.528 | 117.845 | 9.199 | 3.000 | 8.676 | 2546 |
| S1 FP32 | 192 | 198.991 | 199.059 | 200.806 | 200.911 | 5.025 | 3.000 | 17.360 | 2546 |

## Interpretation and matched serving comparisons

The measured BF16 median range is **84.669–108.048 ms/action**. Differences of a few milliseconds should be read alongside trial variability, rather than treated as a definitive architectural ranking.

- **S1: FP32 versus BF16:** 106.164 → 199.059 ms median, a **1.875×** latency ratio (+87.5%). Throughput: 9.169 → 5.025 actions/s; peak allocated memory: 8.676 → 17.360 GiB.
- **M5_E1: B1200 versus served B0:** 85.224 → 103.345 ms median, a **1.213×** latency ratio (+21.3%). Throughput: 11.739 → 9.486 actions/s; peak allocated memory: 8.610 → 8.679 GiB.

S1 changes precision **and attention implementation**, so its ratio is the cost of the complete FP32 serving configuration, not an isolated dtype experiment. M5_E1 served B0 disables the episodic bank of the same trained model; separately trained R4B0 is a different checkpoint. The matched RGB inputs do not force equal generated action histories. Neither comparison establishes a change in navigation accuracy.

Repeated measured token sequences were identical across all three trials in **33/33** completed configurations. These are within-model repetitions, not agreement between different models.

## Effective memory configuration

Values below are read from the loaded model configuration, rather than inferred from model names. FP32 S1 uses the saved patch to replace attention even though its serialized memory backend name remains the BF16 training name.

| Model | Memory mode | Recent groups including current | Episodic budget | Score ID / method |
|---|---|---:|---:|---|
| G1 | window8 | 8 | N/A | N/A |
| G3_fixed01_seed430 | window8 | 8 | N/A | N/A |
| G3 DINOv3 ViTB16 | window8 | 8 | N/A | N/A |
| Window8-B-E1 | window8 | 8 | N/A | N/A |
| R4B0 trained | episodic_window_v1 | 4 | 0 | none |
| M1 | episodic_window_v1 | 9 | 0 | none |
| M5_E1 served B0 | episodic_window_v1 | 4 | 0 | none |
| G3 | window8 | 8 | N/A | N/A |
| A1 | window8 | 8 | N/A | N/A |
| V0 | episodic_window_v1 | 8 | 0 | none |
| M2c | episodic_window_v1 | 8 | 300 | random_hash_v1 |
| A0 | window8 | 8 | N/A | N/A |
| G3A1 | window8 | 8 | N/A | N/A |
| M5c | episodic_window_v1 | 4 | 1200 | random_hash_v1 |
| S2 | episodic_window_gdn_sweep_v1 | 4 | 1200 | S2 |
| S3 | episodic_window_gdn_sweep_v1 | 4 | 1200 | S3 |
| R1 | window8 | 8 | N/A | N/A |
| S6 | episodic_window_gdn_sweep_v1 | 4 | 1200 | S6 |
| M2 | episodic_window_v1 | 8 | 300 | committed_mean_l2_v1 |
| S7 | episodic_window_gdn_sweep_v1 | 4 | 1200 | S7 |
| M5 | episodic_window_v1 | 4 | 1200 | committed_mean_l2_v1 |
| M5 E1 R4/B1200 seed431 | episodic_window_v1 | 4 | 1200 | committed_mean_l2_v1 |
| M5_E1 R4/B1200 | episodic_window_v1 | 4 | 1200 | committed_mean_l2_v1 |
| G3_M5 | episodic_window_v1 | 4 | 1200 | committed_mean_l2_v1 |
| S8N | episodic_window_gdn_sweep_v1 | 4 | 1200 | S8N |
| G3_M5 L8/L16/L24 TRIPLE | episodic_window_v1 | 4 | 1200 | committed_mean_l2_v1 |
| M5 E1 R2/B1800 seed431 | episodic_window_v1 | 2 | 1800 | committed_mean_l2_v1 |
| S8W | episodic_window_gdn_sweep_v1 | 4 | 1200 | S8W |
| S1 BF16 | episodic_window_gdn_sweep_v1 | 4 | 1200 | S1 |
| S9 | episodic_window_gdn_sweep_v1 | 4 | 1200 | S9 |
| S4 | episodic_window_gdn_sweep_v1 | 4 | 1200 | S4 |
| S5 | episodic_window_gdn_sweep_v1 | 4 | 1200 | S5 |
| S1 FP32 | episodic_window_gdn_sweep_v1 | 4 | 1200 | S1 |

## Repeatability and startup costs

Load time starts at checkpoint loader invocation and includes synchronization,
but excludes Python imports, integrity hashing and image loading. It is **not a
cold-disk boot benchmark**: OS/file caches and kernel caches may already be warm.
First action and first reset can include initialization/compilation; they are
excluded from headline steady measurements. FP32 conversion time is separate.

| Model | Trial means ms (1 / 2 / 3) | First reset ms | Later reset mean ms | First action ms | Checkpoint load s | FP32 conversion s | Repeated token sequence identical |
|---|---|---:|---:|---:|---:|---:|---|
| G1 | 84.751 / 84.681 / 87.746 | 516.750 | 20.370 | 299.285 | 5.615 | 0.000 | True |
| G3_fixed01_seed430 | 84.587 / 85.233 / 87.252 | 475.095 | 20.706 | 286.503 | 5.458 | 0.000 | True |
| G3 DINOv3 ViTB16 | 85.391 / 89.274 / 84.951 | 478.102 | 23.541 | 272.528 | 5.445 | 0.000 | True |
| Window8-B-E1 | 84.851 / 84.815 / 87.158 | 445.503 | 20.508 | 278.797 | 5.512 | 0.000 | True |
| R4B0 trained | 85.394 / 84.823 / 85.292 | 424.903 | 20.438 | 268.945 | 5.109 | 0.000 | True |
| M1 | 85.967 / 89.513 / 85.183 | 442.149 | 20.789 | 281.011 | 5.441 | 0.000 | True |
| M5_E1 served B0 | 84.859 / 85.402 / 85.296 | 449.452 | 20.774 | 245.213 | 5.380 | 0.000 | True |
| G3 | 88.427 / 85.596 / 87.973 | 476.703 | 22.010 | 290.527 | 5.481 | 0.000 | True |
| A1 | 88.268 / 85.662 / 86.988 | 553.824 | 20.574 | 272.319 | 5.301 | 0.000 | True |
| V0 | 86.027 / 86.117 / 85.538 | 354.767 | 20.956 | 180.957 | 5.264 | 0.000 | True |
| M2c | 85.843 / 89.351 / 91.481 | 493.933 | 20.203 | 291.244 | 5.596 | 0.000 | True |
| A0 | 86.274 / 86.257 / 85.853 | 499.388 | 20.783 | 272.454 | 5.419 | 0.000 | True |
| G3A1 | 88.178 / 87.909 / 86.328 | 497.278 | 23.816 | 274.703 | 5.541 | 0.000 | True |
| M5c | 86.580 / 93.124 / 87.526 | 472.356 | 20.530 | 290.483 | 5.391 | 0.000 | True |
| S2 | 97.601 / 101.640 / 98.762 | 539.628 | 25.705 | 265.524 | 5.382 | 0.000 | True |
| S3 | 96.725 / 99.379 / 97.164 | 525.629 | 22.682 | 312.577 | 6.104 | 0.000 | True |
| R1 | 101.045 / 101.927 / 99.751 | 400.295 | 24.890 | 224.991 | 5.560 | 0.000 | True |
| S6 | 105.063 / 102.266 / 102.591 | 608.602 | 26.137 | 242.107 | 5.478 | 0.000 | True |
| M2 | 102.173 / 102.748 / 102.997 | 606.352 | 26.049 | 302.474 | 5.489 | 0.000 | True |
| S7 | 104.155 / 103.058 / 108.099 | 505.692 | 26.710 | 233.207 | 5.327 | 0.000 | True |
| M5 | 106.724 / 107.164 / 102.869 | 413.999 | 25.860 | 207.634 | 5.273 | 0.000 | True |
| M5 E1 R4/B1200 seed431 | 103.394 / 103.333 / 103.244 | 561.684 | 26.139 | 289.679 | 5.486 | 0.000 | True |
| M5_E1 R4/B1200 | 107.117 / 105.980 / 103.146 | 531.673 | 27.945 | 258.162 | 5.323 | 0.000 | True |
| G3_M5 | 105.432 / 104.502 / 110.966 | 409.283 | 28.013 | 237.968 | 6.202 | 0.000 | True |
| S8N | 104.457 / 103.746 / 103.551 | 390.803 | 28.004 | 190.850 | 5.316 | 0.000 | True |
| G3_M5 L8/L16/L24 TRIPLE | 111.033 / 105.489 / 103.708 | 575.484 | 28.253 | 263.511 | 12.257 | 0.000 | True |
| M5 E1 R2/B1800 seed431 | 109.112 / 103.961 / 103.782 | 567.913 | 28.056 | 284.116 | 5.356 | 0.000 | True |
| S8W | 105.054 / 107.285 / 104.469 | 518.000 | 26.253 | 248.445 | 5.390 | 0.000 | True |
| S1 BF16 | 107.679 / 112.502 / 107.009 | 393.369 | 29.193 | 189.624 | 5.221 | 0.000 | True |
| S9 | 114.378 / 109.887 / 108.253 | 564.972 | 29.412 | 282.464 | 5.508 | 0.000 | True |
| S4 | 107.801 / 110.487 / 112.250 | 547.198 | 30.304 | 252.050 | 5.236 | 0.000 | True |
| S5 | 108.238 / 109.557 / 108.324 | 529.989 | 30.793 | 222.492 | 5.368 | 0.000 | True |
| S1 FP32 | 197.927 / 198.714 / 200.333 | 431.022 | 47.139 | 268.958 | 5.325 | 0.029 | True |

## Instrumented component timings — diagnostic only

A separate pass enables the existing synchronized component timer: 16 warm-up
steps and16 profiled steps. These samples are excluded from headline results.
Synchronization adds overhead; these numbers must not be substituted for the main
benchmark. Components can be nested: **feedback LM append is inside autoregressive
decode**, so columns are not additive. Memory/scorer work is included in the
corresponding model stages rather than exposed as a universally comparable timer.
“Observation LM” denotes the native streaming_language_model_append timer;
module boundaries are those of each frozen implementation. N/A means absent,
not zero cost. No online VGGT teacher is artificially enabled for timing.

| Model | Preprocess ms | Vision ms | Observation LM ms | Feedback LM ms | AR decode ms | Instrumented total ms |
|---|---:|---:|---:|---:|---:|---:|
| G1 | 2.975 | 8.753 | 25.582 | 43.730 | 46.922 | 85.436 |
| G3_fixed01_seed430 | 2.958 | 8.740 | 25.590 | 43.019 | 46.193 | 84.709 |
| G3 DINOv3 ViTB16 | N/A | N/A | N/A | N/A | N/A | 85.076 |
| Window8-B-E1 | 2.964 | 8.745 | 25.553 | 43.662 | 46.857 | 85.321 |
| R4B0 trained | 2.863 | 8.780 | 25.298 | 43.726 | 47.195 | 85.632 |
| M1 | 3.154 | 8.953 | 25.912 | 48.471 | 52.035 | 91.893 |
| M5_E1 served B0 | 2.920 | 8.762 | 25.263 | 43.885 | 47.361 | 85.775 |
| G3 | 2.972 | 8.753 | 25.630 | 44.139 | 47.328 | 85.897 |
| A1 | 3.081 | 8.729 | 26.015 | 43.982 | 47.190 | 86.245 |
| V0 | 2.941 | 8.811 | 25.611 | 44.100 | 47.601 | 86.630 |
| M2c | 2.918 | 8.787 | 25.707 | 42.898 | 46.422 | 86.755 |
| A0 | 2.994 | 8.747 | 26.010 | 44.067 | 47.269 | 86.217 |
| G3A1 | 3.030 | 8.777 | 26.086 | 44.150 | 47.380 | 86.540 |
| M5c | 2.941 | 8.803 | 25.597 | 43.971 | 47.496 | 88.199 |
| S2 | 2.956 | 8.782 | 29.926 | 48.578 | 52.105 | 96.378 |
| S3 | 2.927 | 8.793 | 29.922 | 48.632 | 52.146 | 96.401 |
| R1 | 3.137 | 8.786 | 29.735 | 53.362 | 56.586 | 99.499 |
| S6 | 2.941 | 8.759 | 34.144 | 51.646 | 55.163 | 103.552 |
| M2 | 2.991 | 8.764 | 34.364 | 52.046 | 55.583 | 103.889 |
| S7 | 3.211 | 8.950 | 35.510 | 59.380 | 63.045 | 112.770 |
| M5 | 2.996 | 8.781 | 34.104 | 51.624 | 55.146 | 103.630 |
| M5 E1 R4/B1200 seed431 | 2.947 | 8.772 | 34.252 | 52.420 | 55.952 | 104.543 |
| M5_E1 R4/B1200 | 3.048 | 8.793 | 34.275 | 52.043 | 55.577 | 104.336 |
| G3_M5 | 3.200 | 8.915 | 35.111 | 57.912 | 61.557 | 111.645 |
| S8N | 2.917 | 8.736 | 35.492 | 51.862 | 55.374 | 104.357 |
| G3_M5 L8/L16/L24 TRIPLE | 2.975 | 8.777 | 34.162 | 52.162 | 55.694 | 104.216 |
| M5 E1 R2/B1800 seed431 | 3.124 | 8.756 | 34.203 | 52.366 | 55.904 | 104.980 |
| S8W | 2.975 | 8.782 | 35.616 | 52.261 | 55.793 | 105.022 |
| S1 BF16 | 2.979 | 8.766 | 36.242 | 52.440 | 55.966 | 106.577 |
| S9 | 3.215 | 8.938 | 37.280 | 59.031 | 62.708 | 114.989 |
| S4 | 3.251 | 8.888 | 38.114 | 58.123 | 61.721 | 114.762 |
| S5 | 3.000 | 8.752 | 37.275 | 53.435 | 56.962 | 108.588 |
| S1 FP32 | 3.097 | 28.627 | 106.537 | 56.009 | 61.838 | 202.842 |

## Previously observed full-navigation timings

These figures come from saved navigation journals/summaries, often under shared
GPU load and differing episode lengths/actions. They are **observational, not a
controlled speed ranking**. “Transport + model” includes subprocess serialization
and IPC. Environment time includes simulator stepping and rendering. Do not add
separately computed p50/p95 values to estimate a total-loop percentile.
G3_M5 is an archived partial135-episode snapshot; S7 includes one failed response.
S6 retains its unqualified-diagnostic label; numerical caveats from navigation
reports remain in force. Latency measurement does not clear those caveats.

| Model | Episodes | Policy mean ms | Policy p50 ms | Policy p95 ms | Transport+policy mean ms | Environment mean ms |
|---|---:|---:|---:|---:|---:|---:|
| S1 BF16 | 1839 | 320.015 | 344.801 | 473.265 | 326.079 | 4.468 |
| S1 FP32 | 1839 | 462.190 | 464.996 | 625.497 | 468.296 | 4.039 |
| S2 | 1839 | 232.931 | 239.155 | 348.814 | 238.996 | 4.729 |
| S3 | 1839 | 269.068 | 270.759 | 350.516 | 275.152 | 5.266 |
| S4 | 1839 | 192.126 | 211.582 | 241.340 | 198.228 | 2.761 |
| S5 | 1839 | 315.708 | 372.503 | 464.984 | 321.877 | 3.616 |
| S7 | 1839 | 105.673 | 103.518 | 112.977 | 112.137 | 1.942 |
| S8N | 1839 | 138.683 | 128.976 | 181.724 | 144.841 | 2.387 |
| S8W | 1839 | 148.501 | 155.482 | 182.950 | 154.557 | 2.899 |
| M5_E1 R4/B1200 | 1839 | 205.266 | 205.578 | 251.449 | 211.405 | 4.619 |
| M5_E1 served B0 | 1839 | 205.066 | 202.414 | 402.558 | 211.262 | 3.285 |
| R4B0 trained | 1839 | 138.917 | 108.227 | 328.895 | 145.231 | 2.787 |
| S6 | 1839 | 405.998 | 394.332 | 510.200 | 412.894 | 7.581 |
| S9 | 1839 | 487.560 | 495.315 | 652.504 | 494.499 | 6.689 |
| A0 | 1839 | 167.260 | 144.225 | 478.962 | 173.447 | 4.395 |
| A1 | 1839 | 399.230 | 320.468 | 816.956 | 406.046 | 8.868 |
| M2 | 1839 | 239.606 | 233.874 | 270.001 | 245.634 | 5.434 |
| M2c | 1839 | 258.124 | 234.396 | 244.360 | 264.271 | 4.589 |
| V0 | 1839 | 224.699 | 229.614 | 238.736 | 230.789 | 4.350 |
| G1 | 1839 | 92.531 | 86.006 | 143.562 | 98.947 | 2.216 |
| G3_fixed01_seed430 | 1839 | 105.317 | 95.193 | 159.766 | 111.352 | 2.306 |
| G3_M5 | 135 | 105.842 | 103.811 | 113.646 | 112.258 | 2.002 |
| G3 | 1839 | 185.307 | 179.031 | 226.676 | 191.348 | 4.127 |
| M5 | 1839 | 212.189 | 162.349 | 287.632 | 218.156 | 4.931 |
| G3A1 | 1839 | 156.672 | 98.352 | 417.757 | 162.855 | 3.412 |
| M1 | 1839 | 248.736 | 117.320 | 825.126 | 254.924 | 5.343 |
| M5c | 1839 | 404.501 | 318.465 | 833.768 | 411.284 | 9.223 |
| R1 | 1839 | 168.009 | 108.364 | 295.403 | 174.314 | 2.312 |
| Window8-B-E1 | 1839 | 178.472 | 172.798 | 227.792 | 184.467 | 4.662 |
| G3 DINOv3 ViTB16 | 1839 | 118.793 | 93.236 | 349.857 | 125.598 | 2.570 |
| M5 E1 R2/B1800 seed431 | 1839 | 168.829 | 113.799 | 439.470 | 175.051 | 3.081 |
| M5 E1 R4/B1200 seed431 | 1839 | 124.796 | 104.785 | 174.872 | 131.269 | 2.239 |
| G3_M5 L8/L16/L24 TRIPLE | 1839 | 332.326 | 350.243 | 357.273 | 339.013 | 5.665 |

## Hardware, precision and provenance

Host `VMO-ANHDH35-D` (ws_anhdh35), AMD Ryzen Threadripper 9960X (24 cores / 48 threads);
NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97,887 MiB reported memory,
driver 580.178.04, 300 W power limit. Python 3.12.13, PyTorch 2.10.0+cu129,
Transformers 5.11.0, FlashAttention 2.8.3+cu.12.9.torch.2.10, Triton 3.6.0,
fla-core/flash-linear-attention 0.5.2, NumPy 2.4.4 and Pillow 12.2.0. Model environment
`/storage/anhdh35/SimpleMemVLN/.venv/bin/python`, base model
`/storage/anhdh35/SimpleMemVLN/artifacts/checkpoints/qwen-base`.
No Habitat process is used in the controlled benchmark. Driver, power limit, CPU model and affinity are saved in hardware.json. GPU telemetry includes
temperature, utilization, power, clocks, memory and foreign compute PIDs sampled
every two seconds. It detects sampled contention, not sub-second external events.
All 33 workers log the Transformers Qwen3.5 fast-path availability warning.
The optional `causal_conv1d` module is absent; `fla` and `flash_attn` are installed.
This warning is emitted when constructing the base model and does not by itself
identify every kernel used by the repository's custom streaming path. These
measurements describe the preserved installed serving environment, not a claimed
fully optimized deployment. No packages were installed or changed during timing.

Per-run GPU allocation includes model parameters and live caches; reserved memory
is separately retained in raw data. No allocator tuning was applied.

| Model | Precision | Frozen source commit | Checkpoint SHA256 | Raw evidence |
|---|---|---|---|---|
| S1 BF16 | bfloat16 | `fc017d9a5bbf1e80ae2c13151eb1aaaf38f0a7a6` | `dd5876f50647fac8bfeb09bb4c7f6c8590dc796aa96db96642ec4b7af4e228df` | result (`/data14t/anhdh35/latency_20261024/results/s1_bf16/result.json`) |
| S1 FP32 | float32 | `fc017d9a5bbf1e80ae2c13151eb1aaaf38f0a7a6` | `dd5876f50647fac8bfeb09bb4c7f6c8590dc796aa96db96642ec4b7af4e228df` | result (`/data14t/anhdh35/latency_20261024/results/s1_fp32/result.json`) |
| S2 | bfloat16 | `fc017d9a5bbf1e80ae2c13151eb1aaaf38f0a7a6` | `88dba8325a014868ca0633634ff0725dbb05b255e36b3d8fb9c78acefeab3044` | result (`/data14t/anhdh35/latency_20261024/results/s2/result.json`) |
| S3 | bfloat16 | `fc017d9a5bbf1e80ae2c13151eb1aaaf38f0a7a6` | `172b541a79a42d493c5acdb9c8c5478f8f540669577893cbf7163f94b0b0e9d1` | result (`/data14t/anhdh35/latency_20261024/results/s3/result.json`) |
| S4 | bfloat16 | `a2d075fca2a3a206f7b9bc45cb9d6cdf772f0aa7` | `38b8b5aa4aa1cb7682df9678db6923e9a45286b015f505e3e53f6facbe416bcc` | result (`/data14t/anhdh35/latency_20261024/results/s4/result.json`) |
| S5 | bfloat16 | `fc017d9a5bbf1e80ae2c13151eb1aaaf38f0a7a6` | `6d4c8e68101ec4dca67b6a8a771108ebdf2877a68fc5abd7dd10955162426f6b` | result (`/data14t/anhdh35/latency_20261024/results/s5/result.json`) |
| S7 | bfloat16 | `a2d075fca2a3a206f7b9bc45cb9d6cdf772f0aa7` | `b927f44f6920ecfeef183851c3135cc503d34ffac53ebfc67515968c292daaeb` | result (`/data14t/anhdh35/latency_20261024/results/s7/result.json`) |
| S8N | bfloat16 | `a2d075fca2a3a206f7b9bc45cb9d6cdf772f0aa7` | `c89de946f3cb05aa91a5cc498fd63991c901852ba0c05e6546cbaddf23446a2b` | result (`/data14t/anhdh35/latency_20261024/results/s8n/result.json`) |
| S8W | bfloat16 | `a2d075fca2a3a206f7b9bc45cb9d6cdf772f0aa7` | `113482a5ce90b4d6819fb2669f94cc18e91a293d6390daf0cc4c5a9a1fe8b755` | result (`/data14t/anhdh35/latency_20261024/results/s8w/result.json`) |
| M5_E1 R4/B1200 | bfloat16 | `fc017d9a5bbf1e80ae2c13151eb1aaaf38f0a7a6` | `92089fa16096f5fb581a7fb85a670b539a887278d7f66bacd20b4d0424d70ea6` | result (`/data14t/anhdh35/latency_20261024/results/m5_e1_r4_b1200/result.json`) |
| M5_E1 served B0 | bfloat16 | `fc017d9a5bbf1e80ae2c13151eb1aaaf38f0a7a6` | `92089fa16096f5fb581a7fb85a670b539a887278d7f66bacd20b4d0424d70ea6` | result (`/data14t/anhdh35/latency_20261024/results/m5_e1_served_b0/result.json`) |
| R4B0 trained | bfloat16 | `a2d075fca2a3a206f7b9bc45cb9d6cdf772f0aa7` | `505fec28668b1bc950c2993cefccc86c939e66c450e00583e8f9bb2c563fe360` | result (`/data14t/anhdh35/latency_20261024/results/r4b0_trained/result.json`) |
| S6 | bfloat16 | `a2d075fca2a3a206f7b9bc45cb9d6cdf772f0aa7` | `69d3f030a46b5ee4f05d75e7951ca1cee608fab0253ad788d5b571510e60a228` | result (`/data14t/anhdh35/latency_20261024/results/s6/result.json`) |
| S9 | bfloat16 | `a2d075fca2a3a206f7b9bc45cb9d6cdf772f0aa7` | `1b4602baed643b791332c4329fd18a37ed01142db0a3fd6bc927f7cfce15dfc1` | result (`/data14t/anhdh35/latency_20261024/results/s9/result.json`) |
| A0 | bfloat16 | `0090a07f72fabe6ac2d1d48c949b3ea9f09bdd9b` | `5e63563ddb891c43e6a02073125ddad8892f0a85c77a5b1d0b6637150b6f08e6` | result (`/data14t/anhdh35/latency_20261024/results/a0/result.json`) |
| A1 | bfloat16 | `4f772d3bac877322e2cb58e127c17f87e202f17c` | `0d9078ffd00e39d60655866912b849e0d9949407b9fa35424545134b00337bea` | result (`/data14t/anhdh35/latency_20261024/results/a1/result.json`) |
| M2 | bfloat16 | `frozen source hashes` | `2cfb26a6213f0a10f19908d1a6343f70a260eb9c00005e3c8a7cd1577ba43919` | result (`/data14t/anhdh35/latency_20261024/results/m2/result.json`) |
| M2c | bfloat16 | `frozen source hashes` | `cea5891796b1ab4b416ff240a66966e4ce6f2abe8d17bdd58942ff6ca55e87c8` | result (`/data14t/anhdh35/latency_20261024/results/m2c/result.json`) |
| V0 | bfloat16 | `frozen source hashes` | `65a7d8b2855510768339a8ef09a1f9ccc866e0e2625e5e6ff459f9131f05333a` | result (`/data14t/anhdh35/latency_20261024/results/v0/result.json`) |
| G1 | bfloat16 | `frozen source hashes` | `a620ab17b1c0cbec6f7c5f95d67177a4b078ae53e83ea08c6696c5c4797a63ae` | result (`/data14t/anhdh35/latency_20261024/results/g1/result.json`) |
| G3_fixed01_seed430 | bfloat16 | `d683574` | `5244107fd43fe4c278414d642763c57437acc4bdc6846965f6775fb42dafdbac` | result (`/data14t/anhdh35/latency_20261024/results/g3_fixed01_seed430/result.json`) |
| G3_M5 | bfloat16 | `295e85e6081be2a8d8d9d786f2ba47ca22e0667b` | `4e2d2095325fa779dcdece278f52213013d6889e72705ff860ab2eba1b13194f` | result (`/data14t/anhdh35/latency_20261024/results/g3_m5/result.json`) |
| G3 | bfloat16 | `9d658c245b2b9d14232e8db0b235bf92e857a43d` | `a71be6b5291b6acddbeb389318befaded5818c5b1438d5d44cf351c64248333b` | result (`/data14t/anhdh35/latency_20261024/results/g3/result.json`) |
| M5 | bfloat16 | `a340e131b63c9ab7846690acdb369599462076c4` | `66ddb5553793d7ae275a7f2531dc098f81e185e2426402e3709be07086280bad` | result (`/data14t/anhdh35/latency_20261024/results/m5/result.json`) |
| G3A1 | bfloat16 | `b5cd3b574fd6299aa866c8de5e4dfa10f8022dd2` | `2e53f3bb918501ceeace310b21f06e49b8a713c08801c9dde145385e0ee564a0` | result (`/data14t/anhdh35/latency_20261024/results/g3a1/result.json`) |
| M1 | bfloat16 | `a340e131b63c9ab7846690acdb369599462076c4` | `3c9064b310235f2e1762e4fcf92d62bb1663c38b4774e8a1a6827a2b3a501eb6` | result (`/data14t/anhdh35/latency_20261024/results/m1/result.json`) |
| M5c | bfloat16 | `a340e131b63c9ab7846690acdb369599462076c4` | `94f0f7aaaa991a4b99b367e6d547bb17a3d2755654c621ccfe207bef1ff895f5` | result (`/data14t/anhdh35/latency_20261024/results/m5c/result.json`) |
| R1 | bfloat16 | `dedb8f9cb8fafc6bf40f77cfbd0dc8e2b1f86450` | `2b25416a500d92d4c13925046551ff9fcfa77d5f57e6c7aea94fc3ac889cb347` | result (`/data14t/anhdh35/latency_20261024/results/r1/result.json`) |
| Window8-B-E1 | bfloat16 | `7cb2d3cd5fbae767f8e7f38a8c106fd2841e141f` | `e5c81d168326d59e6b48321fbd73da713b280a92dc2fcd0865e15aa8382f281e` | result (`/data14t/anhdh35/latency_20261024/results/window8_b_e1/result.json`) |
| G3 DINOv3 ViTB16 | bfloat16 | `9f26be634fb3342d66ff4c466c06613438567cfe` | `50179ba95adba9f4d03fab7e179865b0afbfc30a1c765b8e8d83b3043ba3ce57` | result (`/data14t/anhdh35/latency_20261024/results/g3_dinov3_vitb16/result.json`) |
| M5 E1 R2/B1800 seed431 | bfloat16 | `fde1cbef213e27320920400a826addea8d5e114a` | `5ed04b21391ce8a92c070cdd53ab4e3b0be56b30952302f7c999daadd766c76c` | result (`/data14t/anhdh35/latency_20261024/results/m5_e1_r2_b1800_seed431/result.json`) |
| M5 E1 R4/B1200 seed431 | bfloat16 | `fde1cbef213e27320920400a826addea8d5e114a` | `0b12f7328805949156200cbe0810f68400a8724d650181cf4ef9d9f3d9b6d333` | result (`/data14t/anhdh35/latency_20261024/results/m5_e1_r4_b1200_seed431/result.json`) |
| G3_M5 L8/L16/L24 TRIPLE | bfloat16 | `5bdf5eed274709f0349a9d6c1572b623a0a2f066` | `c206e32e4bbe884ae6ac5dbcd1a46ac7589d9218b7bc8f27a70e29a821b295ce` | result (`/data14t/anhdh35/latency_20261024/results/g3_m5_triple/result.json`) |

## Evidence checks

Independent verification at `2026-10-24T15:16:37.165083+00:00`: **33/33 configurations checked**, overall passed: **True**. The checker recalculates mean, quantiles and throughput from raw samples; validates step coverage, fixture identity, profile coverage, process exit and GPU contention; and confirms the protected navigation report is unchanged. [Verification evidence](evidence/LATENCY_20261024_VERIFICATION.json).

Fixture SHA256: `d7c080da5bd82c108d25bf1bfe78e6ba63e9e0d4d6fa96205a853fd90cb213f1`. Every input image hash is in
manifest.json (`/data14t/anhdh35/latency_20261024/manifest.json`). Raw per-action times, tokens, actions, warm-up
observations, memory, runtime flags, per-trial summaries and source hashes are
in each result; adjacent telemetry.json and exit.json record process health.

Reproduce in a fresh output directory using worker.py (`/data14t/anhdh35/latency_20261024/worker.py`),
run_all.py (`/data14t/anhdh35/latency_20261024/run_all.py`) and the pinned manifest. The controller skips existing
results to avoid accidental overwrites; make a new artifact root for a rerun.
The original navigation reports and frozen model sources were not edited.

To repeat on this machine after confirming the GPU is idle:

```bash
BENCH_SRC=/data14t/anhdh35/latency_20261024
BENCH_OUT=/data14t/anhdh35/latency_repeat_$(date +%Y%m%d_%H%M%S)
mkdir -p "$BENCH_OUT"
cp "$BENCH_SRC/manifest.json" "$BENCH_SRC/worker.py" "$BENCH_SRC/run_all.py" "$BENCH_OUT/"
/storage/anhdh35/SimpleMemVLN/.venv/bin/python -u "$BENCH_OUT/run_all.py"
```

The manifest contains absolute paths to the base model, frozen model sources,
checkpoints and image fixtures. Keep those artifacts available; moving to another
server requires remapping paths and matching the environment. The worker also
uses the preserved navigation evidence in this repository for integrity checks
and the saved S1 FP32 patch. The controller refuses to start a new worker when
another GPU process is detected; a nonzero worker exit must be investigated.
To rerun one configuration, use `worker.py MODEL_KEY` in a fresh benchmark root;
use the controller for contention monitoring and exit/telemetry evidence.


[Download CSV](evidence/LATENCY_20261024.csv) · [Compact results](evidence/LATENCY_20261024.json) · Hardware inventory (`/data14t/anhdh35/latency_20261024/hardware.json`).

[Navigation S-campaign report](S_CAMPAIGN_R2R_EVALUATION.md) ·
[Episodic report](EPISODIC_MEMORY_EVALUATION.md) · [VGGT report](VGGT_EVALUATION.md).
