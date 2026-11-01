# S8N stage 2 — DAgger with the STOP-label fix — R2R evaluation

Snapshot 2026-11-01T18:16:38Z. **Complete for all three stage-2 arms: 1,839/1,839 episodes each, zero execution errors, final verification passed.** The best arm is **S8N_DAGGER_SF_REMAINING: 1,171/1,839 successes, SR 63.675911%, SPL 56.654234%**. That is +11.26 SR and +7.97 SPL points over S8N epoch 1 (964/1,839, SR 52.419793%, SPL 48.679311%), and it is the highest completed SR in this repository's index.

Every number below is recomputed from the per-episode file [S8N_STAGE2_SF_R2R_20261101_EPISODES.csv](evidence/S8N_STAGE2_SF_R2R_20261101_EPISODES.csv). The script [verify_s8n_stage2_sf_20261101.py](evidence/verify_s8n_stage2_sf_20261101.py) reproduces the summary and every paired statistic and checks them against [S8N_STAGE2_SF_R2R_20261101.json](evidence/S8N_STAGE2_SF_R2R_20261101.json); it uses the standard library only. The JSON also holds each campaign's identity, qualification and verification records.

## Results

R2R val_unseen, 1,839 episodes, same episode IDs and order for every row.

| Model | Host | Successes | SR (%) | SPL (%) | NE (m) | OSR (%) | TL (m) | SR / OSR | No model STOP | Mean steps |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S8N epoch 1 (reference) | ws_anhdh35 | 964 | 52.419793 | 48.679311 | 5.71 | 58.02 | 9.40 | 0.903 | 130 | 91.88 |
| S8N_DAGGER_SF | ws_anhdh35_vinh | 1,117 | 60.739532 | 48.838608 | 4.75 | 73.19 | 15.12 | 0.830 | 73 | 118.75 |
| **S8N_DAGGER_SF_REMAINING** | ws_anhdh35_vinh | **1,171** | **63.675911** | **56.654234** | **4.23** | 70.64 | 11.22 | 0.901 | **50** | 91.29 |
| S8N_DAGGER_SF_FULL | ws_anhdh35 | 1,111 | 60.413268 | 51.854865 | 4.42 | 70.80 | 11.98 | 0.853 | 76 | 103.57 |

- NE is the final geodesic distance to the goal and TL the trajectory length.
- SR / OSR is the fraction of oracle successes that the model converts into a success, so it measures how well the model stops.
- "No model STOP" counts episodes that ended without the model emitting STOP.
- All three stage-2 runs had 0 invalid responses and a peak of 2,646 KV rows.

### Paired comparisons

These are paired over the same 1,839 episodes. The table uses these terms:
- **Δ:** the first model minus the second, in percentage points.
- **p:** an exact two-sided McNemar test on discordant successes.
- **CI:** a paired bootstrap interval from 10,000 episode resamples (seed 0, percentile method).
- **Adjusted:** Bonferroni correction for the six comparisons. Adjusted p = min(1, 6p), and the adjusted CI is the 99.17% interval.
- **Outcome pairs:** episodes where both models succeed, only the first succeeds, only the second succeeds, and neither succeeds.

| Comparison | Δ SR | 95% CI | Adjusted CI | p | Adjusted p | Δ SPL | 95% CI | Adjusted CI | Outcome pairs |
|---|---:|---|---|---:|---:|---:|---|---|---|
| SF vs S8N | +8.32 | [+5.44, +11.26] | [+4.57, +12.23] | 1.8e-8 | 1.1e-7 | +0.16 | [−2.40, +2.75] | [−3.31, +3.58] | 674 / 443 / 290 / 432 |
| SF_REMAINING vs S8N | +11.26 | [+8.54, +13.97] | [+7.61, +14.85] | 4.3e-16 | 2.6e-15 | +7.97 | [+5.50, +10.48] | [+4.66, +11.29] | 741 / 430 / 223 / 445 |
| SF_FULL vs S8N | +7.99 | [+5.22, +10.77] | [+4.35, +11.69] | 1.5e-8 | 8.7e-8 | +3.18 | [+0.70, +5.67] | [−0.11, +6.50] | 703 / 408 / 261 / 467 |
| SF_REMAINING vs SF | +2.94 | [+0.27, +5.60] | [−0.71, +6.47] | 0.033 | 0.198 | +7.82 | [+5.47, +10.15] | [+4.72, +10.91] | 835 / 336 / 282 / 386 |
| SF_FULL vs SF | −0.33 | [−2.99, +2.45] | [−3.92, +3.37] | 0.842 | 1.000 | +3.02 | [+0.70, +5.42] | [−0.08, +6.12] | 799 / 312 / 318 / 410 |
| SF_REMAINING vs SF_FULL | +3.26 | [+0.76, +5.76] | [−0.16, +6.63] | 0.012 | 0.070 | +4.80 | [+2.54, +7.05] | [+1.71, +7.78] | 867 / 304 / 244 / 424 |

### Interpretation

- **Success rate:** all three stage-2 arms raise SR over S8N, and each gain holds after the six-way correction.
- **DAgger-only (SF):**
  - SF reaches the goal area far more often: OSR rises 58.02 → 73.19.
  - It also walks much longer: TL rises 9.40 → 15.12 m and mean steps 91.9 → 118.7.
  - It converts fewer oracle successes into stops: SR / OSR falls 0.903 → 0.830.
  - So its SPL is unchanged against S8N (+0.16 points, CI includes 0).
- **Adding original demonstrations (SF_REMAINING):** adding back the original R2R/RxR15deg demonstrations of the instructions DAgger did not re-collect restores path efficiency without losing the SR gain.
  - SF_REMAINING is back to S8N's step count (91.3 vs 91.9 mean steps) and stop precision (0.901 vs 0.903). Its trajectories are only moderately longer than S8N's (11.22 vs 9.40 m).
  - It has the highest SR and SPL of the four models. Its SPL gain over both S8N and SF holds after correction.
- **SF_REMAINING vs SF on SR:** SF_REMAINING's SR advantage over SF (+2.94) is significant only before correction.
- **SF_REMAINING vs SF_FULL:**
  - SF_REMAINING is ahead by +3.26 SR (p = 0.012; adjusted p = 0.070) and +4.80 SPL (adjusted CI excludes 0).
  - SF_FULL differs from SF_REMAINING in two ways at once:
    - It also keeps the original demonstrations of the 14,195 instructions DAgger re-collected.
    - It trains 46% more updates (5,627 vs 3,852).
  - This evaluation cannot separate those two causes.
- **Limits:** each model was trained once (seed 429) and evaluated once on one GPU type. These are single-run results, not seed-averaged estimates.

## Models and training

All three arms start strictly weights-only from S8N epoch 1's final weights.
- **Parent:** `anhdao69/M5_S/S8N`, weights SHA-256 `c89de946f3cb05aa91a5cc498fd63991c901852ba0c05e6546cbaddf23446a2b`.
- **Optimizer and schedule:** fresh AdamW and a fresh schedule.
- **Config:** S8N's resolved config, unchanged except for each manifest's one-epoch schedule:
  - peak LR 5e-6, 3% warmup, cosine decay to 0.1;
  - global batch 8 (4 H100 × 1 episode × accumulation 2);
  - Adam 0.9/0.95, weight decay 0.01, clip 1;
  - seed 429;
  - ZeRO-2 with CPU optimizer offload.
- **Base architecture:** Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- **Training source:** commit `15cfca0dd1860708385d12a4babe2c36f1490cbc`, branch `s8n_stage2`. It is built on `c2a4b1b` (oracle-relabel targets) and `97cbd14`.
- **Navigation settings:** unchanged from S8N (R4/B1200, guard 4784).

| Model | Training data (one epoch) | Episodes | Steps | Updates / warmup | Epoch loss | Wall time | Hub revision | Weights SHA-256 |
|---|---|---:|---:|---:|---:|---:|---|---|
| S8N_DAGGER_SF | DAgger, oracle-relabeled, STOP-label fix | 14,195 | 1,607,249 | 1,775 / 54 | 0.17695 | 6.82 h | `dc25073d6f989c8709802f5924ffdb2dfe0f568d` | `d6f0dedd99aab2572cd7ad4388168cb835c875a584d221f3ad1d43fadeb4c047` |
| S8N_DAGGER_SF_REMAINING | SF data + 16,620 original episodes whose instruction DAgger did not re-collect | 30,815 | 2,875,045 | 3,852 / 116 | 0.14824 | 12.60 h | `7ef60965c4b35ad56a886026d347ddb08545e09d` | `e3ada114b9b8d9435c6d92f02304ac8232c3d3905599b3d6aa38312dec46bb5a` |
| S8N_DAGGER_SF_FULL | SF data + all 30,815 original S8N training episodes | 45,010 | 4,735,873 | 5,627 / 169 | 0.12996 | 20.43 h | `20ce0b057b343eff833760e980588e980c7f9509` | `a86d0d35df7b755ea88e524a20c1c2a11abb80b0451dddcb1f202347f850c0d3` |

- **Repository:** all three are published in [`anhdao69/stage2_S8N`](https://huggingface.co/anhdao69/stage2_S8N), each in its own folder.
- **Training data sources:**
  - The DAgger part is `anhdao69/S8N-dagger-data` revision `5410aae9`: 14,195 episodes (R2R 3,043 + English RxR15deg 11,152).
  - The original episodes are lines copied byte-identically from S8N's own R2R/RxR15deg manifest (SHA-256 `5deb425d…`).
  - SF_REMAINING adds R2R 7,776 + RxR15deg 8,844 of them. SF_FULL adds R2R 10,819 + RxR15deg 19,996.
- **Manifest SHA-256 values:**
  - SF: `f3e477b8b56000cfb6333a4bfc59ef9286e993b45a651dd32ae9dd1956bb75c8`
  - SF_REMAINING: `ffc860b3009a22232ba9169f49fbbe9ba7659f0674e5c4e3f18538139cdf32d7`
  - SF_FULL: `2e5913fab31248a8b43dcc977eced2f98ca251f189a8d8cee426bbc4a15bcdd2`
- **Training validation:** each training run passed its own validation.
  - Every manifest episode was trained.
  - The scheduler and optimizer were fresh, initialized from the parent hash.
  - There was no GPU or host OOM.
- **Losses:** the arms train on different data and targets, so their losses are not comparable with each other or with S8N's. Training loss is not navigation success.

**STOP-label fix.**
- **The problem:** during DAgger collection the expert overrode S8N's STOP proposals until the agent was 0.25 m from the goal. Both the executed and the oracle labels at those states therefore said "keep moving".
- **The fix:** it is applied offline to the oracle-relabeled rows. In each episode, the first step where S8N proposed STOP within 1.5 m geodesic distance of the goal becomes the final step, labeled STOP, and later steps are dropped.
  - This truncated 4,468 episodes (R2R 1,187 / RxR15deg 3,281) and dropped 25,513 steps.
  - STOP proposals farther than 1.5 m keep their oracle targets.
  - All other rows are byte-identical to the oracle rows.
- **Where it is recorded:** the method and the tau analysis are kept with the data, in `s8n-stage2-data-20261030/manifests/MANIFESTS_STOPFIX.json` on the training cluster.

## Evaluation protocol

The protocol is the same one the user's Codex run used for S8N_DAGGER_SF (`evaluation/s8n_dagger_sf_20261101` on `10.148.182.137`). It was reused file for file.
- **Campaign files:** the SF_REMAINING and SF_FULL campaigns were generated from Codex's eight campaign files with counted replacements only:
  - the arm folder, Hub revision and weights SHA-256;
  - on ws_anhdh35, the server's own Python, data and checkpoint paths.

  Each campaign's `campaign_setup.json` lists every replaced string, and the evidence JSON repeats them.
- **Evaluation source:** `97cbd14`. Its `src/` is blob-identical to the documented inference reference `11baeb1`. The training-only commit `15cfca0` does not touch the inference path.
- **Settings:**
  - R2R val_unseen, 1,839 episodes, fixed order, seed 42;
  - a 500-step cap, 0.25 m moves, 15-degree turns, a 3 m success radius;
  - RGB 480×640, S8N R4/B1200 with guard 4784.
- **Execution:** four disjoint contiguous panels ran on one RTX PRO 6000 Blackwell per host.

The following checks passed before the full run, for every arm.
- **Download:** a pinned Hub revision, and the weights SHA-256 checked against the published value. All 25 published files matched `SHA256SUMS`.
- **Simulator and model:** a simulator preflight, and all 724 model tensors finite.
- **Serving audit:** two forced-history replays of one real Habitat episode (64 and 128 steps).
  - Each must pass `torch.allclose(atol 0.75, rtol 0.03)` and RMS ≤ 0.15 against the offline forward pass.
  - Each must also pass a bitwise session reset and identical repeated greedy decisions.
  - The audit also requires a 520-step bounded stream under the KV guard. That stream peaked at 2,546 resident KV rows against guard 4,784, with 14.48 GiB allocated.

| Model | Replay | Max logit error | RMS | Forced-token argmax agreement | Mean bank overlap | Gate |
|---|---|---:|---:|---:|---:|---|
| S8N_DAGGER_SF | 64 steps | 0.3438 | 0.02293 | 100.000% (191 tokens) | 99.896% | PASS |
| S8N_DAGGER_SF | 128 steps | 0.6250 | 0.02322 | 100.000% (383 tokens) | 99.950% | PASS |
| S8N_DAGGER_SF_REMAINING | 64 steps | 0.6250 | 0.02511 | 99.476% (191 tokens) | 99.948% | PASS |
| S8N_DAGGER_SF_REMAINING | 128 steps | 0.5938 | 0.02548 | 99.739% (383 tokens) | 99.950% | PASS |
| S8N_DAGGER_SF_FULL | 64 steps | 0.5625 | 0.02901 | 99.476% (191 tokens) | 97.396% | PASS |
| S8N_DAGGER_SF_FULL | 128 steps | 0.5625 | 0.02838 | 99.739% (383 tokens) | 97.127% | PASS |

This serving audit is the stage-2 campaign's qualification. It is not the three-fixture native replay gate of the first-stage campaign, so [ALL_RESULTS.md](ALL_RESULTS.md) labels these rows "PASS (serving audit 2/2)" and not "3/3". Forced-action agreement is a protocol check, not free-greedy equivalence. The minimum bank overlap was 0.75 for every replay.

- **Smoke run:** a two-episode Habitat smoke (episodes 10 and 1006) completed with zero errors for every arm. SF and SF_REMAINING had 1/2 successes and SF_FULL had 0/2. That is a setup check, not a performance estimate.
- **Full-run checks:** after the full run, the four panels' actions on the smoke episodes matched the smoke run exactly.
- **Final verification (`finish_verify.py`):**
  - exact episode order, uniqueness and completeness;
  - shard rows equal to the merged journal;
  - summary recomputed from the journal;
  - STOP and success consistency.

**Cross-host equivalence.** Before SF_FULL ran on ws_anhdh35, the published S8N_DAGGER_SF model was re-run there and compared with Codex's Vinh run.
- The 139 fixture frames were pixel-identical, and the scenes and render checks were equal.
- The smoke episodes matched exactly, as did a 22-episode sample (two episodes in each of 11 scenes): actions, tokens, KV, trajectories and metrics.
- The serving audit also passed on ws_anhdh35. Its only difference was in the first, 64-step replay: RMS 0.023042 against Vinh's 0.022935, max error 0.25 against 0.34375, and a slightly different bank overlap. The 128-step replay was identical.
- The Habitat packages match (habitat-sim/lab 0.2.4, numpy 1.26.4). Python differs (3.9.25 on Vinh, 3.10.20 on ws), as do the hydra-core and omegaconf patch versions.
- The comparison is recorded in `CROSSHOST_VERIFIED.json` and summarized under `crosshost_check` in the evidence JSON.

## Run history

- **S8N_DAGGER_SF:** Codex ran it on ws_anhdh35_vinh. A host reboot interrupted it after 1,520 completed episodes. It resumed from the existing journals (`reboot_resume.json`), and the merged journal passed final verification at 05:53:41Z.
- **S8N_DAGGER_SF_REMAINING** (ws_anhdh35_vinh):
  - **Attempt 1:** passed the preflight, hash check, serving audit and smoke run. It then failed at 09:53Z with CUDA OOM while loading panels 00/01, because another user's training job on the shared GPU left 8.6 GiB free. No episodes were written.
  - **Resume:** after that job ended, `pipeline.py --resume-full` restarted only the full evaluation, at 12:57Z, reusing the attempt-1 qualification records. The attempt-1 log, status and failure note are kept.
  - **Completion:** it completed at 17:54Z (4.94 h), and final verification passed at 17:56:20Z.
- **S8N_DAGGER_SF_FULL** (ws_anhdh35):
  - **Download:** the first download failed with an SSL certificate error caused by the site web gateway. The failed log is kept, and after network authentication the pinned download verified.
  - **Freeing the GPU:** the RoboTwin evaluation on that GPU was stopped at the user's request when training finished. It was later resumed and completed separately.
  - **Run:** the full evaluation ran 10:45–15:35Z (4.83 h), and final verification passed at 15:39:17Z.
- **Not in this report:** S8N_DAGGER and S8N_DAGGER_ORACLE, the stage-2 DAgger arms without the STOP-label fix, are also published in `anhdao69/stage2_S8N`.

## Evidence

In this repository:
- [S8N_STAGE2_SF_R2R_20261101.json](evidence/S8N_STAGE2_SF_R2R_20261101.json) holds model identities, training records, each campaign's download, audit, smoke, shard validation and final verification, the cross-host check, and the computed statistics.
- [S8N_STAGE2_SF_R2R_20261101_EPISODES.csv](evidence/S8N_STAGE2_SF_R2R_20261101_EPISODES.csv) is the per-episode table for all four models: success, SPL, distance to goal, oracle success, trajectory length, no-STOP flag and steps.
- `python3 evaluation_reports/evidence/verify_s8n_stage2_sf_20261101.py` recomputes everything above and prints `"passed": true`. It takes a few minutes, because the bootstrap is pure Python.

On the evaluation servers, the full journals and records are:
- **S8N:** `/data14t/anhdh35/m5_s8n_epoch1/evaluation/results/full/S8N_s429/episodes.jsonl` on ws_anhdh35. SHA-256 `2249dacc332f3d31f3aaeb6c30b9d367439903474d5ee09b8163a84325a9b487`, the same journal as the S8N row of [ALL_RESULTS.md](ALL_RESULTS.md).
- **S8N_DAGGER_SF:** `/media/vmo-perception/disk_2/anhdh35/SimpleMemVLN/evaluation/s8n_dagger_sf_20261101` on `10.148.182.137`. Journal SHA-256 `7fb493b468dd…`.
- **S8N_DAGGER_SF_REMAINING:** `/media/vmo-perception/disk_2/anhdh35/SimpleMemVLN/evaluation/s8n_dagger_sf_remaining_20261101` on `10.148.182.137`. Journal SHA-256 `e4d672fc22ea2cdfd3735ee037e628c09253ef381035033a4b7ec4a34ac86088`. This campaign also holds `FINAL_RESULTS_4MODEL.json`.
- **S8N_DAGGER_SF_FULL:** `/data14t/anhdh35/s8n_stage2_sf_eval_20261101/S8N_DAGGER_SF_FULL` on `10.148.183.119`. Journal SHA-256 `d47b2b19a14e290ddeffe819ed6dbcd3c042b7554b3275655a5b3ac1034d917a`.
- **Four-model summary:** `/data14t/anhdh35/s8n_stage2_sf_eval_20261101/` on `10.148.183.119` holds `FINAL_RESULTS.json`, `compact/`, `compare.py` and `CROSSHOST_VERIFIED.json`.

Each campaign directory contains:
- `download_verified.json`, `assets_verified.json`, `simulator_preflight.json` and `source_identity.json`;
- `S8N/serving_audit.json` and `S8N/smoke/`;
- `S8N/full/{episodes.jsonl, summary.json, validation.json}`, `pipeline.log` and `FINAL_VERIFIED.json`.
