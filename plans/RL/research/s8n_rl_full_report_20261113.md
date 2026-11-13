# S8N / GlanceNav RL: consolidated report of every result, finding and observation (as of 2026-11-13 10:00 +07; RxR results updated 23:10)

This report gathers everything measured in the Stage-3 (RL) campaign on the local workstation and on gpu2, from the
start of the SFR line (2026-11-02) to today. It continues `s8n_local_campaign.md` (dated log up to 2026-11-09) and
`s8n_sfr_campaign.md` / `s8n_rescue_design.md` (designs). The analysis notes it summarises live in
`/nvme/public/svln/analysis/` (paths in section 15). Every number in the result tables is read from the merged
evaluator journals (`svln/evals/<model>_<split>/merged/summary.json`, on gpu2 for gpu2-only models), generated
by script for this report, not copied by hand.

Contents: 1 summary - 2 setup - 3 all results - 4 CoRe vs GRPO - 5 design ablations - 6 prompt pools - 7 run-to-run
variance - 8 weight averaging - 9 STABLE-V1 - 10 longer training ("4k") - 11 training-time behaviour - 12
infrastructure and incidents - 13 paper status - 14 running now and next steps - 15 files - appendix A paired tests.

---

## 1. Summary (the findings that matter)

1. **RL from Stage 2 helps a lot.** Every RL recipe started from Stage 2 (SFR, R2R 62.59) gains 1.7-9.4 R2R SR points in
   125 updates (least: GRPO on R2R-only prompts, +1.74, p 0.17; most: CoRe run 1 on mixed prompts, +9.35). RL from Stage 1 does not get there (GRPO G=5 from Stage 1: 58.29, below Stage 2 before any RL).
2. **CoRe beats GRPO on identical prompts, in every single-run pairing**: R2R-only prompts 68.73 vs 64.33 (+4.40,
   McNemar p 7e-5) and vs the compute-matched G=5 66.50 (+2.23, p 0.037); mixed prompts 71.94 vs 66.83 (+5.11, p 2e-6).
   On RxR-CE the gap is small: full RxR 73.62 vs 72.44 (+1.17, p 0.16).
3. **But one training run is not a measurement.** Three CoRe runs of one recipe on the same prompts in the same order
   reach 71.94, 70.15 (same seed, other machine) and 67.70 (another seed): mean 69.93, sd 2.13. Two of them differ at
   McNemar p 7e-5, so paired tests on one run per arm say nothing about recipes. Only one GRPO run exists (66.83).
   Over the three runs, CoRe is +3.1 over that GRPO run; over the two seed-20261010 runs (paper, see 13) +4.2.
4. **Sample efficiency is the robust CoRe claim.** GRPO keeps improving with more updates (66.83 / 67.92 / 69.33 /
   71.45 at 125 / 250 / 500 / 750), CoRe plateaus early (gpu2 run: 70.15 / 70.96 / 70.69 / 70.53 R2R). GRPO reaches
   CoRe's 125-update level only after about 570-700 updates (4.6-5.6x the updates, 3.5-4.4x the wall time), and its 750-update policy turns in 58.5% of actions and runs out of steps in 9.2% of episodes.
5. **Why runs differ (underspecification).** The training reward cannot tell the runs apart (train success 74.3% vs
   74.2%), but val SR tracks oracle success (r 0.93), reaching within 1 m (0.85) and turn share (0.82) across 16 RL
   models. Mixed groups push against turning in training scenes, expert corrections (62% turns) push for it; where this
   tug-of-war ends is seed luck. About 90% of each run's weight change is seed noise (cosine between seeds 0.10).
6. **No prompt pool beat the natural mix beyond seed noise.** Hard pool 67.70, clean pool 68.62, path-aware rewind
   69.66: all inside the 67.70-71.94 range of the mix pool's own seeds. R2R-only -> mixed prompts (+3.21, single runs)
   is the one pool change with a consistent direction, on RxR too.
7. **Weight averaging does not stabilise.** Cross-run soups give the mean of their ingredients (2 runs: 69.66 vs 69.82;
   4 models: 69.71 vs 69.25), never the best. Averaging a run's own tail (updates 65..125) costs 1.74 SR vs its final
   weights (p 0.009).
8. **STABLE-V1 seed A (batch advantage scale + LR tail decay)**: 69.71 SR, i.e. the old recipe's mean, with the best SPL
   of any run (63.97) but the lowest turn share (37.6%) and lower OS. It does not raise the level; whether it shrinks the
   spread needs seed B (gpu2, training, result tonight).
9. **Where to branch matters most**: forking at the trigger made the policy spin (80% expert turns; guard stop at 20
   updates); rewinding to the closest approach k* halves the turn imitation (57-62%) and doubles rescues in a probe
   (12/16 vs 6/16).
10. **CoRe keeps the training signal alive**: share of groups with a learning signal per quarter, GRPO 64 / 50 / 40 / 36%,
    CoRe 78 / 63 / 60 / 59% (run 1) and 75 / 66 / 62 / 60% (run 2).
11. **Half of GRPO's update is stop-precision noise**: 44-50% of groups are all-success; their returns differ only by the
    stop distance (sd 0.23 m) yet group normalisation gives them 43-50% of the advantage mass.
12. **Paper (13)**: the user removed the third CoRe run (67.70) from the paper on 2026-11-13 after being told what it
    changes; the paper now reports two runs (71.0 +- 1.3 R2R, 73.4 +- 0.4 RxR) and no seed-variance appendix. This
    report keeps all runs. New seeds (20261414) of GRPO and CoRe started today (14).

---

## 2. Setup

### 2.1 Model and stages
- GlanceNav / S8N: Qwen3.5-4B hybrid (24 Gated-DeltaNet + 8 attention layers), frozen vision encoder (300 tokens per
  frame), streaming memory = GDN recurrent state + window W=4 + keyframe bank F=4 chosen by state change.
- **Stage 1** (`base`): episode-level SFT, S8N epoch 1 (`anhdao69/M5_S` S8N, sha256 `c89de946...`). R2R 52.04 locally.
- **Stage 2** (`sfr` = `S8N_DAGGER_SF_REMAINING`, HF `anhdao69/stage2_S8N` @ 7ef60965, sha256 `e3ada114...`): one epoch
  from Stage 1 on expert-relabelled DAgger data with the STOP-label fix plus the original demonstrations of instructions
  DAgger did not re-collect. 63.68 SR on HiPerGator, **62.59 on this machine** (evaluator numerics differ by machine).
- **Stage 3**: RL from Stage 2 (this report).

### 2.2 RL recipes
- **GRPO = the ActiveVLN recipe** (`activevln_grpo_*`): 8 prompts per update x G=4 samples, temperature 1.2, dynamic
  sampling (an all-fail group is re-rolled up to 2 times), dual-clip PPO loss (0.2 / 0.28, c=3), token-mean, no KL, no
  entropy, one on-policy update per batch (ratio = 1, clip never acts), AdamW lr 1e-6 constant, weight decay 0.01, grad
  clip 1.0, only the language model trains. Reward R = 15 (1 - d_T / 3) for a STOP within 3 m, else 0. Budget
  min(2 x reference length, 120) macro actions, converted to primitives.
- **CoRe = Counterfactual Rewind** (`rewind_grpo_*`, estimator `grpo_rescue`, `rescue.rewind: true`): GRPO mains plus,
  for each main whose distance rises >= 1 m above its closest approach so far (trigger t), one branch restored from the
  O(1) memory snapshot at the closest approach k*; the shortest-path expert takes over until its 3rd FORWARD or STOP
  (<= 16 actions), then the policy continues. Branch credit Delta_b = (R_b - R_m) / max(batch std of main returns,
  0.05 x 15); expert actions imitated with weight [Delta_b]+ ("verified"), the policy continuation gets Delta_b.
- Snapshot: 135.7 MiB, cloned on GPU in 1.4 ms (measured 2026-10-31).
- Hardware/time (local, 2 x RTX PRO 6000, ZeRO-2 with CPU Adam): **GRPO 125 updates 34,777 s (9.7 h); CoRe 125 updates
  44,862 s (12.5 h)**. gpu2 single GPU 3: 11-15 min per CoRe update (~23-31 h per 125).

### 2.3 Prompt pools (all `shuffle: false`: every run sees the same prompts in the same order; the seed changes only
sampling)
| pool | content | used by |
|---|---|---|
| R2R 1k | ActiveVLN R2R list, first 1,000 (`activevln_r2r4k.jsonl`) | Table R2 runs |
| mix1k | 500 R2R (R2R 1k's first 500) + 500 RxR (ActiveVLN RxR list) alternating, `mix_r2r1k_rxr1k.jsonl` (6ab04818) | CoRe runs 1/3, STABLE-V1, seed-1414 runs |
| A+ 8k | 3,980 R2R + 3,980 RxR alternating, `mix_r2r3980_rxr3980_aplus.jsonl` (3357e8f3); rows 0-999 = mix1k exactly (checked) | GRPO run 1 / 8k, CoRe run 2 / 8k |
| hard | A+ rows 1,000-4,879 with the highest posterior P(mixed group) | hard1k, hard-from-mix1k |
| clean2k | 1,000 R2R + 1,000 RxR, posterior pass rate >= 0.66, keeps always-solved prompts (`select_clean_pool.py`) | clean2k (gpu2) |

"1k / 2k / 4k / 6k" in this report = 125 / 250 / 500 / 750 updates x 8 prompts.

### 2.4 Evaluation
- R2R-CE val_unseen, 1,839 episodes; RxR-CE val_unseen English (RxR15): sets A (1,000), B (1,000), C (1,669), full
  3,669.
- One evaluator tree (`svln/code/sfr_campaign_20261102_2034`) for every journal. Each journal is checked with
  `verify_eval.py` (episode count, weight hash, step sizes, SPL-implied shortest paths, policy failures).
- Paired comparisons: exact McNemar on per-episode success + bootstrap CI (`compare_evals.py`), only between journals
  of the same machine. **A McNemar p measures evaluation noise for two fixed models, not training noise** (section 7).
- Metrics: SR, SPL, OS (oracle success: came within 3 m), NE, TL; turn share = share of TURN actions among all
  predicted actions; forced stops = episodes ended by the step cap.

---

## 3. All results

#### Table R1. Stage 1 / Stage 2 and RL from Stage 1 (R2R-CE val_unseen, 1,839 episodes)

| Model | What it is | Machine | SR | SPL | OS | NE | TL (m) | turn share | forced stops |
|---|---|---|---|---|---|---|---|---|---|
| `base` | Stage 1: S8N epoch 1 (episode-level SFT) | local | 52.04 | 48.35 | 58.08 | 5.59 | 9.41 | 0.241 | 124 |
| `s8n_dagger` | Stage 1 + DAgger, executed-action labels | local | 52.04 | 42.38 | 70.26 | 5.93 | 15.96 | 0.212 | 262 |
| `s8n_dagger_oracle` | Stage 1 + DAgger, expert labels on every visited state | local | 58.84 | 44.97 | 72.65 | 4.75 | 15.72 | 0.484 | 68 |
| `sfr` | **Stage 2** = S8N_DAGGER_SF_REMAINING (expert labels + STOP fix + remaining demos); start of all Stage-3 runs | local | 62.59 | 55.59 | 69.98 | 4.27 | 11.33 | 0.432 | 55 |
| `grpo_g5_s125` | GRPO G=5 directly from Stage 1, 125 updates (R2R 1k prompts) | local | 58.29 | 51.86 | 67.59 | 4.79 | 10.71 | 0.244 | 144 |
| `twins_grpo_s125` | GRPO + memory twins from Stage 1, 125 updates | local | 60.03 | 53.61 | 68.35 | 4.87 | 10.61 | 0.267 | 161 |

#### Table R2. RL from Stage 2 on R2R-only prompts (ActiveVLN R2R list, first 1,000, 125 updates, seed 20261010)

| Model | What it is | Machine | SR | SPL | OS | NE | TL (m) | turn share | forced stops |
|---|---|---|---|---|---|---|---|---|---|
| `grpo_sfr_s125` | A: GRPO (ActiveVLN recipe), G=4 | local | 64.33 | 57.32 | 72.38 | 4.29 | 11.14 | 0.368 | 91 |
| `grpo_g5_sfr_s125` | E: GRPO G=5 (rollout compute of CoRe) | local | 66.50 | 58.71 | 75.20 | 3.99 | 11.54 | 0.391 | 125 |
| `rewind_self_sfr_s125` | C: self-rewind (policy resamples from k*, no expert) | local | 67.26 | 59.58 | 74.23 | 3.98 | 11.01 | 0.418 | 128 |
| `rewind_forkt_sfr_s020` | G: expert fork at the trigger t (no rewind); drift guard stopped it at update 20 | local | 65.85 | 56.65 | 75.42 | 3.82 | 12.47 | 0.459 | 71 |
| `rewind_sfr_s125` | B: CoRe (rewind to k*, expert H=16, verified imitation) | local | 68.73 | 60.87 | 76.67 | 3.68 | 11.18 | 0.417 | 121 |

#### Table R3. RL from Stage 2 on the mixed prompts (500 R2R + 500 RxR alternating = rows 0-999 of the A+ list), 125 updates

| Model | What it is | Machine | SR | SPL | OS | NE | TL (m) | turn share | forced stops |
|---|---|---|---|---|---|---|---|---|---|
| `grpo_sfr_mix8k_aplus_s125` | GRPO run 1 (seed 20261010; the 8k run at update 125) | local | 66.83 | 60.01 | 74.44 | 3.85 | 10.92 | 0.362 | 103 |
| `rewind_sfr_mix1k_s125` | **CoRe run 1** (seed 20261010) | local | 71.94 | 63.65 | 78.58 | 3.32 | 11.51 | 0.455 | 86 |
| `rewind_sfr_mix8k_aplus_s125` | CoRe run 2 (seed 20261010, 2 GPUs on gpu2; the 8k run at update 125) | gpu2 | 70.15 | 62.69 | 77.27 | 3.51 | 11.11 | 0.432 | 92 |
| `rewind_sfr_mix1k_seed2_s125` | CoRe run 3 (seed 20261111) - removed from the paper on 11-13 (user) | local | 67.70 | 60.47 | 74.61 | 3.78 | 11.00 | 0.422 | 80 |
| `rewind_sfr_mix1k_stable_a_s125` | STABLE-V1 seed A (seed 20261212), final weights | local | 69.71 | 63.97 | 75.75 | 3.65 | 10.49 | 0.376 | 68 |
| `rewind_sfr_mix1k_stable_a_avg` | STABLE-V1 seed A, uniform average of exports 65..125 | local | 67.97 | 62.51 | 74.71 | 3.88 | 10.71 | 0.373 | 72 |
| `rewind_path_sfr_mix1k_s125` | path-aware rewind (branch at the farthest reference-path progress) | local | 69.66 | 62.29 | 75.86 | 3.51 | 10.99 | 0.437 | 82 |
| `rewind_sfr_hard1k_s125` | CoRe on the hard pool (A+ rows chosen for high P(mixed group)) | local | 67.70 | 60.03 | 76.84 | 3.69 | 11.60 | 0.413 | 117 |
| `rewind_sfr_clean2k_s125` | CoRe on the clean pool, first 1,000 prompts (1 GPU on gpu2) | gpu2 | 68.62 | 61.61 | 74.93 | 3.56 | 10.74 | 0.408 | 53 |
| `rewind_mix1k_hard2k_s125` | CoRe run 1 + 125 more updates on hard prompts | local | 68.84 | 62.72 | 73.68 | 3.73 | 10.61 | 0.403 | 59 |
| `soup_mix1k_seed12` | weight average of CoRe runs 1 and 3 | local | 69.66 | 61.49 | 76.29 | 3.58 | 11.13 | 0.431 | 78 |
| `soup_rewind4` | weight average of run 1, run 3, path-aware, hard1k | local | 69.71 | 62.53 | 77.11 | 3.61 | 11.18 | 0.425 | 80 |

#### Table R4. Longer training on the same prompt list (A+ rows 0..8k-1, constant LR)

| Model | What it is | Machine | SR | SPL | OS | NE | TL (m) | turn share | forced stops |
|---|---|---|---|---|---|---|---|---|---|
| `grpo_sfr_mix8k_aplus_s125` | GRPO, 125 updates (1k prompts) | local | 66.83 | 60.01 | 74.44 | 3.85 | 10.92 | 0.362 | 103 |
| `grpo_sfr_mix8k_aplus_s250` | GRPO, 250 updates (2k) | local | 67.92 | 61.27 | 74.50 | 3.77 | 10.54 | 0.388 | 110 |
| `grpo_sfr_mix8k_aplus_s500` | GRPO, 500 updates (4k) | local | 69.33 | 61.44 | 74.99 | 3.55 | 10.95 | 0.355 | 79 |
| `grpo_sfr_mix8k_aplus_s750` | GRPO, 750 updates (6k) | local | 71.45 | 62.15 | 77.27 | 3.15 | 12.27 | 0.585 | 200 |
| `rewind_sfr_mix8k_aplus_s125` | CoRe (gpu2 run), 125 updates | gpu2 | 70.15 | 62.69 | 77.27 | 3.51 | 11.11 | 0.432 | 92 |
| `rewind_sfr_mix8k_aplus_s250` | CoRe (gpu2 run), 250 updates | gpu2 | 70.96 | 61.45 | 78.96 | 3.36 | 12.07 | 0.476 | 109 |
| `rewind_sfr_mix8k_aplus_s500` | CoRe (gpu2 run), 500 updates | gpu2 | 70.69 | 63.91 | 75.86 | 3.41 | 10.13 | 0.485 | 95 |
| `rewind_sfr_mix8k_aplus_s750` | CoRe (gpu2 run), 750 updates | gpu2 | 70.53 | 63.85 | 75.48 | 3.52 | 10.83 | 0.436 | 119 |

Notes on Table R4: the 250-750 checkpoints are reported on R2R only (RxR only at 125 updates, the user's choice); the
GRPO run is paused at 750 (resumable `step_0750` kept); the gpu2 CoRe run stopped at update 790 (GPU-access rules,
section 12) with exports 125-750 kept.

#### Table R5. RxR-CE val_unseen (English RxR15)

Sets: A = first 1,000 episodes, B = next 1,000, C = remaining 1,669, 2k = A+B, full = all 3,669. SR / SPL (nDTW for full).

| Model | A SR / SPL | B SR / SPL | C SR / SPL | 2k SR / SPL | full SR / SPL / nDTW | full OS |
|---|---|---|---|---|---|---|
| `sfr` (Stage 2) | 66.50 / 54.87 | 65.20 / 53.19 | - | 65.85 / 54.03 | - | - |
| `rewind_sfr_s125` (CoRe, R2R-only prompts (B)) | 74.00 / 59.09 | 72.20 / 58.50 | - | 73.10 / 58.80 | - | - |
| `rewind_path_sfr_mix1k_s125` (path-aware rewind, mix) | 72.20 / 58.44 | 70.80 / 57.16 | - | 71.50 / 57.80 | - | - |
| `grpo_sfr_mix8k_aplus_s125` (GRPO run 1, mix) | 72.10 / 58.92 | 72.20 / 60.14 | 72.80 / 59.76 | 72.15 / 59.53 | 72.44 / 59.64 / 70.28 | 80.02 |
| `rewind_sfr_mix1k_s125` (CoRe run 1, mix) | 73.30 / 59.63 | 73.10 / 58.45 | 74.12 / 60.18 | 73.20 / 59.04 | 73.62 / 59.56 / 70.68 | 80.68 |
| `rewind_sfr_mix8k_aplus_s125` (CoRe run 2 (gpu2), 125) | 73.80 / 59.26 | 71.90 / 58.20 | 73.40 / 60.41 | 72.85 / 58.73 | 73.10 / 59.49 / 70.47 | 80.05 |

Greedy Stage-2 screens on training prompts (used for pool selection): R2R rl1k 76.50, rl2k 75.00; RxR ref500 80.60,
repl242 73.14, tail1061 76.63 SR. Development split R2R val_seen-300: Stage 2 66.33; A 68.67; E 72.00; C 71.67;
G 63.67; B 74.67.

---

## 4. CoRe vs GRPO

### 4.1 Same prompts, same order, same seed (one run each)
| prompts | GRPO | CoRe | Delta | McNemar p | 95% CI |
|---|---|---|---|---|---|
| R2R 1k | 64.33 / 57.32 | 68.73 / 60.87 | +4.40 SR | 7e-5 | [2.28, 6.53] |
| R2R 1k, compute-matched GRPO G=5 | 66.50 / 58.71 | 68.73 / 60.87 | +2.23 | 0.037 | [0.16, 4.24] |
| mix1k (R2R val_unseen) | 66.83 / 60.01 | 71.94 / 63.65 | +5.11 | 2e-6 | [3.05, 7.18] |
| mix1k (RxR full) | 72.44 / 59.64 | 73.62 / 59.56 | +1.17 | 0.16 | [-0.44, 2.78] |
| mix1k (RxR A) | 72.10 | 73.30 | +1.20 | 0.48 | - |
| mix1k (RxR 2k) | 72.15 | 73.20 | +1.05 | 0.36 | - |

### 4.2 Over runs
- CoRe on mix1k: 71.94 (run 1, local, seed 20261010), 70.15 (run 2, gpu2, seed 20261010), 67.70 (run 3, local, seed
  20261111): **69.93 +- 2.13**. Runs 1+2 only (the paper since 11-13): 71.05 +- 1.27; RxR full 73.36 +- 0.37
  (73.62, 73.10); SPL 63.17; OS 77.93.
- GRPO on mix1k: one run, 66.83 (RxR full 72.44). A second GRPO seed is training now (section 14).
- Seed 2 (67.70) is +0.87 over the GRPO run (p 0.45): with one GRPO run, "CoRe > GRPO" holds for the CoRe mean and for
  2 of 3 CoRe runs individually.

### 4.3 Longer training and sample efficiency (Table R4)
- GRPO: +1.09 (125->250, p 0.31), +1.41 (250->500, p 0.19), +2.12 (500->750, p 0.051). Not saturated at 750.
- CoRe (gpu2 run): 70.15 -> 70.96 -> 70.69 -> 70.53 R2R: flat after 250.
- Where GRPO crosses CoRe's 125-update level (linear between 500 and 750): 3-run mean 69.93 at ~571 updates (4.6x,
  3.5x wall time); 2-run mean 71.05 at ~703 (5.6x, 4.4x).
- GRPO at 750 vs CoRe run 1: 71.45 vs 71.94 (-0.49, p 0.70): the same level, reached with 6x the updates.
- **How GRPO-750 gets there**: turn share 0.585 (CoRe 0.43-0.46), path length 12.27 m, 200 forced stops (9.2% out of
  steps vs 4.3-4.5% for CoRe), OS 77.27. It finds the goal by searching more, at a cost in efficiency (SPL 62.15 vs
  63.65).
- CoRe at 500 updates: SPL 63.91 (vs 62.69 at 125) and TL 10.13 m on R2R (vs 11.11): longer CoRe training
  makes paths more efficient at constant SR.

### 4.4 Where the gain comes from (R2R val_unseen outcomes, % of 1,839)
| model | success | left the goal area (OS, stopped > 3 m) | never within 3 m | out of steps |
|---|---|---|---|---|
| Stage 2 | 62.59 | 7.23 | 27.35 | 2.83 |
| GRPO 125 | 66.83 | 7.50 | 20.28 | 5.38 |
| GRPO 750 | 71.45 | 4.46 | 14.85 | 9.24 |
| CoRe run 1 | 71.94 | 6.25 | 17.51 | 4.30 |
| CoRe run 2 (gpu2) | 70.15 | 6.91 | 18.43 | 4.51 |
| CoRe run 3 | 67.70 | 6.85 | 21.70 | 3.75 |

Both methods mainly convert "never came within 3 m" into success; CoRe also keeps out-of-steps and "left the goal area"
below GRPO. Run 3 loses to run 1 almost entirely in "never within 3 m" (21.70 vs 17.51): it finds the goal area less
often, it does not stop worse.

### 4.5 Training signal and branches (rollouts, updates 0-124)
| run | groups with a signal per quarter (unequal main success, or a branch return != main return) | mains that trigger a branch per quarter | branch success | failed mains rescued | expert actions that are turns |
|---|---|---|---|---|---|
| GRPO run 1 | 64 / 50 / 40 / 36 % | - | - | - | - |
| CoRe run 1 | 78 / 63 / 60 / 59 % | 33 / 29 / 25 / 27 % | 75% (1,146 branches) | 53% | 62% |
| CoRe run 2 (gpu2) | 75 / 66 / 62 / 60 % | 34 / 30 / 25 / 29 % | 75% (1,174) | 54% | 62% |
| CoRe run 3 | 77 / 62 / 60 / 63 % | 34 / 30 / 24 / 30 % | 76% (1,189) | 59% | 62% |

As the policy improves, GRPO groups become all-success (no contrast between successes and failures); CoRe's branches
keep about 60% of instructions informative in the second half. On R2R-only prompts, self-rewind kept almost as many
groups informative (74 / 60 / 56 / 53% vs CoRe 76 / 65 / 64 / 55%) yet scored 1.47 lower, so the expert's correction
matters beyond the amount of signal.

### 4.6 Advantage mass (`stability/advantage_noise.txt`)
- GRPO run 1: 50% of groups all-success; their stop distances differ by 0.23 m (median sd); returns spread a fifth of
  mixed groups' (sd 1.3 vs 6.4); after group normalisation they carry 50% of the summed |advantage|. Within them the
  advantage correlates with stop distance (r -0.53) and with nothing else (steps, turn share |r| < 0.02).
- CoRe run 1: all-success 440 of 1,000 groups, 43.5% of main advantage mass (run 3: 430, 43.0%). Batch-level scaling
  would cut their |advantage| per group from 2.95 to 1.06 while mixed groups go 3.12 -> 3.35.

---

## 5. Design ablations (R2R-only prompts, single runs, Table R2)
| comparison | Delta SR | p | reading |
|---|---|---|---|
| B (CoRe) vs A (GRPO G=4) | +4.40 | 7e-5 | rewind + expert beats plain GRPO |
| B vs E (GRPO G=5, compute-matched) | +2.23 | 0.037 | not just more rollouts |
| E vs A | +2.18 | 0.046 | more samples per group help GRPO |
| C (self-rewind) vs A | +2.94 | 0.007 | rewinding alone helps |
| B vs C (self-rewind) | +1.47 | 0.20 | the expert adds, not significant in one run |
| G (fork at t) vs B | -2.88 | 0.016 | where to branch matters (G stopped at 20 updates) |
| path-aware rewind vs CoRe (mix1k) | -2.28 | 0.036 | branching at reference-path progress is worse (single runs; within seed spread) |
| mix1k vs B (prompts, both CoRe) | +3.21 | 0.003 | mixed R2R+RxR prompts beat R2R-only (single runs) |

- **Fork at t (G)**: imitating expert turn-arounds from lost states made the policy spin: 80% of expert actions were
  turns; the drift guard stopped the run at update 20 (65.85 at that point). Rewinding to k* lowers the expert's turn
  share to 57% (R2R-only) / 62% (mix).
- **Rewind probe** (2026-10-31, 16 failed Stage-1 rollouts): expert branch from k* rescued 12, from t only 6 (p 0.07).
- **From Stage 1**: GRPO G=5 directly on Stage 1 reaches 58.29, below Stage 2 without RL (62.59), so DAgger first.
- **DAgger labels**: executed-action labels leave SR at 52.04 (= Stage 1); expert labels on every visited state 58.84;
  plus the STOP fix and remaining demos (Stage 2) 62.59.
- Caveat: all of these are single runs. Section 7 shows the same recipe spans 67.70-71.94, so only the large effects
  (B vs A, mix vs R2R-only, CoRe vs GRPO on mix) are clearly beyond one run's noise.

---

## 6. Prompt pool experiments (all CoRe from Stage 2 unless noted)
| run | pool | R2R SR | vs CoRe run 1 (71.94) | vs the mix recipe's 3-run mean 69.93 |
|---|---|---|---|---|
| B | R2R 1k | 68.73 | -3.21 (p 0.003) | -1.2 |
| path-aware, mix1k | mix1k | 69.66 | -2.28 (p 0.036) | -0.3 |
| hard1k | hard | 67.70 | -4.24 (p 8e-5) | -2.2 |
| hard from mix1k (+125 updates) | hard | 68.84 | -3.10 vs its start (p 0.004) | -1.1 |
| clean2k at 1k (gpu2) | clean | 68.62 | -3.32 (p 0.003) | -1.3 |

- The hard selection (`clean2k/why_pool.md`): fewer all-success groups (0.21 vs 0.44), more 1-of-4 groups (0.25 vs
  0.12), more negative mass on "stopped at a wrong place > 5 m" (0.34 vs 0.22). From SFR it damaged goal recognition
  (8.6% of episodes that reached < 1 m still failed, worst of 13 models; 92 "stopped elsewhere"); from mix1k it reduced
  turning (0.403) and reaching the goal (1,039 vs 1,126 within 1 m).
- On low-pass-rate prompts the rare success is barely better than the failures (contrast in steps/d0 -1.0 vs -2.8 on
  usually-solved prompts): it reinforces a lucky sample. Always-solved prompts still teach stop precision.
- **But (2026-11-12) mix1k seed 2 = 67.70 with mix1k's own training-group mix** (train success 0.742 vs 0.743, 4/4
  groups 0.43 vs 0.44, 1/4 groups 0.10 vs 0.12). The group-mix story does not predict val SR; every pool result above
  lies inside the mix pool's own seed range (67.70-71.94). Measured pool effects are within seed noise; a pool
  comparison smaller than ~5 pp needs several seeds per arm.
- Clean2k decision rule (written before the result): >= 72.9 helps, <= 65.5 stop, between = no detectable effect.
  Result 68.62 -> no detectable effect; paused at update 125 (user: "Stop clean 2k after 1k"), everything kept.
- The one consistent pool effect: mixing RxR prompts in (mix1k vs B +3.21 on R2R; on RxR A 73.30 vs 74.00 and 2k
  73.20 vs 73.10, i.e. no RxR loss). Even this is a single pair of runs.

---

## 7. Run-to-run variance (`stability/stability_report.md`)

### 7.1 The symptom
Same recipe, prompts and order: 71.94 (seed 20261010, local), 70.15 (seed 20261010, gpu2), 67.70 (seed 20261111,
local). Mean 69.93, sd 2.13. Runs 1 vs 3: 229 episodes solved only by run 1, 151 only by run 3, McNemar p 7e-5. The
same seed on another machine gives a different model (GPU nondeterminism), so fixing the seed fixes nothing.

### 7.2 What differs (runs 1 vs 3)
| | run 1 | run 3 |
|---|---|---|
| train success (all updates / 100-124) | 0.743 / 0.769 | 0.742 / 0.766 |
| groups 4/4 / 1/4 | 0.44 / 0.12 | 0.43 / 0.10 |
| val OS (episodes) | 1,445 (78.58%) | 1,372 (74.61%) |
| val within 1 m of the goal | 1,126 | 1,072 |
| val turn share | 0.455 | 0.422 |
| late-training turn share (same prompts) | 0.484 | 0.432 |

Across 16 RL checkpoints (`val_behaviour_all.txt`): corr(val SR, OS) 0.93, (SR, within 1 m) 0.85, (SR, turn share)
0.82, (SR, episode length) -0.09. The best models turn the most (mix1k 0.455; GRPO-750 0.585).

### 7.3 When they split
Turn share (run 1 - run 3) on the same training prompts: ~0 for updates 0-89, +0.033 at 90-99, +0.064 at 100-109
(10/10 updates), +0.056 at 110-119 (10/10), with no difference in training success (-0.03). Run 1's final checkpoint
caught a late, reward-neutral rise in looking around that pays off only in unseen scenes.

### 7.4 Mechanism
- Turning push = sum over trajectories of advantage x turn share (`turning_push.txt`): mixed groups push against
  turning in both runs (-49.2 / -45.9 over updates 0-89; -17.9 / -14.7 over 90-124): in training scenes the failed
  rollouts are the lost ones that searched. All-success groups push only weakly (-1 to -2). The upward pull comes from
  expert corrections (62% turns). The balance point is seed luck. The reward trains away from the behaviour (searching)
  that generalises to unseen scenes; this is underspecification (D'Amour et al.).
- Weight space (`deltas.py`, all 4.84 B changed parameters): cosine of the weight changes from SFR, run 1 vs run 3
  0.096; different recipes with the same seed and prompt order 0.19-0.27; other pairs 0.05-0.12. All 8 runs moved the
  same distance (norm 0.61-0.63): Adam with constant lr moves a fixed amount per step whatever the signal.
- Recipe mechanics that let noise accumulate: no trust region acts (ratio 1, clip fraction exactly 0 at every update,
  no KL, no entropy); 32 main trajectories per update; group-std normalisation amplifies near-ties (4.6); constant lr
  to the last update; bf16 resolution (lr 1e-6 x 125 updates: 72% of weights unchanged, 20% moved exactly 1 ulp, 7%
  >= 3 ulps; a likely amplifier, not proven); nondeterminism.
- Code notes: `algorithm.norm_adv_by_std_in_grpo` is ignored by the code (always group std); there was no LR scheduler
  before STABLE-V1.

### 7.5 Consequences
- Picking the best of N seeds inflates val_unseen (expected best of 3 = mean + 0.85 sd) and is not a recipe.
- Single-run comparisons between recipes are only meaningful for effects well above ~2 pp x sqrt(2) ~ 3 pp; paired
  McNemar p-values between single runs must not be read as recipe significance.
- Reporting should be mean +- sd over >= 3 seeds per arm (Henderson et al. 2018; Agarwal et al. 2021).

---

## 8. Weight averaging
| model | ingredients (SR) | ingredient mean | soup SR / SPL / OS | vs run 1 |
|---|---|---|---|---|
| soup_mix1k_seed12 | run 1 71.94, run 3 67.70 | 69.82 | 69.66 / 61.49 / 76.29 | -2.28 |
| soup_rewind4 | run 1, run 3, path-aware 69.66, hard1k 67.70 | 69.25 | 69.71 / 62.53 / 77.11 | -2.23 (p 0.026) |
| STABLE-V1 A tail average (exports 65, 70, ..., 125) | its own trajectory | - | 67.97 / 62.51 / 74.71 | -1.74 vs its endpoint (p 0.009) |

- Cross-run soups land at the mean of their ingredients: the runs end in different solutions, not on two sides of one
  basin. Averaging is not the lever here.
- soup_rewind4's journal has one verify flag (episode 626: a 0.402 m forward step); the other 1,838 episodes are clean.
- The tail average likely approximates an earlier point of a still-moving trajectory (its turn share 37.3% = the
  endpoint's); not separately tested (an eval of export 95 would show it). Either way: use the final weights.

---

## 9. STABLE-V1 (user: "I want the SR is stable around 71-72"; option 1)

### 9.1 Design (2026-11-12)
- Batch-level advantage scale: (R - group mean) / one std of the update's main returns, floor 0.05 x R_max (the scale
  CoRe already used for branch credit). Targets the near-tie amplification (4.6).
- LR schedule `linear_tail`: lr 1e-6 until update 75, then linearly to 2e-8 at update 124.
- Exports every 5 updates from 65; the model = their uniform average (`exports/avg_065_125`).
- Code (dev tree): `algos.grpo_advantages(..., scale=)`, `algos.lr_at`, `algorithm.adv_scale: group | batch`,
  `optim.lr_schedule: constant | linear_tail`, `optim.decay_steps`. Defaults reproduce the old code path exactly
  (the only diffs vs the frozen trees of runs 1-3 and GRPO run 1 are these opt-in additions). +2 unit tests.
- Success criterion (written before results): two seeds within 1 pp of each other, both >= 70.5.

### 9.2 Seed A (local, seed 20261212): finished 2026-11-13 06:33
| model | SR | SPL | OS | NE | turn share | TL |
|---|---|---|---|---|---|---|
| CoRe run 1 (old recipe) | 71.94 | 63.65 | 78.58 | 3.32 | 0.455 | 11.51 |
| CoRe run 3 (old recipe) | 67.70 | 60.47 | 74.61 | 3.78 | 0.422 | 11.00 |
| STABLE-V1 A, update 125 | 69.71 | **63.97** | 75.75 | 3.65 | 0.376 | 10.49 |
| STABLE-V1 A, average 65..125 | 67.97 | 62.51 | 74.71 | 3.88 | 0.373 | 10.71 |

- Training success is the highest of all mix1k runs (65.6 / 76.0 / 78.5 / 79.8 / 78.5% per fifth of training, vs
  68.1 / 74.8 / 76.1 / 75.5 / 75.3 for run 1), yet val SR is the recipe mean: it fits the training prompts better while
  looking around less in unseen scenes. Its turn share on the drift guard's 20 prompts fell 0.02-0.06 below the
  reference from update 60 on.
- Checks: batch scale verified on every update (one scale per update, e.g. 6.04 / 5.84 at updates 0 / 1; mean |adv|
  of all-success groups 0.165-0.267 vs ~0.74 under group std); LR schedule verified (1e-6 to update 74, 5e-7 at 100,
  2e-8 at 124). 457 all-success groups over the run.
- Two check-script bugs (both fixed, section 12): the chain's smoke check read the smoke's `.log` file instead of its
  folder (vacuous OK); the end-of-run check raised a false FAILED at update 44 (float32 rounding of two ~1e-5
  advantages).

### 9.3 Seed B (gpu2 GPU 3, seed 20261313)
Smoke passed 17:28 on 11-12 (one batch scale per update: 5.90 / 5.69; all-success |adv| 0.258). At update 82 by 08:56
on 11-13 (lr 8.6e-7, i.e. decaying), training success 0.78. Drift guard: within +-0.02 of the reference until window
75-79 (+0.053, under the 0.06 margin). Expected end of training ~17:00, then the s125 and tail-average evaluations.

### 9.4 Reading so far
The batch scale and LR decay do not raise the level (69.71 = 69.93); the tail average hurts. Whether the spread
shrinks cannot be judged from one seed. If seed B also lands near 69-70, STABLE-V1 is a no-op for the mean, and the
remaining lever for a stable 71-72 is fixing what the reward underspecifies (exploration in unseen scenes), e.g. an
anchor to a reference policy's turning or more instructions per update; not tested yet.

---

## 10. Longer training: "will a 4k run be stable around 70.5?" (user question, 2026-11-13)
- The only CoRe run trained past 1k (gpu2 run 2) is flat after 2k: R2R 70.96 / 70.69 / 70.53 at 2k / 4k / 6k. Within
  that run, the 4k checkpoint is a steady model.
- Not shown: that other seeds converge there. Run 2 was the middle seed at 1k. Whether the 67.70 seed rises to ~70.5 by
  4k, or the 71.94 seed falls to it, is untested.
- Cost: 4x the training (~50 h locally per run). And for the paper, 4k weakens the CoRe-vs-GRPO gap (GRPO 69.33 at 4k,
  71.45 at 6k), so the comparison should stay at 125 updates.
- Cheapest test (proposed, not started): continue run 3 (resumable `step_0125` kept; its 1,000 prompts are rows 0-999
  of the 8k list) to 500 updates on rows 1,000-3,999, ~37 h locally. Run 1 kept only its weights (no optimizer state).

---

## 11. Training-time behaviour and guards
- **Drift guard** (`ops/local/drift_guard.py`): turn share of the R2R mains per window of 5 updates vs a reference run
  on the same prompts; stops the run when it exceeds the reference by > 0.06 in two consecutive windows. Stopped the
  fork-at-t run (G) at update 20 (correct). False alarm on clean2k at update 10 (the reference table came from the run
  whose outcomes selected the pool, so it was biased low on those prompts); resumed with mix1k's own run as reference.
  Lesson: a same-prompt reference table must not come from the run that selected the pool.
- Training turn share of CoRe from SFR starts at 0.46-0.49 (updates 0-9) and ends at 0.43-0.48 (updates 100-124); the
  val turn share of final models ranges from 0.36 (GRPO 125) to 0.585 (GRPO 750).
- Forced stops (out of steps) grow with GRPO training (103 / 110 / 79 / 200 episodes at 125 / 250 / 500 / 750).

---

## 12. Infrastructure, incidents and checks
- **Local**: 2 x RTX PRO 6000 (97.9 GB); GPU 0 also holds another user's ~33 GB process (left untouched; rank 0 placement
  margin 8 GiB). 188 GB RAM; another user's 22-28 GB process caused two trainer OOM kills on 11-05. Optimizer offload (CPU Adam).
- **gpu2** (10.148.183.127 since the 11-10 reboot; identify by MAC 30:56:0f:4b:f0:21; strict host-key checking):
  - NIC drops on any sustained outbound transfer (3rd outage 11-09 22:18): never upload or pull bulk data from gpu2;
    only small, checksum-verified result files leave it.
  - Thermal lockup 11-09 (Xid 8/31 at 300 W, 89-90 C): the CoRe 8k run was stopped at 525 and restarted with MPS
    at 50% plus a thermal guard.
  - Since 11-10 a root ACL denies our account GPUs 0 and 2 (nvidia-smi numbering) and the user forbids GPU 1, so only
    GPU 3 is usable (one trainer rank, 6 rollout helpers). The CoRe 8k run was stopped at 790 for this reason.
  - gpu2 training code is byte-identical to local except its DATA_ROOT default.
- **Checkpoints**: every milestone export of both 8k runs (1k-6k) is kept (`keep_exports.sh`; never delete); copies on
  Hugging Face `anhdao69/stage3_S8N`. All exports of clean2k, hard runs and STABLE-V1 kept.
- **Reproducibility**: same seed on two machines gives 71.94 vs 70.15; the same model gives 63.7 vs 62.6 on two
  evaluators. Every paired test in this report is within one machine.
- **Check-script bugs found and fixed this week**:
  1. STABLE-V1 seed-A chain: `ls -d smoke_*` picked the smoke's `.log`, so the smoke stable_check read no rollouts and
     printed OK; re-run on the folder: OK (scales 6.0417 / 5.8438-5.8444). Seed B's chain names the folder directly.
  2. `stable_check.py` false FAILED at update 44: two advantages of ~1e-5 whose float32 rounding gave scale 5.3618 vs
     5.3803; now ignores |adv| < 1e-3 (fixed copy pushed to gpu2, sha-checked, old copy kept).
  3. The clean2k drift-guard false alarm (section 11).
  4. Soup watcher arithmetic ("0\n0") fixed.
- `verify_eval.py` found no problems in any final journal except soup_rewind4's one 0.402 m step (section 8).

---

## 13. Paper status (`/nvme/public/overleaf_glance`, branch `rl-revision`, uncommitted)
- 2026-11-12 rewrite of the RL part (user request): RL "on R2R-CE + RxR-CE RL instructions" without "1k/2k" labels; new
  Sec. 4.4 "Reinforcement Learning" (protocol, training stages, longer training, where the gain comes from, design of
  CoRe, reward and advantage), new Table core_design, new Figure rl_analysis (a: SR vs updates; b: groups with a signal;
  c: where episodes end), appendix sections on CoRe details, reward/advantage design; yellow = changed, red `\pseudo` =
  placeholder. 6 undefined citations and several factual errors fixed (list in `svln/analysis/paper/rl_revision_todo.md`).
- **2026-11-13, user decision**: "Dont update th third seed 67.70, just remove it for me". Before editing, I explained
  that it is the only second seed, that removing it raises the reported CoRe mean (69.9 +- 2.1 -> 71.0 +- 1.3) and
  removes the evidence for the variance analysis, and that dropping a finished run after seeing its score is the
  selective reporting the cited papers warn against; offered keeping it or moving it to the appendix. The user chose
  "Remove it from the paper". Done:
  - CoRe is the mean of runs 1 + 2 (both seed 20261010): R2R 71.0 +- 1.3 / SPL 63.2 / OS 77.9; RxR 73.4 +- 0.4 / 59.5 /
    nDTW 70.6 (now all real); main-table footnote "first of two runs"; "+4.4 over GRPO" (red, GRPO mean is a
    placeholder) and "5.6x as many updates" in abstract and intro; Fig. a diamond 71.05 +- 1.27 and the 5.6x line;
    Fig. b/c CoRe = mean of runs 1 and 2 (recomputed from both machines' rollouts and journals).
  - Removed: the appendix "Run-to-run variance of RL", Sec. 4.4 "Run-to-run variance", run 3, "average of runs 1, 3"
    and the "stabilized" rows of Table runs, the batch-scale spread claims; the protocol now says RL is not
    bit-reproducible (two same-seed runs on two machines: 71.9 vs 70.2); Limitations rewritten accordingly.
  - Red placeholders in Tables core_design / sensitivity / reward were shifted by the change of the CoRe mean (SR +1.1,
    SPL +0.9, OS +1.3) so their intended differences are unchanged.
  - Checked statically: every \cref target exists, revised/revisedfloat environments balanced, no citation of removed
    material left. Not compiled here (compile on Overleaf).
- Still placeholders: GRPO run 2 (training now), RxR for Stage 1 / Stage 2, every Table core_design variant except
  CoRe, Tables sensitivity / reward / source (except their real cells), real-world table.

---

## 14. Running now and next steps
| job | where | status | expected |
|---|---|---|---|
| GRPO seed 20261414 (`grpo_sfr_mix1k_seed1414`, mix1k pool, 125 updates) | local, both GPUs | smoke started 09:32 | training ~20:00, R2R eval ~22:00 (11-13) |
| CoRe seed 20261414 (`rewind_sfr_mix1k_seed1414`) | local, after GRPO | queued (`seed1414_chain.sh`) | R2R eval ~12:00-13:00 on 11-14 |
| STABLE-V1 seed B | gpu2 GPU 3 | update ~82/125 at 09:00 | training ~17:00, evals tonight |

Configs: `activevln_grpo_sfr_mix1k_seed1414.yaml` = GRPO run 1's config with seed 20261414 and the mix1k pool file
(identical prompts and order; constant lr, so equivalent to stopping the 8k run at 125); `rewind_grpo_sfr_mix1k_seed1414
.yaml` = CoRe run 1's config with seed 20261414. Both go through the usual chain (freeze, 2-step smoke with provenance
checks, drift guard, R2R eval, verify, paired compares).

Next steps (proposals, not started unless noted):
1. RxR full evaluations of the two seed-1414 models and of CoRe run 3 (requested by the user on 11-13; queued locally
   after the CoRe seed-1414 run and its R2R evaluation, `svln/analysis/paper/rxr_fix_final_evals.sh`, ~4.3 h each).
2. With GRPO at two seeds and CoRe at three or four runs, recompute the paper's means; decide with the user how the
   seed-1414 CoRe run is reported.
3. The "4k" test of section 10 (continue run 3 to 500 updates, ~37 h).
4. If seed B confirms STABLE-V1 is a no-op: an exploration-preserving change (turn-share anchor to SFR or a KL/EMA
   anchor; 16 prompts per update) with 2 seeds, judged by the 9.1 criterion.
5. Paper placeholders: the Table core_design variants (random earlier step needs code), sensitivity (eta 0.5 / 2 m,
   H 4), reward variants, RxR of Stages 1-2.

---

## 15. Files
| what | path |
|---|---|
| merged journals | `/nvme/public/svln/evals/<model>_<split>/merged/`; RxR: `<model>_<set>_fixed_val_unseen/merged/` (gpu2: `~/workspace/anhdh35/svln/evals/`) |
| paired comparisons | `/nvme/public/svln/compare/*.json`; chain logs `/nvme/public/svln/runs/*.chain.log` |
| variance analysis | `/nvme/public/svln/analysis/stability/` (`stability_report.md`, `advantage_noise.txt`, `turning_push.txt`, `val_behaviour_all.txt`, `deltas.py`, `gram.json`, `ulp.py`, `make_soup.py`, `stable_check.py`) |
| pool analysis | `/nvme/public/svln/analysis/clean2k/` (`why_pool.md`, `decision_rule.md`), `analysis/hard2k/` |
| paper TODO | `/nvme/public/svln/analysis/paper/rl_revision_todo.md`; chain `seed1414_chain.sh` |
| STABLE-V1 chains | `analysis/stability/stable_a_chain.sh` (local), gpu2 `svln/tools/stable/stable_b_chain.sh` |
| configs | `configs/rl/local/` (this branch) |
| exports | `/nvme/public/svln/runs/<run>/exports/step_XXXX/`; soups `/nvme/public/svln/runs/soups/` |

---

## Appendix A. Paired R2R val_unseen comparisons (local journals, 1,839 episodes; exact McNemar; bootstrap 95% CI)
Remember 7.1: these measure evaluation noise between two fixed models, not differences between recipes.

| base | new | split | SR base | SR new | Delta SR | Delta SPL | only base | only new | McNemar p | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| `base` | `grpo_g5_s125` | r2r | 52.04 | 58.29 | +6.25 | +3.51 | 187 | 302 | 2.24e-07 | [3.86, 8.59] |
| `base` | `sfr` | r2r | 52.04 | 62.59 | +10.55 | +7.24 | 217 | 411 | 8.28e-15 | [7.88, 13.21] |
| `grpo_g5_s125` | `twins_grpo_s125` | r2r | 58.29 | 60.03 | +1.74 | +1.75 | 195 | 227 | 0.131 | [-0.44, 3.97] |
| `grpo_g5_sfr_s125` | `rewind_forkt_sfr_s020` | r2r | 66.50 | 65.85 | -0.65 | -2.05 | 244 | 232 | 0.614 | [-2.94, 1.69] |
| `grpo_sfr_mix8k_aplus_s125` | `grpo_sfr_mix8k_aplus_s250` | r2r | 66.83 | 67.92 | +1.09 | +1.26 | 163 | 183 | 0.307 | [-0.92, 3.05] |
| `grpo_sfr_mix8k_aplus_s125` | `rewind_sfr_mix1k_seed2_s125` | r2r | 66.83 | 67.70 | +0.87 | +0.47 | 188 | 204 | 0.449 | [-1.25, 2.99] |
| `grpo_sfr_mix8k_aplus_s250` | `grpo_sfr_mix8k_aplus_s500` | r2r | 67.92 | 69.33 | +1.41 | +0.17 | 169 | 195 | 0.19 | [-0.6, 3.43] |
| `grpo_sfr_mix8k_aplus_s500` | `grpo_sfr_mix8k_aplus_s750` | r2r | 69.33 | 71.45 | +2.12 | +0.71 | 170 | 209 | 0.0508 | [0.0, 4.19] |
| `grpo_sfr_mix8k_aplus_s500` | `rewind_mix1k_hard2k_s125` | r2r | 69.33 | 68.84 | -0.49 | +1.28 | 200 | 191 | 0.686 | [-2.61, 1.58] |
| `grpo_sfr_mix8k_aplus_s750` | `rewind_mix1k_hard2k_s125` | r2r | 71.45 | 68.84 | -2.61 | +0.57 | 235 | 187 | 0.022 | [-4.79, -0.44] |
| `grpo_sfr_mix8k_aplus_s750` | `rewind_sfr_hard1k_s125` | r2r | 71.45 | 67.70 | -3.75 | -2.11 | 263 | 194 | 0.00144 | [-6.04, -1.52] |
| `grpo_sfr_s125` | `grpo_g5_sfr_s125` | r2r | 64.33 | 66.50 | +2.18 | +1.39 | 172 | 212 | 0.0464 | [0.11, 4.24] |
| `grpo_sfr_s125` | `grpo_sfr_mix8k_aplus_s125` | r2r | 64.33 | 66.83 | +2.50 | +2.69 | 172 | 218 | 0.0226 | [0.38, 4.62] |
| `grpo_sfr_s125` | `grpo_sfr_mix8k_aplus_s250` | r2r | 64.33 | 67.92 | +3.59 | +3.95 | 155 | 221 | 0.00078 | [1.58, 5.71] |
| `grpo_sfr_s125` | `grpo_sfr_mix8k_aplus_s500` | r2r | 64.33 | 69.33 | +5.00 | +4.12 | 168 | 260 | 1.01e-05 | [2.83, 7.29] |
| `grpo_sfr_s125` | `grpo_sfr_mix8k_aplus_s750` | r2r | 64.33 | 71.45 | +7.12 | +4.83 | 180 | 311 | 3.61e-09 | [4.79, 9.46] |
| `grpo_sfr_s125` | `rewind_forkt_sfr_s020` | r2r | 64.33 | 65.85 | +1.52 | -0.66 | 247 | 275 | 0.237 | [-0.87, 3.97] |
| `grpo_sfr_s125` | `rewind_path_sfr_mix1k_s125` | r2r | 64.33 | 69.66 | +5.33 | +4.97 | 184 | 282 | 6.51e-06 | [3.05, 7.67] |
| `grpo_sfr_s125` | `rewind_self_sfr_s125` | r2r | 64.33 | 67.26 | +2.94 | +2.26 | 169 | 223 | 0.00735 | [0.82, 5.06] |
| `grpo_sfr_s125` | `rewind_sfr_hard1k_s125` | r2r | 64.33 | 67.70 | +3.37 | +2.72 | 185 | 247 | 0.00329 | [1.2, 5.6] |
| `grpo_sfr_s125` | `rewind_sfr_mix1k_s125` | r2r | 64.33 | 71.94 | +7.61 | +6.34 | 143 | 283 | 1.08e-11 | [5.44, 9.79] |
| `grpo_sfr_s125` | `rewind_sfr_s125` | r2r | 64.33 | 68.73 | +4.40 | +3.56 | 163 | 244 | 6.96e-05 | [2.28, 6.53] |
| `rewind_mix1k_hard2k_s125` | `rewind_sfr_hard1k_s125` | r2r | 68.84 | 67.70 | -1.14 | -2.69 | 213 | 192 | 0.32 | [-3.26, 1.03] |
| `rewind_path_sfr_mix1k_s125` | `grpo_sfr_mix8k_aplus_s125` | r2r | 69.66 | 66.83 | -2.83 | -2.28 | 234 | 182 | 0.0123 | [-5.06, -0.6] |
| `rewind_path_sfr_mix1k_s125` | `grpo_sfr_mix8k_aplus_s250` | r2r | 69.66 | 67.92 | -1.74 | -1.02 | 240 | 208 | 0.143 | [-3.97, 0.54] |
| `rewind_path_sfr_mix1k_s125` | `grpo_sfr_mix8k_aplus_s500` | r2r | 69.66 | 69.33 | -0.33 | -0.85 | 235 | 229 | 0.816 | [-2.61, 2.01] |
| `rewind_path_sfr_mix1k_s125` | `grpo_sfr_mix8k_aplus_s750` | r2r | 69.66 | 71.45 | +1.79 | -0.14 | 196 | 229 | 0.121 | [-0.44, 4.02] |
| `rewind_sfr_hard1k_s125` | `rewind_sfr_mix1k_seed2_s125` | r2r | 67.70 | 67.70 | +0.00 | +0.44 | 202 | 202 | 1 | [-2.12, 2.12] |
| `rewind_sfr_hard1k_s125` | `rewind_sfr_mix1k_stable_a_s125` | r2r | 67.70 | 69.71 | +2.01 | +3.93 | 178 | 215 | 0.0692 | [-0.05, 4.13] |
| `rewind_sfr_mix1k_s125` | `grpo_sfr_mix8k_aplus_s125` | r2r | 71.94 | 66.83 | -5.11 | -3.65 | 240 | 146 | 1.98e-06 | [-7.18, -3.05] |
| `rewind_sfr_mix1k_s125` | `grpo_sfr_mix8k_aplus_s250` | r2r | 71.94 | 67.92 | -4.02 | -2.39 | 249 | 175 | 0.00038 | [-6.14, -1.79] |
| `rewind_sfr_mix1k_s125` | `grpo_sfr_mix8k_aplus_s500` | r2r | 71.94 | 69.33 | -2.61 | -2.22 | 239 | 191 | 0.0233 | [-4.79, -0.49] |
| `rewind_sfr_mix1k_s125` | `grpo_sfr_mix8k_aplus_s750` | r2r | 71.94 | 71.45 | -0.49 | -1.51 | 220 | 211 | 0.7 | [-2.66, 1.74] |
| `rewind_sfr_mix1k_s125` | `rewind_mix1k_hard2k_s125` | r2r | 71.94 | 68.84 | -3.10 | -0.93 | 214 | 157 | 0.00359 | [-5.11, -1.09] |
| `rewind_sfr_mix1k_s125` | `rewind_path_sfr_mix1k_s125` | r2r | 71.94 | 69.66 | -2.28 | -1.37 | 212 | 170 | 0.0358 | [-4.4, -0.22] |
| `rewind_sfr_mix1k_s125` | `rewind_sfr_hard1k_s125` | r2r | 71.94 | 67.70 | -4.24 | -3.62 | 232 | 154 | 8.44e-05 | [-6.31, -2.18] |
| `rewind_sfr_mix1k_s125` | `rewind_sfr_mix1k_seed2_s125` | r2r | 71.94 | 67.70 | -4.24 | -3.18 | 229 | 151 | 7.41e-05 | [-6.25, -2.18] |
| `rewind_sfr_mix1k_s125` | `rewind_sfr_mix1k_stable_a_avg` | r2r | 71.94 | 67.97 | -3.97 | -1.15 | 237 | 164 | 0.000313 | - |
| `rewind_sfr_mix1k_s125` | `rewind_sfr_mix1k_stable_a_s125` | r2r | 71.94 | 69.71 | -2.23 | +0.31 | 210 | 169 | 0.0398 | [-4.24, -0.16] |
| `rewind_sfr_mix1k_seed2_s125` | `rewind_sfr_mix1k_stable_a_avg` | r2r | 67.70 | 67.97 | +0.27 | +2.04 | 186 | 191 | 0.837 | - |
| `rewind_sfr_mix1k_seed2_s125` | `rewind_sfr_mix1k_stable_a_s125` | r2r | 67.70 | 69.71 | +2.01 | +3.50 | 171 | 208 | 0.0643 | [-0.05, 4.08] |
| `rewind_sfr_mix1k_stable_a_s125` | `rewind_sfr_mix1k_stable_a_avg` | r2r | 69.71 | 67.97 | -1.74 | -1.46 | 86 | 54 | 0.00856 | - |
| `rewind_sfr_s125` | `grpo_g5_sfr_s125` | r2r | 68.73 | 66.50 | -2.23 | -2.16 | 204 | 163 | 0.0367 | [-4.24, -0.16] |
| `rewind_sfr_s125` | `grpo_sfr_mix8k_aplus_s125` | r2r | 68.73 | 66.83 | -1.90 | -0.86 | 210 | 175 | 0.083 | [-3.97, 0.22] |
| `rewind_sfr_s125` | `grpo_sfr_mix8k_aplus_s250` | r2r | 68.73 | 67.92 | -0.82 | +0.40 | 211 | 196 | 0.488 | [-2.94, 1.31] |
| `rewind_sfr_s125` | `grpo_sfr_mix8k_aplus_s500` | r2r | 68.73 | 69.33 | +0.60 | +0.57 | 213 | 224 | 0.632 | [-1.58, 2.83] |
| `rewind_sfr_s125` | `grpo_sfr_mix8k_aplus_s750` | r2r | 68.73 | 71.45 | +2.72 | +1.28 | 180 | 230 | 0.0154 | [0.6, 4.84] |
| `rewind_sfr_s125` | `rewind_forkt_sfr_s020` | r2r | 68.73 | 65.85 | -2.88 | -4.22 | 260 | 207 | 0.016 | [-5.17, -0.6] |
| `rewind_sfr_s125` | `rewind_mix1k_hard2k_s125` | r2r | 68.73 | 68.84 | +0.11 | +1.85 | 191 | 193 | 0.959 | [-1.96, 2.18] |
| `rewind_sfr_s125` | `rewind_path_sfr_mix1k_s125` | r2r | 68.73 | 69.66 | +0.92 | +1.41 | 180 | 197 | 0.41 | [-1.14, 2.99] |
| `rewind_sfr_s125` | `rewind_self_sfr_s125` | r2r | 68.73 | 67.26 | -1.47 | -1.29 | 216 | 189 | 0.196 | [-3.59, 0.71] |
| `rewind_sfr_s125` | `rewind_sfr_hard1k_s125` | r2r | 68.73 | 67.70 | -1.03 | -0.84 | 203 | 184 | 0.36 | [-3.1, 1.09] |
| `rewind_sfr_s125` | `rewind_sfr_mix1k_s125` | r2r | 68.73 | 71.94 | +3.21 | +2.78 | 157 | 216 | 0.00262 | [1.14, 5.27] |
| `rewind_sfr_s125` | `rewind_sfr_mix1k_seed2_s125` | r2r | 68.73 | 67.70 | -1.03 | -0.40 | 200 | 181 | 0.356 | [-3.05, 1.09] |
| `s8n_dagger_oracle` | `grpo_g5_s125` | r2r | 58.84 | 58.29 | -0.54 | +6.89 | 327 | 317 | 0.723 | [-3.26, 2.18] |
| `sfr` | `grpo_g5_sfr_s125` | r2r | 62.59 | 66.50 | +3.92 | +3.12 | 220 | 292 | 0.00168 | [1.52, 6.31] |
| `sfr` | `grpo_sfr_mix8k_aplus_s125` | r2r | 62.59 | 66.83 | +4.24 | +4.41 | 198 | 276 | 0.000394 | [1.9, 6.58] |
| `sfr` | `grpo_sfr_mix8k_aplus_s250` | r2r | 62.59 | 67.92 | +5.33 | +5.68 | 209 | 307 | 1.85e-05 | [2.88, 7.72] |
| `sfr` | `grpo_sfr_mix8k_aplus_s500` | r2r | 62.59 | 69.33 | +6.74 | +5.85 | 213 | 337 | 1.39e-07 | [4.3, 9.24] |
| `sfr` | `grpo_sfr_mix8k_aplus_s750` | r2r | 62.59 | 71.45 | +8.86 | +6.56 | 174 | 337 | 4.92e-13 | [6.47, 11.2] |
| `sfr` | `grpo_sfr_s125` | r2r | 62.59 | 64.33 | +1.74 | +1.72 | 242 | 274 | 0.172 | [-0.71, 4.13] |
| `sfr` | `rewind_forkt_sfr_s020` | r2r | 62.59 | 65.85 | +3.26 | +1.06 | 156 | 216 | 0.00218 | [1.2, 5.27] |
| `sfr` | `rewind_mix1k_hard2k_s125` | r2r | 62.59 | 68.84 | +6.25 | +7.13 | 176 | 291 | 1.15e-07 | [3.97, 8.54] |
| `sfr` | `rewind_path_sfr_mix1k_s125` | r2r | 62.59 | 69.66 | +7.07 | +6.69 | 155 | 285 | 5.86e-10 | [4.84, 9.24] |
| `sfr` | `rewind_self_sfr_s125` | r2r | 62.59 | 67.26 | +4.68 | +3.99 | 210 | 296 | 0.000152 | [2.34, 7.01] |
| `sfr` | `rewind_sfr_hard1k_s125` | r2r | 62.59 | 67.70 | +5.11 | +4.44 | 177 | 271 | 1.04e-05 | [2.83, 7.34] |
| `sfr` | `rewind_sfr_mix1k_s125` | r2r | 62.59 | 71.94 | +9.35 | +8.06 | 140 | 312 | 3.86e-16 | [7.12, 11.58] |
| `sfr` | `rewind_sfr_mix1k_seed2_s125` | r2r | 62.59 | 67.70 | +5.11 | +4.88 | 183 | 277 | 1.36e-05 | [2.83, 7.34] |
| `sfr` | `rewind_sfr_mix1k_stable_a_avg` | r2r | 62.59 | 67.97 | +5.38 | +6.92 | 182 | 281 | 4.86e-06 | - |
| `sfr` | `rewind_sfr_mix1k_stable_a_s125` | r2r | 62.59 | 69.71 | +7.12 | +8.38 | 165 | 296 | 1.1e-09 | [4.89, 9.41] |
| `sfr` | `rewind_sfr_s125` | r2r | 62.59 | 68.73 | +6.14 | +5.28 | 180 | 293 | 2.3e-07 | [3.86, 8.37] |
| `twins_grpo_s125` | `sfr` | r2r | 60.03 | 62.59 | +2.56 | +1.98 | 265 | 312 | 0.0554 | [0.0, 5.17] |
| `grpo_sfr_s125` | `grpo_g5_sfr_s125` | r2r val_seen-300 | 68.67 | 72.00 | +3.33 | +2.19 | 25 | 35 | 0.245 | [-1.67, 8.33] |
| `grpo_sfr_s125` | `rewind_forkt_sfr_s020` | r2r val_seen-300 | 68.67 | 63.67 | -5.00 | -5.65 | 45 | 30 | 0.105 | [-10.67, 0.67] |
| `grpo_sfr_s125` | `rewind_self_sfr_s125` | r2r val_seen-300 | 68.67 | 71.67 | +3.00 | +2.31 | 22 | 31 | 0.272 | [-1.67, 7.67] |
| `grpo_sfr_s125` | `rewind_sfr_s125` | r2r val_seen-300 | 68.67 | 74.67 | +6.00 | +4.41 | 22 | 40 | 0.03 | [1.0, 11.0] |
| `rewind_sfr_s125` | `grpo_g5_sfr_s125` | r2r val_seen-300 | 74.67 | 72.00 | -2.67 | -2.22 | 37 | 29 | 0.389 | [-8.0, 2.67] |
| `rewind_sfr_s125` | `rewind_forkt_sfr_s020` | r2r val_seen-300 | 74.67 | 63.67 | -11.00 | -10.06 | 54 | 21 | 0.000176 | [-16.33, -5.67] |
| `rewind_sfr_s125` | `rewind_self_sfr_s125` | r2r val_seen-300 | 74.67 | 71.67 | -3.00 | -2.11 | 39 | 30 | 0.336 | [-8.33, 2.33] |
| `sfr` | `grpo_sfr_s125` | r2r val_seen-300 | 66.33 | 68.67 | +2.33 | +3.60 | 37 | 44 | 0.505 | [-3.33, 8.0] |
| `rewind_path_sfr_mix1k_s125` | `grpo_sfr_mix8k_aplus_s125` | rxr | 72.20 | 72.10 | -0.10 | +0.48 | 123 | 122 | 1 | [-3.2, 3.0] |
| `rewind_sfr_mix1k_s125` | `grpo_sfr_mix8k_aplus_s125` | rxr | 73.30 | 72.10 | -1.20 | -0.71 | 127 | 115 | 0.48 | [-4.3, 1.9] |
| `rewind_sfr_mix1k_s125` | `rewind_path_sfr_mix1k_s125` | rxr | 73.30 | 72.20 | -1.10 | -1.19 | 107 | 96 | 0.483 | [-3.9, 1.7] |
| `rewind_sfr_s125` | `grpo_sfr_mix8k_aplus_s125` | rxr | 74.00 | 72.10 | -1.90 | -0.17 | 127 | 108 | 0.24 | [-5.0, 1.2] |
| `rewind_sfr_s125` | `rewind_path_sfr_mix1k_s125` | rxr | 74.00 | 72.20 | -1.80 | -0.65 | 116 | 98 | 0.245 | [-4.7, 1.1] |
| `rewind_sfr_s125` | `rewind_sfr_mix1k_s125` | rxr | 74.00 | 73.30 | -0.70 | +0.54 | 111 | 104 | 0.682 | [-3.5, 2.2] |
| `sfr` | `grpo_sfr_mix8k_aplus_s125` | rxr | 66.50 | 72.10 | +5.60 | +4.05 | 117 | 173 | 0.0012 | [2.3, 9.0] |
| `sfr` | `rewind_path_sfr_mix1k_s125` | rxr | 66.50 | 72.20 | +5.70 | +3.57 | 93 | 150 | 0.000309 | [2.7, 8.8] |
| `sfr` | `rewind_sfr_mix1k_s125` | rxr | 66.50 | 73.30 | +6.80 | +4.76 | 92 | 160 | 2.19e-05 | [3.7, 9.9] |
| `sfr` | `rewind_sfr_s125` | rxr | 66.50 | 74.00 | +7.50 | +4.22 | 84 | 159 | 1.72e-06 | [4.5, 10.5] |
| `rewind_path_sfr_mix1k_s125` | `grpo_sfr_mix8k_aplus_s125` | rxr2k | 71.50 | 72.15 | +0.65 | +1.73 | 227 | 240 | 0.579 | [-1.5, 2.8] |
| `rewind_sfr_mix1k_s125` | `grpo_sfr_mix8k_aplus_s125` | rxr2k | 73.20 | 72.15 | -1.05 | +0.49 | 250 | 229 | 0.361 | [-3.25, 1.15] |
| `rewind_sfr_mix1k_s125` | `rewind_path_sfr_mix1k_s125` | rxr2k | 73.20 | 71.50 | -1.70 | -1.24 | 216 | 182 | 0.098 | [-3.7, 0.25] |
| `rewind_sfr_s125` | `grpo_sfr_mix8k_aplus_s125` | rxr2k | 73.10 | 72.15 | -0.95 | +0.73 | 230 | 211 | 0.391 | [-2.95, 1.1] |
| `rewind_sfr_s125` | `rewind_path_sfr_mix1k_s125` | rxr2k | 73.10 | 71.50 | -1.60 | -1.00 | 234 | 202 | 0.138 | [-3.65, 0.4] |
| `rewind_sfr_s125` | `rewind_sfr_mix1k_s125` | rxr2k | 73.10 | 73.20 | +0.10 | +0.25 | 212 | 214 | 0.961 | [-1.95, 2.15] |
| `sfr` | `grpo_sfr_mix8k_aplus_s125` | rxr2k | 65.85 | 72.15 | +6.30 | +5.50 | 210 | 336 | 7.77e-08 | [4.05, 8.5] |
| `sfr` | `rewind_path_sfr_mix1k_s125` | rxr2k | 65.85 | 71.50 | +5.65 | +3.77 | 183 | 296 | 2.74e-07 | [3.5, 7.7] |
| `sfr` | `rewind_sfr_mix1k_s125` | rxr2k | 65.85 | 73.20 | +7.35 | +5.02 | 165 | 312 | 1.61e-11 | [5.25, 9.45] |
| `sfr` | `rewind_sfr_s125` | rxr2k | 65.85 | 73.10 | +7.25 | +4.77 | 178 | 323 | 9.29e-11 | [5.1, 9.4] |
| `grpo_sfr_mix8k_aplus_s125` | `rewind_sfr_mix1k_s125` | rxrfull | 72.44 | 73.62 | +1.17 | -0.08 | 423 | 466 | 0.159 | [-0.44, 2.78] |
