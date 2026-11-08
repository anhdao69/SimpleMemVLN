# SFR campaign: RL from S8N_DAGGER_SF_REMAINING (design, pre-registered 2026-11-02 12:00)

User goal (2026-11-02): pull the latest code and reports; download `S8N_DAGGER_SF_REMAINING` (SFR); first run
ActiveVLN GRPO from it, then the method that makes the CVPR story, with ablations, within ~5 days (until
~2026-11-07 12:00); a method that matches the baseline is acceptable if the story is good. Then the RL run on 8k
samples, with a justified answer to "how do we choose the 8k?". Results go to `s8n_local_campaign.md`.

## 1. Starting point

- `main` @ `0c77bb7` (`evaluation_reports/S8N_STAGE2_STOPFIX_R2R_EVALUATION.md`): SFR = one epoch from S8N epoch 1 on
  oracle-relabelled DAgger data with the STOP-label fix (14,195 episodes) + the original demonstrations of the
  16,620 instructions DAgger did not re-collect. R2R val_unseen (their host): SR 63.68 / SPL 56.65 / OSR 70.64 /
  NE 4.23 / TL 11.22 / 91.3 steps; SR/OSR 0.901 (S8N's stop precision is back); paired +11.26 SR over S8N.
- Weights `anhdao69/stage2_S8N` @ `7ef60965`, sha256 `e3ada114…`: all 25 files match `SHA256SUMS.json`;
  config / tokenizer / chat template / processor / navigation identical to S8N's. Local copy read-only at
  `/nvme/public/svln/checkpoints/stage2_S8N/S8N_DAGGER_SF_REMAINING`.
- Its remaining failures are mostly route failures: ~7 pp reach the goal area and leave it, ~29 pp never get
  within 3 m. From base S8N: GRPO +6.25 (G5, p = 2.2e-7) / +7.99 (run 1); F-B, event forks, memory twins
  (= GRPO at matched rollouts, p = 0.13) and fork-at-t rescue (spinning) are negative; rewind was never trained
  to completion (probe: 12/16 vs 6/16 hard failures rescued, p = 0.07).
- Interrupted: run 5 (GRPO G5 from ORACLE) stopped at step 98 by the 2026-11-01 08:57 power loss (resumable
  step 95); superseded by this campaign, kept, not evaluated.

## 2. The method under test: rewind

For each sampled main trajectory, the closest approach k* = first argmin_{k<t} d_k before the first decision t at
which the geodesic distance is >= 1 m above it (training-time, privileged like the reward). S8N's whole streaming
state is snapshotted at every new closest approach (constant 135.7 MiB, 1.4 ms clone; a full-history cache grows
to 0.9 GiB at t = 100 and 3.5 GiB at t = 400), so after the main ends one branch restarts *the agent's own belief*
at k*:
- **expert rewind** (`rewind_grpo_sfr_r2r1k`): the geodesic follower re-decides from k* (<= 16 forced decisions,
  until its 3rd forward move, or its STOP), then the policy continues; forced decisions are imitated only if the
  branch beat the main (max(0, R_b - R_m)/s, a counterfactual Q-filter), the policy's continuation gets
  (R_b - R_m)/s; mains keep GRPO;
- **self-rewind** (`rewind_self_grpo_sfr_r2r1k`, new 2026-11-02, `rescue.self_branch`): no expert; the branch is
  the policy's own independent retry from k*, every decision credited (R_b - R_m)/s. Expert-free; isolates the
  fork point from the privileged correction.

Claim to test: the decision where a streaming agent loses the route is its closest approach, and restoring its
constant-size belief there gives per-decision credit (and, with the expert, a verified correction) that
outcome-only GRPO lacks.

## 3. Runs

All: R2R 1k (ActiveVLN prefix), 125 steps, seed 20261010, ActiveVLN hyper-parameters, frozen code per launch;
evaluation once at step 125 on R2R val_unseen (1,839, paired McNemar + bootstrap, `compare_evals.py`) and
val_seen-300; per-episode verification (`scripts/rl/s8n_analysis/verify_eval.py`).

| slot | run | config | role |
|---|---|---|---|
| 0 | SFR journals | - | local reference (val_unseen, val_seen-300) |
| 1 | A | `activevln_grpo_sfr_r2r1k` (G = 4) | ActiveVLN baseline (user: first) |
| 2 | B | `rewind_grpo_sfr_r2r1k` (guard vs A; fallback no-imitation) | the method |
| 3 | C | `rewind_self_grpo_sfr_r2r1k` | who acts at k*: expert vs the policy itself |
| 4 | D or E | `rewind_noimit_grpo_sfr_r2r1k` / `activevln_grpo_g5_sfr_r2r1k` | credit of the expert segment / compute control |
| 5-6 | F | second seeds of the best method and of A | only if the best method leads A but n.s. |

Slots 1-2 are chained (`ops/local/sfr_campaign.sh`); 3+ are appended after B's result. Probes beside slot 0
(no training, `/nvme/public/svln/probes/sfr/`): expert rewind vs fork-at-t on the 32 base-hard instructions,
expert rewind on 64 pool rows.

## 4. Decision rules (fixed now)

- **Primary:** val_unseen SR at step 125 vs A, paired. A method is *better* at McNemar p < 0.05 (and only
  "compute-fair" if it also beats E, G = 5, when E is run); *non-inferior* if its SR is >= A - 1.0 pp.
- **Secondary (pre-declared, for a non-inferior method):** SPL; SR/OSR (stop precision); failure taxonomy
  ("never within 3 m", "reached then left", cap); RxR15 1,000-episode subset (zero-shot transfer); training
  efficiency (val_seen-300 at steps 25/50/75/100 of A and the method).
- **Mechanism (B, C):** H-B1 the rescue rate of triggered failed mains rises over training; H-B2 the trigger
  rate of mains falls faster than in A (same prompts); H-B3 B's forced decisions do not make the policy spin
  (drift guard never trips; mains' turn share within +0.06 of A).
- **Choice of the paper method:** the variant with the best primary result among the non-inferior ones; ties
  broken by the secondaries, then by simplicity (self-rewind needs no expert).

## 5. The 8k question (observational study, then a rule)

Pre-registered measures, from run A's rollouts on its 1k prompts (each prompt visited once, G = 4 at T = 1.2):
1. **Waste:** fraction of groups with zero advantage (all-success / all-fail) before and after dynamic-sampling
   rerolls, and the simulator time they cost. GRPO's signal per prompt is ∝ 1 - p^G - (1 - p)^G for pass rate p.
2. **Predictors of informative prompts** (mixed groups), AUC on A's prompts: (a) free: reference length N*,
   geodesic distance, instruction length, scene; (b) one greedy SFR pass on the same prompts (success, minimum
   distance, steps, mean decision entropy, STOP log-prob); (c) prior policies' rollouts on the same prompts
   (run 1, run 3 from base S8N: does difficulty transfer across policies?).
3. **Rule:** candidates are the 10,819 R2R train episodes (the RL env and budget are R2R-calibrated; RxR needs
   its own budget); the 8k = the best predictor's most informative episodes, stratified by scene, compared with
   random 8k on the expected informative fraction. If a slot is free, a 1k pilot (selected vs the ActiveVLN
   prefix, same 125 steps) tests whether selection improves learning per step.

### 5.1 Additions 2026-11-02 20:50 (before any SFR RL result)

- **ActiveVLN's own 8k is "4k R2R + 4k RxR, respectively"** (paper; two separate RL runs, one per dataset). Both
  parquets downloaded from GitHub at the audited commit `3a0c63b` (hashes equal to the audit's) to
  `/nvme/public/svln/data/downloads/activevln_gh/`:
  - R2R 4k (`r2r_4000_train.parquet`): consistent with a uniform sample of the 10,819 train episodes (mean
    geodesic 9.257 vs 9.274 m, all 61 scenes).
  - RxR 4k (`rxr_4000_train_new.parquet`): every row is in our local RxR15 English-guide train (19,996), language
    mix as the full set (en-IN 69 % vs 67 %), and it carries `extra_info.difficulty` in {-1, 0, 1, 2, 3, 4}
    (773 / 186 / 1,154 / 905 / 861 / 121). Not a length bin (path lengths overlap across all values), but
    monotone in length (mean primitives 136 at 0 -> 68 at 4), as expected of "successes out of 4 rollouts".
    **Rows 0-2,919 are exactly the 2,920 rows with difficulty 1-3** (shuffled); rows 2,920-3,999 are the 1,080
    rows with -1 / 0 / 4. With shuffle off and 8 prompts per step, the released step-350 RxR checkpoint has seen
    only rows 0-2,799: **ActiveVLN's RxR RL used only prompts with 1-3 successes out of 4**, a pass-rate
    (learnability) filter. The generating script is not released; reading "difficulty = successes of 4
    rollouts" is inferred from the file, UNVERIFIED.
- **Screen (chain v2, step 1b):** one greedy SFR pass over run A's 1k prompts (`evals/sfr_screen_rl1k_r2r_train`,
  unchanged evaluator; the copy at `code/screen_train_20261102` differs from the frozen tree only in
  `--split` accepting `train`). Analysis: `scripts/rl/s8n_analysis/prompt_signal.py --runs grpo_sfr --screen ...`
  (AUC of greedy success / final distance / minimum distance / steps for run A's all-success and mixed groups;
  informative share after dropping the screen's most-solved prompts).
- **Decision structure for the 8k** (to be filled with run A + screen numbers):
  1. composition: R2R-only (8k of 10,819: selection can only drop 2.8k) vs ActiveVLN-style 4k R2R + 4k RxR
     (RxR needs RL-environment support for `RxRVLNCE-v1`, budgets from RxR N* and the macro length);
  2. within a dataset: random vs screen-based (drop "solved" prompts; ActiveVLN-RxR-style 1 <= k <= 3 of 4)
     vs learnability-weighted p(1-p), stratified by scene;
  3. ordering: the pass rate of a fixed prompt rises during training (S8N runs: informative share 0.70 -> 0.60
     over 125 steps; ORACLE 0.84 -> 0.43), so a step-0 screen goes stale over a 1,000-step 8k epoch; hard
     prompts may become learnable later (curriculum: frontier first, hard last) and online skipping of
     all-success groups handles the drift.

### 3.1 Chains (2026-11-02 21:00, pre-registered before any SFR RL result)

- Part 1 `ops/local/sfr_campaign2.sh` (v2 replaced v1 at 20:45 while v1's SFR evaluation kept running; same
  frozen tree `code/sfr_campaign_20261102_2034`): SFR journals -> screen (step 1b) -> rewind smoke -> run A ->
  run B (guarded vs A; fallback no-imitation).
- Part 2 `ops/local/sfr_campaign_part2.sh` (waits for "part 1 finished"): run C self-rewind (smoke first, guarded
  vs A) -> run E G5 compute control; comparisons A->C, B->C, A->E, E->{B, C}.
- Part 3 (decided from parts 1-2): the 8k-selection pilot (1k screen-selected prompts, GRPO G4, vs A) and
  second seeds (slot F). Estimated finish of part 2: ~2026-11-05 10:00.

### 3.2 Ablation table on the 1k prompts (user 2026-11-03 19:30: no second seeds; finish the ablation table)

Chain part 2 v3 (`ops/local/sfr_campaign_part2c.sh`, launched 19:29, replaced the waiting v2). Every row: from
SFR, run A's 1k prompts, seed 20261010, 125 steps, one component changed (configs diffed), val_unseen (1,839) +
val_seen-300 + per-episode verification, paired comparisons with A, B and SFR; intervention runs keep the drift
guard vs A (a stopped run is evaluated at its latest export and reported as stopped).

| row | run | what changes vs its reference | question |
|---|---|---|---|
| SFT | SFR | - | starting point |
| A | grpo_sfr | ActiveVLN GRPO, G = 4 | baseline RL (done: 64.33) |
| B | rewind_sfr | + expert rewind at k*, verified imitation | full method (done: 68.73) |
| C | rewind_self_sfr | B without the expert: the policy re-decides from k* | is the privileged expert needed? (done: 67.26) |
| E | grpo_g5_sfr | A with G = 5 | is B's gain extra compute (+33 % time, +28 % tokens)? (done: 66.50; B vs E +2.23, p 0.037) |
| F | rewind_noimit_sfr | B with no credit on the expert's decisions | imitation vs continuation credit (skipped by the user at step 20) |
| G | rewind_forkt_sfr | B intervening at the trigger t (no rewind) | does the rewind to k* matter? (drift-guard stop at step 20: 65.85) |
| H | rewind_unverified_sfr | B imitating every correction (DAgger-style) | does outcome gating matter? (dropped) |
| D | grpo_learn_sfr | A on the learnability_top pool (5.3) | data selection (8k pilot) (skipped by the user) |

Status 2026-11-08: later runs (mix1k 71.94, path_mix1k 69.66, the 8k-pool ActiveVLN and rewind runs) and one table
of every result are in `s8n_local_campaign.md` §4.1.

Order C -> E -> F -> D -> G -> H, then val_seen-300 learning curves of A and B (steps 25-100) and RxR val_unseen
subset-1000 for SFR / A / B. Disk: the resumable states of A and B are deleted at the start; each later run's
resumable state and intermediate exports are deleted after its evaluations. Estimated: C ~11-04 08:00, E ~19:00,
F ~11-05 06:00, D ~15:30, G ~11-06 03:00, H ~14:00, curves + RxR ~21:00 (deadline 11-07 12:00).

### 5.2 Literature for the 8k rule (checked 2026-11-02)

- Bae et al., *Online Difficulty Filtering for Reasoning Oriented RL* (arXiv 2504.03380, EACL 2026): keep only
  prompts whose G-sample pass rate lies in a band (e.g. 0.25-0.75); expected improvement is lower-bounded by the
  variance of task success, maximal near p = 0.5; +10 % AIME over GRPO.
- Foster, Sims, Forkel, Foerster, *LILO: Learning to Reason at the Frontier of Learnability* (arXiv 2502.12272,
  NeurIPS 2025): sample prompts by p(1-p) (Sampling for Learnability from UED); up to 3x fewer steps.
- Zheng et al., *GRESO: Act Only When It Pays* (arXiv 2506.02177): zero-variance prompts are temporally
  consistent (> 90 % of prompts flagged in an epoch were flagged before) but ~20 % become informative again;
  skip probabilistically before rollout; up to 3.35x fewer rollouts than dynamic sampling at equal accuracy.
- DAPO (dynamic sampling: over-sample, drop all-0 / all-1 groups); SPEED-RL (arXiv 2506.09016: cheap pass-rate
  screen before full inference, 2-6x faster). ActiveVLN itself: re-rolls all-fail groups only; RxR pool filtered
  to 1-3 successes of 4 (section 5.1).
- What is specific to VLN and must be measured here: rollouts are long (90-150 decisions, ~1 min each), so a
  wasted group costs far more than in math RL; the budget, not a token limit, truncates; and the pool is small
  (R2R train 10,819), so selection power is limited for an 8k R2R-only pool.

### 5.3 Run A steps 0-30 and a retrospective selection study (2026-11-03 01:05, before any SFR RL evaluation)

**Run A, first visit of 248 prompts (steps 0-30):** groups with signal 63.3 %, all-success 32.7 %, all-fail after
re-rolls 4.0 % (9.7 % re-rolled), 39 % of sampled trajectories on zero-signal groups (S8N runs: 30 / 28 %
all-success). `analysis/prompt_signal_grpo_sfr_s30.json`.

**Sampled pass rate per screen bin** (`select_pool.py calibrate`, attempts counted: a record of attempt a also
stands for a failed rollouts of the earlier all-fail groups; `analysis/calibration_grpo_sfr_s30.json`). The
final distance of a greedy success does not predict the sampled pass rate (0.73 / 0.73 / 0.71 / 0.70 for d < 0.5 /
1 / 2 / 3 m), the reference length does; bins were simplified accordingly:

| greedy SFR | short (N* < 49) | mid | long (N* >= 64) | | failed: reached <= 3 m | near 3-5 m | far |
|---|---:|---:|---:|---|---:|---:|---:|
| solved: p_hat | 0.82 | 0.73 | 0.54 | | 0.36 | 0.34 | 0.15 |

A greedy success is solved only ~72 % of the time at T = 1.2 (a greedy failure ~35 %), so a "solved" prompt is
not solved under sampling: the greedy outcome alone separates informative prompts weakly.

**Retrospective selection study** (`scripts/rl/s8n_analysis/selection_sim.py`,
`analysis/selection_sim_s30.json`): leave-one-out on logged groups (first visit, first attempt; re-rolled prompts
excluded and counted): the screen is one rollout of the group, the label is the signal of the other G - 1;
scores computed only from the screen rollout, free features or the greedy SFR pass; the combined models are
2-fold cross-fitted by prompt. Signal share of the kept half (keep 0.5) and gain over keeping all, bootstrap 95 %
CI over prompts:

| rule (drop the most-solved half by ...) | run A (SFR), 232 prompts | run 1 twins (S8N), 848 | run 3 G5 (S8N), 853 |
|---|---|---|---|
| none (base rate, G - 1 rollouts) | 0.528 | 0.542 | 0.599 |
| random half | +0.6 | +0.3 | -0.4 |
| one sampled rollout: success | **+16.6** [10.8, 21.4] | +16.2 | +13.7 |
| one sampled rollout: success + entropy + agreement + steps + N* (logistic) | +15.1 | **+16.3** | **+15.6** |
| greedy SFR pass: success | +14.7 [9.5, 19.8] | +8.4 (cross-policy) | +6.7 (cross-policy) |
| greedy SFR pass: calibrated bins, top p(1-p) | +12.5 | +8.5 | +6.8 |
| free: shortest N* dropped | +14.0 | +7.3 | +7.7 |
| one sampled rollout: entropy only | +11.4 | +8.0 | +9.3 |

Findings: (1) any outcome-based pre-screen of the trained policy lifts the informative share by ~14-17 points
(~+30 % relative); (2) the screen must come from the policy that is trained: SFR's greedy pass predicts the S8N
runs at half the gain; (3) entropy is a weaker selector than the outcome; (4) one greedy pass is as good as one
sampled rollout for the matched policy (AUC all-success 0.72 vs 0.72), so the unchanged evaluator can screen;
(5) naive drop-solved keeps every greedy failure, including the far ones (p_hat 0.15, all-fail prone): in a dry
run (500 of the 1k) its expected informative share under the calibration is 0.675, below random (0.688), while the
top p(1-p) rule reaches 0.79. Caveat for the pilot: GRPO's loss here is verl's token mean over every micro-batch,
including zero-advantage ones, so a higher informative share also raises the effective step size; that is part
of the mechanism (as in DAPO's dynamic sampling) and is reported, not controlled.

**Run D, pre-registered now** (chain part 2 v2, `ops/local/sfr_campaign_part2b.sh`, which replaced the waiting
`sfr_campaign_part2.sh` at 01:05; steps 5 and 6 unchanged): after part 1, a second greedy SFR screen over the 4k
rows 1,000-1,999; calibration from all of run A (bins above, attempts counted); `select_pool.py --rule
learnability_top --n 1000 --seed 20261103` over rows 0-1,999 (scene-stratified; the 4k row blocks have equal N*
58.9 / 58.3 / 58.2 / 57.6, geodesic 9.37 / 9.21 / 9.27 / 9.17 m and 59-61 scenes, so A's prefix is a random-pool
control); config `activevln_grpo_sfr_learn1k.yaml` = run A's with only the pool changed. Order C -> D -> E.
Decision for the 8k recommendation: D >= A - 1 pp on val_unseen with a higher realised signal share -> screened
top-p(1-p) selection; D >= A + 1.5 pp -> strong recommendation; D < A - 1 pp -> random (ActiveVLN) plus online
skipping of all-success groups.

## 6. SFR local journals (2026-11-02 22:15) and the fallback if rewind fails (fixed before any SFR RL result)

- SFR, local, R2R val_unseen 1,839: **SR 62.59 / SPL 55.59 / OSR 69.98 / NE 4.27 / TL 11.33**, 0 evaluation
  failures (report host: 63.68; base S8N is 52.42 there vs 52.04 here). vs base +10.55 SR (411 vs 217, McNemar
  p = 8.3e-15, CI +7.9..+13.2), SPL +7.24; vs run 1 (best GRPO from S8N) +2.56 (p = 0.055). val_seen-300: SR 66.33
  / SPL 60.46 / OSR 78.0. Verifier: 1 flag, episode 610 step 142, one forward move of 0.425 m horizontal (no height
  change; navmesh snap; never seen in other journals, max 0.289 m); the episode ends at the 500-action cap 1.54 m
  from the goal (success). Simulator event, not an evaluator error; the journal is used.
- Failure taxonomy (% of 1,839): SFR success 62.6 / reached <= 3 m then stopped outside 7.4 / never within 3 m
  30.0 (10.5 got within 3-5 m) / cap 3.0. Base 52.0 / 6.0 / 41.9 / 6.7; run 1 60.0 / 8.3 / 31.6 / 8.8; ORACLE
  58.8 / 13.8 / 27.4 / 3.7. SFR's dominant remaining error is the route, not the stop.
- **Fallback ladder** (decided now):
  1. B or C non-inferior to A (>= A - 1 pp): rewind stays the method; the story rests on the pre-declared
     secondaries and on C (expert-free) vs B.
  2. B fails, C non-inferior: self-rewind is the method (no privileged actions; only the trigger uses the
     distance, as the reward does).
  3. Both fail: memory forks are **not** re-run blindly: on S8N the retention rule's return gradient was zero
     within noise (808 CRN pairs, sum z^2 = 5.74, p = 0.99) and MemFork on M5 collapsed. They come back only if a
     cheap probe shows memory headroom on SFR: SFR on val_seen-300 with the alternative retention rules (start-frame
     rule; recent-only) -> if SR moves by >= 3 pp in either direction, a memory-twin run from SFR; if not, there is
     nothing for a memory policy to learn. Otherwise the paper is S8N + SFR (DAgger with the STOP fix) + GRPO with
     the learnability-selected data (section 5) and the negative results as analysis.
