# RL on S8N (streaming GDN-drift memory): ActiveVLN GRPO, GRPO+F-B, retention probes and memory twins

Status (2026-10-29 10:45 EDT): **IN PROGRESS, INTERRUPTED.** Finished and evaluated: base S8N reproduction,
ActiveVLN GRPO, GRPO+F-B, the event-boundary analyses H1/H2 and the retention probes. The candidate method
(memory twins) is implemented and tested but has **no evaluation result**: it has only a 10-step pilot. A UF
Research Computing administrator cancelled all queued and running jobs at 08:47:03 on 2026-10-29 (§8).
Resubmission waits for the user.

**Current reading (one seed per method):**

1. **ActiveVLN GRPO on S8N works.** On val_unseen, SR goes from 51.50 to 60.41 (+8.92, 95% CI [+6.6, +11.2],
   McNemar p = 4e-14) and SPL from 47.93 to 54.66.
2. **No RL method beats ActiveVLN GRPO on S8N yet.**
   - GRPO+F-B (tree RL with entropy-placed forks and sibling credit) reaches SR 60.63. Against GRPO that is
     dSR +0.22 (CI [-2.0, +2.4], p = 0.89), with 2.2x the simulator steps and 1.5x the wall time. This is a
     negative result.
3. **The event-boundary forking hypothesis fails its pre-registered test (H1).**
   - S8N's drift at a fork does not predict how far sibling outcomes diverge once position in the episode is
     controlled: z = 1.32-1.88 and the quintiles are not monotone.
   - Event-boundary forks (EBF) were therefore not run as a method.
4. **The retention rule matters, and the best rule depends on the policy.**
   - An inference-only "keep the start frames" rule gives base S8N +5.67 SR on val_seen-300 (p = 0.040; Holm
     p = 0.12 over the three probe arms).
   - The same rule gives the RL policy -2.0 SR (n.s.).
   - No tested rule is worse than S8N's own.
5. **Memory twins** learns the per-layer retention rule during GRPO, by antithetic ES on common-random-number
   twins.
   - The 247 CPU tests pass.
   - The 10-step pilot shows that the learning signal exists: 72.5% of twin pairs diverge and 12.5% differ in
     success.
   - There is no navigation result yet, so the paper story in §9 is **unvalidated**.

Tables: `scripts/rl/s8n_report.py` -> `results/s8n/` (`s8n_tables.md`, `s8n_report.json`, one
`*_comparison.json` per pair). They are regenerated from the finished journals only; pending rows are listed
as pending. The design log, with the pre-registration and its dated amendments, is `s8n_event_rl_design.md`.
The M5 and Window8 predecessors are `memfork_report.md` and `hico_po_report.md`.

## 1. Protocol (identical for every row)

**Initial policy: S8N, epoch 1.**
- Checkpoint `M5_S/S8N`: `pytorch_model.bin` sha256 `c89de946…`, Qwen3.5-4B revision `851bf6e8…`.
- Memory: `episodic_window_gdn_sweep_v1`. It keeps R = 4 recent observation groups (including the current one)
  plus, per attention layer, the top-4 whole frames (B = 1,200 tokens). Frames are ranked by the FP32
  normalised GDN state drift `||S_after - S_before||_F / max(||S_before||_F, eps_l)`, fixed at the frame's
  birth.
- Reference SR 52.42 (964/1,839) on the original server.

**RL data.**
- The 1k prefix of ActiveVLN's fixed-order R2R 4K list: subset sha256 `ba062de3…`, pool `a1f93b80…`.

**RL recipe** (identical to the Window8 and M5 runs):
- 125 steps x 8 instructions, G = 4, T = 1.2, step budget 2 x N* (cap 120).
- Dynamic sampling with up to 2 attempts.
- AdamW lr 1e-6 (0.9/0.999, wd 0.01), grad clip 1.
- Clip 0.2/0.28 (dual clip 3), 1 PPO epoch, no KL, no entropy bonus, token-mean loss.
- Reward `success_reward_base` 15 (nDTW term off). Seed 20261010.

**Training setup.**
- 2 x RTX PRO 6000 Blackwell (96 GB), 3 rollout helper processes per rank.
- Realised-memory replay: the trainer's full-episode forward attends to exactly the per-layer frames each
  rollout retained (`rl/sweep_memory.py`). The trainer-vs-rollout log-prob mismatch is 0.004 on average.

**Evaluation.**
- The unchanged SimpleMem R2R-VLNCE evaluator: Habitat 0.2.4, greedy, seed 42, 500-action cap,
  `zero_sr_spl_keep_partial_metrics` failure policy.
- Split across 6 worker processes on 2 GPUs. Episodes are independent, so the merged journal does not depend
  on the split.
- **val_unseen (1,839 episodes) is run once per final checkpoint (step 125).**
- Development and selection decisions use only the fixed val_seen-300 subset
  (`data/rl/r2r_val_seen_subset300.json`).

**Paired tests.**
- `scripts/rl/compare_evals.py`: protocol and evaluation-source hashes must match; exact McNemar on success;
  10,000-draw bootstrap over episodes.

**Base reproduction (gate before any RL).**
- Base S8N on HiPerGator: val_unseen 947/1,839 (SR 51.50, SPL 47.93, 0 failures).
- The reference is 964, so this is inside the pre-set +-28 tolerance.
- **All RL deltas are taken against the local 51.50 journal.**

**Code snapshots** (`svln_runs/code/s8n_rl_v*`, frozen from this checkout and checksum-verified at job start):
- v1/v2: base evaluation, GRPO, F-B.
- v3/v4: tilt-capable evaluation, probes; v4 also has twins.
- v5: v4 plus the twins-GRPO and G = 5 configs.

v1/v2 and v3/v4/v5 have different evaluation-source hashes, because v3 can load `sweep_retention.json`.
- **Code-path check:** base S8N re-evaluated on v4 without a tilt file reproduces the v2 val_seen-300 journal
  exactly. All 300/300 episodes have identical step records (response tokens, executed actions, retained KV
  tokens; wall-clock timings excluded) and identical metrics.
- F-B + start tilt (v4) is therefore compared with F-B (v2) using `--allow-source-diff`. This is the only such
  pair.

## 2. Runs

| Run | Config | Code | Slurm | Status |
|---|---|---|---|---|
| Base S8N, val_unseen + val_seen-300 | — | v1 | 48012616 (dev allocation) | **done** 10-27 23:32 |
| GRPO+F-B (entropy forks) | `hico_grpo_fork_bs_s8n_r2r1k` | v2 | train 48037308, evals 48037310-15 | **done** 10-28 12:10, 125 steps in one launch |
| ActiveVLN GRPO | `activevln_grpo_s8n_r2r1k` | v2 | train 48037316 -> 48134830, evals 48037317-22 | **done** 10-29 03:10 (resumed once, §7) |
| Retention probes, base (b = +0.1 / +2 / -2) | `scripts/rl/make_tilt_checkpoint.py` | v3 | 48105427 / 29 / 30 | **done** 10-28 |
| Code-path check, base on v4 | — | v4 | 48130812 | **done** 10-28 18:43 |
| F-B step 125 + start tilt | — | v4 | 48130813 | **done** 10-28 19:11 |
| Memory twins on F-B (pre-registered) | `memory_twins_s8n_r2r1k` | v4 | 48130814 | **stopped at step 10** by the user (10-29 08:30); resumable checkpoint archived to /orange |
| Start tilt, base, val_unseen | — | v4 | 48130821 | **cancelled by the admin** at 259/1,839 (partial journal not used) |
| **Memory twins on GRPO (the method)** | `memory_twins_grpo_s8n_r2r1k` | v5 | — | **not started** (queued chain cancelled by the admin) |
| GRPO G = 5 (compute control for twins) | `activevln_grpo_g5_s8n_r2r1k` | v5 | — | not started (cancelled) |
| GRPO G = 8 (control for F-B's interaction) | `activevln_grpo_g8_s8n_r2r1k` | — | 48055208 | not started (cancelled while pending; optional since F-B = GRPO) |
| Twins without common random numbers (ablation) | `memory_twins_grpo_nocrn_s8n_r2r1k` | v6 (to freeze) | — | config + test only |
| Event-boundary forks (EBF / EBF-pos) | `ebf_s8n_r2r1k`, `ebf_pos_s8n_r2r1k` | v2 | — | **dropped** by the pre-registered H1 rule (§4); 2-step dev runs only |

## 3. Results

### 3.1 val_unseen, step 125 (primary)

| Policy | SR | SPL | NE | OSR | TL | forced stops | failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| S8N (no RL; local journal) | 51.50 | 47.93 | 5.70 | 58.08 | 9.43 | 128 | 0 |
| ActiveVLN GRPO (G = 4) | 60.41 | 54.66 | 4.78 | 67.16 | 10.19 | 146 | 0 |
| GRPO+F-B (G = 4) | **60.63** | 54.36 | 4.66 | 67.92 | 10.37 | 114 | 0 |
| Memory twins on GRPO | pending | | | | | | |
| GRPO G = 5 | pending | | | | | | |

Paired over the 1,839 episodes:

| new vs base | dSR [95% CI] | dSPL [95% CI] | dNE | gained / lost | McNemar p |
|---|---|---|---:|---|---:|
| GRPO vs S8N | +8.92 [+6.6, +11.2] | +6.73 [+4.6, +8.8] | -0.92 | 319 / 155 | 4.0e-14 |
| F-B vs S8N | +9.14 [+6.9, +11.4] | +6.43 [+4.3, +8.5] | -1.04 | 325 / 157 | 1.5e-14 |
| **F-B vs GRPO** | **+0.22 [-2.0, +2.4]** | -0.30 [-2.3, +1.6] | -0.12 | 219 / 215 | **0.89** |

Successes: S8N 947, GRPO 1,111, F-B 1,115. RL helps S8N about as much as it helped the M5 policy with F-B
(`memfork_report.md`: M5_E1 51.22 -> GRPO+F-B 60.03). On S8N, however, plain GRPO already reaches the F-B
level, whereas on M5 GRPO alone reached only 55.68. Gains do not depend on instruction length: val_unseen SR
by instruction-word tercile (<= 21 / 22-30 / > 30 words) is 50.9 / 51.6 / 52.2 for S8N, 60.7 / 59.4 / 61.1
for GRPO and 60.6 / 59.4 / 62.0 for F-B.

### 3.2 Compute (training records, last record of a step)

| Run | updates | main rollouts (+ discarded by dynamic sampling) | trained sequences | forks | simulator steps | policy forward steps | sum of step time |
|---|---:|---:|---:|---:|---:|---:|---:|
| GRPO | 125 | 4,000 (+1,112) | 3,680 | 0 | 269 k | 269 k | 8.05 h |
| GRPO+F-B | 125 | 4,000 (+980) | 6,138 | 3,598 | 596 k (272 k main + 187 k branch + 138 k replay) | 458 k | 12.45 h |
| Twins on F-B, pilot | 10 (stopped) | 320 (+84) | 480 | 281 | 49 k (incl. 5.2 k twin) | 38 k | 0.96 h |

GRPO's resume re-ran steps 5-9 (10.5 k simulator steps, 0.33 h); these runs are not on the final trajectory
and are not counted above. Training SR (main rollouts, T = 1.2) rose from 0.581 to 0.668 for GRPO and from
0.596 to 0.681 for F-B (steps 0-24 to 100-124). Truncation at the step budget rose from 4.1% to 9.1% for GRPO
and from 3.8% to 11.1% for F-B.

### 3.3 val_seen-300 by export step (development split; base S8N 155/300 = 51.67)

| step | GRPO SR / SPL | F-B SR / SPL | F-B - GRPO dSR (p) | F-B - base dSR (p) |
|---:|---|---|---|---|
| 25 | 50.67 / 46.43 | 52.00 / 47.90 | +1.33 (0.67) | +0.33 (1.0) |
| 50 | 54.67 / 49.22 | 56.00 / 50.75 | +1.33 (0.71) | +4.33 (0.17) |
| 75 | 60.00 / 54.75 | **63.00** / 58.16 | +3.00 (0.34) | +11.33 (1.5e-4) |
| 100 | 56.33 / 50.32 | 61.00 / 56.74 | +4.67 (0.15) | +9.33 (0.002) |
| 125 | **61.00** / 56.79 | 61.33 / 56.81 | +0.33 (1.0) | +9.67 (0.001) |

- GRPO's journals at steps 75/100/125 each contain one failed episode (id 547). The policy emitted the
  invalid response `TURN_LEFT_LEFT`, which the unchanged failure policy scores as SR = SPL = 0. It is a policy
  output, not an infrastructure failure. F-B has none, and both val_unseen journals have 0 failures.
- F-B led GRPO numerically at every export, with the largest gap at 75-100, but never significantly. Both runs
  have their val_seen-300 maximum at step 75 or 125.
- Step 125 is the pre-registered checkpoint, so it is the only one evaluated on val_unseen.

## 4. Decision D2 for (a) "bounded-memory tree RL" (GRPO+F-B on S8N): negative

The pre-registered rule (design §8) was: a method qualifies only if it beats ActiveVLN GRPO with
McNemar p < 0.05 on val_unseen at step 125. **F-B: +0.22 SR, p = 0.89 -> does not qualify.** It used 2.2x
GRPO's simulator steps (596 k vs 269 k), 1.7x the policy forward steps and 1.5x the wall time for the same
result. The branch-state argument (an S8N snapshot is O(1): about 230 MiB, against 1.0 / 1.9 / 3.9 GiB of
full-history KV at t = 50 / 100 / 200) is real engineering, but it does not buy accuracy here. (a) is reported
as a negative result. Together with the M5 results (`memfork_report.md`: F-B > GRPO on M5_E1 by +4.35), the
advantage of fork credit therefore depends on the policy: it does not carry over to S8N, where plain GRPO
already gains +8.9.

## 5. Event-boundary forking: H1 fails, H2 holds moderately

The original thesis (design §1) was that S8N's drift p_t marks event boundaries, and that RL should branch
rollouts there. Data: all 3,598 entropy-placed forks of the F-B run (997 episodes, steps 0-124). Script
`scripts/rl/analyze_event_forks.py`, controls `scripts/rl/s8n_analysis/`, outputs `results/s8n/h1h2/`.

**H1 (pre-registered): does the event score at a fork predict sibling informativeness?**
- Statistic: the position-normalised event score `event_pos` (score minus the same step's median in the same
  position bucket).
- Pass rule: z > 2 and monotone quintiles.

| statistic | value | needed |
|---|---|---|
| `event_pos` logistic z (with entropy, log(1+t)) | 1.88 | > 2 |
| `event_pos` quintiles, P(R_branch != R_main) | 0.625 / 0.705 / 0.668 / 0.672 / 0.650 | monotone |
| `event_pos` z with t/N* and log(1+t) controls | 1.32 (coef 0.05 log-odds/SD) | — |
| raw event z: no position control / + log(1+t) / + t/N* + log(1+t) | 5.51 / 2.19 / 1.54 | — |

**H1 fails.** The raw drift is mostly a proxy for "early in the episode". `||S_before||` grows with t, so the
raw score falls with t: Spearman with t/N = -0.36 on base S8N, and the share of decisions above the pooled
median goes from 1.00 at t < 3 to 0.29 at t >= 50 (`h1h2/drift_by_t_fb_step0.txt`). Two more observations:

- Exploratory, not pre-registered: success flips between main and branch (16.2% of forks). With controls,
  `event_pos` z = 2.42 and entropy z = 3.81. The flip rate by `event_pos` quintile is 0.146 / 0.146 / 0.161 /
  0.186 / 0.171, a lift of at most 1.2x. Flips depend mostly on position (t/N* z = -11.5): they are rare near
  the goal. No placement feature lifts the flip rate above about 1.3x the base rate, so fork placement has
  little headroom as a lever.
- The pre-registered "informative" target (R differs at all) is dominated by the main outcome. The reward has
  a final-distance term, so 93% of the 2,298 forks of successful mains differ, against 19% of the 1,300 forks
  of failed mains.

**H2 proxy holds moderately.** H2 asks whether event peaks sit at path geometry.
- Data: 115,221 decisions after two forward steps, so o_t is not a rotation of o_{t-1}. Positions are not
  logged, so bends come from the logged bearing error to the goal.
- The share of top-20% `event_pos` decisions rises from 0.19 on straight segments (bend < 6 degrees) to 0.29
  at 17-34 degree bends of the shortest path.
- Spearman(`event_pos`, bend) = 0.17, Spearman(`event_pos`, progress-slope change) = 0.11.

Reading: S8N's drift does mark geometric events (good for the memory claim), but those are not where sampled
alternatives decide the outcome. The drift-fork coupling of the original thesis is not supported, and EBF was
not run (pre-registered consequence).

## 6. The retention rule matters and is policy-dependent

### 6.1 What S8N keeps (F-B steps 0-4 rollouts, policy close to base S8N, 159 episodes with >= 20 decisions)

Frame sources each attention layer retains at the end of the episode:

| layer | 3 | 7 | 11 | 15 | 19 | 23 | 27 | 31 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| share of retained frames from t <= 3 | 0.70 | 0.33 | 0.40 | 0.31 | 0.41 | 0.64 | 0.72 | 0.81 |
| median source position t/N | 0.03 | 0.37 | 0.25 | 0.25 | 0.12 | 0.03 | 0.03 | 0.03 |

The outer layers (3, 23, 27, 31) mostly keep episode-start frames; the middle layers (7, 11, 15) keep
mid-episode frames. S8N's rule is also fragile. In an offline replay of the logged drift with the tilted
priority (1 - a) log p^l + a mean_l' log p^l' + b log(1 + s), where s is the frame's step:

| tilt | overlap with S8N's frames | share from t <= 3 | mean source position |
|---|---:|---:|---:|
| S8N (a = 0, b = 0) | 1.00 | 0.54 | 0.21 |
| b = +0.1 | 0.54 | 0.24 | 0.51 |
| b = +0.3 | 0.22 | 0.02 | 0.77 |
| b = +2 (about recency) | 0.07 | 0.00 | 0.93 |
| b = -0.3 | 0.58 | 0.92 | 0.03 |
| b = -2 (start frames) | 0.54 | 1.00 | 0.03 |
| a = 1 (one consensus ranking) | 0.63 | 0.68 | 0.12 |

### 6.2 Decision D1: inference-only retention probes on base S8N (val_seen-300, paired vs 155/300)

Same weights; only `sweep_retention.json` changes. The tilt is applied to every layer. D1, amended before any
probe ran, is: some arm moves SR by >= 3 pp.

| arm | SR | SPL | dSR [95% CI] | gained / lost | McNemar p | Holm p (3 arms) |
|---|---:|---:|---|---|---:|---:|
| balanced b = +0.1 | 54.00 | 50.27 | +2.33 [-2.0, +6.7] | 25 / 18 | 0.36 | 0.72 |
| recency b = +2 | 53.00 | 49.53 | +1.33 [-4.0, +6.7] | 33 / 29 | 0.70 | 0.72 |
| start b = -2 | **57.33** | 53.17 | **+5.67 [+0.7, +10.7]** | 39 / 22 | **0.040** | 0.12 |

D1 passed, through the start arm, so memory twins was queued. Readings, with care:
- Only one arm of three is significant, and not after the Holm correction, so the start effect needs its
  val_unseen confirmation, which was cancelled at 259/1,839 (§8).
- No arm is worse than S8N's own drift rule, not even pure recency. The policy tolerates large changes in
  which frames it keeps, and S8N's rule is not optimal for the policy it was trained with.
- The start rule helps most on short instructions: 67.0 vs 58.5 SR in the <= 21-word tercile, 49.5 vs 48.4
  in the > 30-word tercile.

### 6.3 The best rule depends on the policy (F-B step 125 + start tilt, val_seen-300)

| policy | SR | SPL | dSR vs same weights without tilt [95% CI] | gained / lost | McNemar p |
|---|---:|---:|---|---|---:|
| base S8N + start | 57.33 | 53.17 | +5.67 [+0.7, +10.7] | 39 / 22 | 0.040 |
| F-B step 125 + start | 59.33 | 54.61 | -2.00 [-7.0, +2.7] | 24 / 30 | 0.50 |

The rule that helps base S8N does not transfer to the RL policy, which was trained with S8N's drift rule.
This is the argument for learning the retention rule *jointly* with the policy rather than fixing it after
training. It is also the bar for memory twins: they must beat GRPO, not merely recover "keep the first
frames".

## 7. Memory twins (the candidate method): implemented, pilot only

**Rule.** Per attention layer l:

`priority_l(frame) = (1 - a_l) log p^l + a_l mean_l' log p^l' + b_l log(1 + s)`

- Parameters: w = (a, b), 16 numbers. w = 0 is S8N (tested: a zero tilt keeps S8N's frames).
- Code: `stream/sweep_retention.py` (`RetentionTilt`, saved as `sweep_retention.json` in every export and
  loaded by the evaluator) and `rl/memory_tilt.py` (`TiltLearner`).

**Twins.**
- Sample 0 of every final GRPO group acts with w + sigma eps.
- Its twin re-runs the same episode with the **same action random stream** (common random numbers), no forks,
  and w - sigma eps. The two trajectories are identical until the memory first changes an action.
- Twins are never trained on.
- sigma = 0.25 (consensus) and 0.05 (birth).

**Gradient and update.**
- ES gradient for w: g = mean over pairs of (R+ - R-) / (2 s) x eps / sigma, where s is one std of the step's
  rewards (floor 0.05 x 15).
- Adam ascent (lr_scale 0.2); w is identical on all ranks.
- Cost: one extra untrained rollout per group and no extra backward pass.

**Why it targets MemFork's failure on M5.** There, per-decision memory credit was drowned by action noise and
the learned retention collapsed onto the prior. Common random numbers remove the action noise from the memory
comparison, because a pair differs only through memory.

**Tests.**
- `tests/rl/test_memory_tilt.py` (8 tests): twins share their main's action stream; they diverge only after
  memory matters; perturbation determinism; the antithetic gradient recovers a linear return's slope;
  scheduling of twins for final groups only; the independent-seed ablation; config validation.
- `tests/rl/test_sweep_retention.py`: the tilted streaming state equals the offline reference and replays
  exactly; zero tilt keeps S8N's frames.
- The full CPU suite (`tests/rl`, `tests/memory_sweep`) passes: 247 passed on 2026-10-29 (job 48195324).

**Pilot (F-B + twins, steps 0-9, then stopped by the user to move twins onto plain GRPO after D2).**

| | value |
|---|---|
| twin pairs | 80 (8 per step) |
| pairs whose actions diverge | 72.5% (62.5-87.5% per step) |
| pairs whose success differs | 12.5% |
| first divergent decision (per-step mean over diverging pairs) | 23-49 |
| birth tilt b (mean over layers) | 0 -> -0.033 after 10 steps, monotone |
| consensus tilt a (mean over layers) | 0 -> -0.071 |

The ES signal exists: memory changes outcomes in a measurable share of CRN pairs. The birth tilt drifts
negative on 6 of 8 layers, toward episode-start frames, the direction the probe found for base S8N. Ten steps
say nothing about accuracy.

**Amended D2 for twins** (design §8, fixed before any twins result):
- Twins-GRPO must beat ActiveVLN GRPO **and** the G = 5 control, with McNemar p < 0.05 each on val_unseen at
  step 125. G = 5 matches twins' 5 rollouts per group while training on all of them.
- It must also not lose to F-B.
- Then a second seed of twins-GRPO and of GRPO; the claim needs the same sign on both seeds.

**Reproducibility note (GRPO resume).** GRPO 48037316 died at the step-10 save when the group's 1 TB /blue
quota filled; 159 GB were archived to /orange. The resume (48134830, from step 5) re-ran step 5 identically
(same rollouts, rewards and environment steps), but its update differs in the 4th digit of the grad norm
(2.5716 vs 2.5713, non-deterministic GPU reductions). Later steps therefore diverge as any re-run would. Runs
are reproducible in distribution, not bitwise, which is one more reason for second seeds.

## 8. Interruption and what remains

**2026-10-29 08:47:03 EDT.** UID 1013, a UF Research Computing staff account, cancelled every job of the user
and the login session was ended:
- the running start-tilt val_unseen probe (259/1,839 episodes);
- the queued twins-GRPO, G = 5 and G = 8 chains;
- about 50 stale jobs from 10-27 that were pending with never-satisfiable dependencies.

Slurm recorded no reason (no AdminComment, no maintenance reservation). Nothing on disk was lost: /blue is at
833 GB of 1 TB, and resumable checkpoints are on /orange. Changes made for the restart:
- `ops/hipergator/campaign/restart_after_cancel.sh` queues one stage at a time with `afterany` dependencies
  only, so a failed job cannot leave jobs pending forever.
- `eval_r2r.sbatch` exits at once when an export is missing.
- Every training run is followed by an archive job to /orange.

Remaining evidence, in order (about 62 GPU-hours on the 2-GPU group allocation):

| # | run | command | GPU h |
|---|---|---|---:|
| 1 | twins-GRPO train + val_unseen + val_seen-300 at step 125, then the start-tilt val_unseen probe | `STAGE=twins restart_after_cancel.sh` | 12.5 + 2 |
| 2 | G = 5 control | `STAGE=g5 restart_after_cancel.sh` | 12.5 |
| 3 | second seeds, twins-GRPO and GRPO | (seed in config) | 23 |
| 4 | twins without common random numbers (ablation of the central claim) | freeze v6, `STAGE=nocrn SNAPSHOT=<v6>` | 12 |
| 5 | learned per-layer rule (interpretation) | from `memory_tilt.json` / exports | CPU |

## 9. Paper story: status

The draft in design §9 is *one* contribution with two halves: what a bounded streaming memory keeps should be
learned with the policy, by RL, at the level of the memory rule, not of actions. Its four parts and their
evidence:

1. S8N, constant-memory streaming VLN: **done** (51.5 val_unseen on this evaluator).
2. The retention rule is policy-dependent: **supported on val_seen-300 only** (§6). The base + start effect has
   p = 0.04 and is not significant after Holm; its val_unseen confirmation is pending.
3. Memory twins beats GRPO: **no evidence yet** (§7, pilot only).
4. Negative results that motivate (3): **done**. F-B = GRPO at 2.2x interaction (§4). Event-placed forks are not
   more informative (§5). On M5, per-decision memory credit (MemFork) collapsed onto the prior.

If twins-GRPO does not beat GRPO, the paper loses its method, and (2)-(4) remain an analysis of RL with
bounded memory; it will be reported as such.

## 10. Files

- Code (this branch):
  - S8N rollouts and replay: `rl/sweep_memory.py`, `rl/sampling.py`, `rl/rollout.py`, `rl/episode.py`.
  - Event forks: `rl/rollout.py` (`fork.trigger`, `drift_by_position`).
  - Retention tilt: `stream/sweep_retention.py`.
  - Twins: `rl/memory_tilt.py`, `rl/train_grpo.py` (`memory_tilt` block).
  - Probes: `scripts/rl/make_tilt_checkpoint.py`.
  - Analyses: `scripts/rl/analyze_event_forks.py`, `scripts/rl/calibrate_drift.py`,
    `scripts/rl/check_s8n_replay.py`, `scripts/rl/s8n_analysis/`.
  - Configs: `configs/rl/*_s8n_r2r1k.yaml`.
- Cluster jobs: `ops/hipergator/` (environment, train / eval / archive / test jobs, snapshot freezing) and
  `ops/hipergator/campaign/` (the submission scripts actually used).
- Results: `results/s8n/` (tables, per-pair comparisons, retention and twins-pilot analyses) and
  `results/s8n/h1h2/` (H1/H2 outputs).
- Run artifacts (not in git): `/blue/yu954774.ucf/an221229.ucf/svln_runs/{rl,evals,probes}`, plus resumable
  checkpoints under `/orange/yu954774.ucf/an221229.ucf/svln_archive`.
