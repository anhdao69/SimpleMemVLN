# Why the prompt pool matters (2026-11-11)

> **2026-11-12 caveat (read first).** A second seed of mix1k (same prompts, config, code) scored 67.70 on R2R, the
> same as hard1k; mix1k's three draws are 71.94 / 70.15 / 67.70 (sd 2.1). The hard-pool loss is within seed noise,
> and seed 2 had mix1k's training-group mix, so the group-mix story below does not predict val SR. The log
> measurements stand; the causal claims about the pool are unproven. See decision_rule.md (2026-11-12).

User: "Think deeply, check the training log for me to see why the dataset matter here?" and "Analysis super deeply for
me the reason why? then try to choose the best 2k prompt to train for me".

Scripts in this folder (`why_pool.py`, `m12.py`, `m3.py`, `contrast_vs_p.py`, `val_stop.py`) read the training
rollouts (`runs/<run>/rollouts`, steps 0-124, main samples) and the R2R val_unseen journals. Pass rates per prompt come
from `hard2k/local_outcomes.tsv`, `rewind8k_outcomes_s790.tsv` (gpu2 Rewind run to step 790, built on gpu2, sha256
bf6090d7...) and `hard_outcomes.tsv` (the two hard runs).

## The results to explain (R2R val_unseen, 1,839 episodes, paired McNemar)

| Run | start | prompts | R2R SR |
|---|---|---|---|
| mix1k | SFR | mix pool rows 0-999 (= A+ rows 0-999, natural difficulty) | 71.94 |
| hard1k | SFR | hard pool rows 0-999 (top posterior P(mixed group), A+ rows 1,000-4,879) | 67.70 (-4.24, p 8e-5) |
| hard from mix1k | mix1k | the same hard prompts | 68.84 (-3.10 vs its start, p 0.004) |
| gpu2 Rewind 1k -> 2k | its 1k checkpoint | A+ rows 1,000-1,999 (natural difficulty) | 70.15 -> 70.96 (+0.8) |

The selection, not the source rows, does the harm: natural-difficulty rows from the same range help slightly when
continued; hard-selected ones cost 3-4 pp from either start. Surface features barely differ (hard vs mix1k: R2R
geodesic 10.06 vs 9.33 m, RxR N* 123 vs 112, en-IN share 0.75 vs 0.70, same scenes).

## What the hard selection did to the training signal

| | mix1k | hard1k (SFR) | hard (mix1k) |
|---|---|---|---|
| groups with 4/4 successes | 0.44 | 0.21 | 0.32 |
| groups with 1/4 | 0.12 | 0.25 | 0.17 |
| advantage mass on 4/4 groups | 0.36 | 0.16 | 0.27 |
| advantage mass on 1/4 groups | 0.15 | 0.29 | 0.19 |
| negative mass on "stopped at a wrong place > 5 m" | 0.22 | 0.34 | 0.28 |
| negative mass on the worse of good trajectories | 0.36 | 0.18 | 0.29 |
| rescue branches per main / imitation mass vs GRPO | 0.29 / 0.034 | 0.38 / 0.038 | 0.35 / 0.037 |

Rescue/imitation barely changes: it is not the driver.

### 1. Low-pass-rate prompts carry a weak (lucky) contrast

GRPO normalises every group's advantages to unit scale, so each group moves the policy equally, however informative
it is. Success - failure difference inside mixed groups, by the prompt's pass rate p estimated from the OTHER runs:

| p (other runs) | mix1k steps/d0, turn share | hard1k (SFR) | ActiveVLN 8k run (6,000 prompts) |
|---|---|---|---|
| < 0.50 | -0.98, -0.066 | -1.22, -0.045 | -1.85, -0.060 |
| 0.50-0.65 | -2.20, -0.078 | -1.12, -0.056 | -3.41, -0.074 |
| 0.65-0.75 | -2.69, -0.087 | -1.86, -0.066 | -3.19, -0.059 |
| 0.75-0.85 | -2.84, -0.084 | -2.08, -0.075 | -4.39, -0.082 |
| 0.85-0.92 | -0.54, -0.089 | -2.31, -0.063 | -4.47, -0.076 |
| >= 0.92 | -2.24, -0.117 | (none in pool) | -5.23, -0.094 |

On prompts the policy usually fails, the rare success is hardly more direct than the failures: the update mostly
reinforces whatever the lucky sample did. On prompts the policy usually solves, the one failure is clearly the worse
trajectory (much longer, more turning): a clean signal. Success and failure diverge in the first ~4-7 % of the
episode (the initial heading), so a noisy contrast steers the route choice itself.

Composition by p (other runs): mix1k pool 31 % below 0.65, 40 % 0.65-0.92, 29 % >= 0.92; hard pool 39 % / 61 % /
**0 %**. At equal p, hard-pool prompts give as clean a contrast as other prompts (ActiveVLN rollouts: -4.64 vs
-3.13 at 0.66-0.80, -4.40 vs -5.00 at 0.80-0.92): what hurt was the p distribution (many low, no always-solved).

### 2. Always-solved prompts are not wasted under this reward

R = 15 (1 - d/3) for a STOP within 3 m, so an all-success group still has non-zero advantages: they rank the stop
distance (corr(advantage, final distance) -0.54; within-group spread 0.24 m) and nothing else (corr with steps or
turn share ~0). The hard pool removed this stop-precision signal and replaced it with 1-of-4 groups.

### 3. What each hard run learned (R2R val_unseen behaviour)

| model | reached < 1 m of the goal | of those, failed | how | val turn share | turns before 1st move |
|---|---|---|---|---|---|
| SFR | 900 | 6.8 % | 60 stopped > 3 m away | 0.432 | 10.1 |
| mix1k | 1,126 | 5.0 % | 50 stopped elsewhere, 6 ran out | 0.455 | 7.9 |
| hard1k (SFR) | 1,125 | **8.6 %** (worst of 13 models) | **92 stopped elsewhere** | 0.413 | 8.4 |
| hard (mix1k) | **1,039** | 3.8 % | 37 | **0.403** | **7.3** |

- From SFR, the hard pool taught routes as well as mix1k (reaches the goal as often; wrong-route failures 270 vs
  260) but damaged goal recognition: it walks through the true goal (~14 steps within 1.5 m) and stops somewhere
  else. A third of its negative signal punished "stopped at a place that is not the goal" on prompts whose described
  endpoint is ambiguous (that is why they are hard), so the policy learned to distrust plausible endpoints,
  including the right one.
- From mix1k (already good at stopping), the dominant hard-pool signal punished long, turn-heavy searches: the
  policy turns less, starts moving earlier, commits to a heading sooner and reaches the goal 87 times less often.

## Selection rule that follows (select_clean_pool.py)

Keep the natural mix's always-solved share, drop the low-pass-rate prompts, fill with "usually solved, failed at
least once" prompts: A+ rows outside mix1k's 1,000; observed and solved at least once; posterior p >= 0.66; per
dataset 650 (R2R: all 637) from 0.66 <= p < 0.92 and the rest from p >= 0.92; scene-stratified; R2R/RxR interleaved
(1,000 + 1,000). Pool data/rl/clean2k.jsonl, sha256 2a8f97b6...; audit data/rl/clean2k.audit.json (posterior p
0.86, 61 / 56 scenes, R2R geodesic 9.08 m, RxR en-IN 0.71). Pass rates come from policies trained on 1-6k prompts,
so SFR will pass fewer: its group mix should land near mix1k's (all-success ~40-50 %).

Caveats: one seed per run; run-to-run spread of the same recipe is ~1.5-2 pp (gpu2 vs local on mix1k's prompts);
the mechanism is read from correlations in the logs, not from a controlled ablation. The clean-2k run is that test.
