# Clean-2k (rewind_sfr_clean2k): decisions fixed before training starts

Written 2026-11-11 ~18:00, before the run's smoke. Run: Rewind (mix1k's recipe) from SFR on data/rl/clean2k.jsonl,
250 steps, milestones 125 (1k prompts) and 250 (2k). R2R val_unseen, 1,839 episodes, paired (compare_evals.py).

At 1k (step 125), against mix1k (same start, recipe and steps; 71.94 / 63.65):
- SR >= 72.94 (+1.0): the cleaner pool helps; continue to 2k.
- -2.0 < delta < +1.0: no detectable difference from mix1k; continue to 2k (the 2k milestone is the user's target).
- SR <= 69.94 (-2.0, the size of the hard-pool losses): stop the run at once (manual stop, reason written; exports
  and the resumable state kept), diagnose with diagnose_pair.py and val_stop.py.

Mechanism checks at 1k (what the analysis predicts if it is right):
- failed after reaching < 1 m of the goal <= 5.0 % (mix1k) and well below hard1k's 8.6 %;
- val turn share >= 0.43 and reached-within-1-m >= 1,100 (no "turns less, commits early" drift);
- training: all-success groups 35-55 %, 1-of-4 groups <= 15 %.

At 2k (step 250): compare with mix1k (71.94) and with the gpu2 Rewind run's 2k (70.96, same 2k prompt budget, A+
rows 0-1,999; another machine, so unpaired). Any drift-guard stop or verify_eval problem overrides the above.

## 2026-11-11 17:45: moved to gpu2; mix1k replicate locally (user: "train the 2k in gpu2 and train the mix1k again")

- The local clean-2k chain was stopped during its smoke (nothing trained). The run is `rewind_sfr_clean2k` on gpu2,
  GPU 3 alone (one trainer rank, 6 rollout helpers; root ACL denies GPUs 0 + 2, the user forbids GPU 1):
  `W/svln/tools/clean2k/clean2k_chain.sh`, config `rewind_grpo_sfr_clean2k_gpu2.yaml` (the local config with paths
  mapped and helpers_per_rank 6; recipe keys identical), training code byte-identical to local (src/ checked).
- Locally: `rewind_sfr_mix1k_seed2` = mix1k's config with run.seed 20261111 (src/ identical to mix1k's frozen tree).
- Reading the 1k milestone: machine and seed spread are now measured, not assumed. Same-machine comparisons first:
  clean-2k 1k (gpu2) vs the gpu2 Rewind 1k (70.15, mix1k's prompts and seed, 2 ranks); mix1k seed 2 vs mix1k
  (local, seeds 20261111 vs 20261010). The thresholds above (+1.0 / -2.0) are applied against the mean of mix1k,
  mix1k seed 2 and gpu2 Rewind 1k (the recipe's level on mix1k's prompts), not against 71.94 alone.

## 2026-11-11 20:10: drift-guard stop at step 10 (false alarm), resumed 20:17 with a valid reference

- The guard (table reference: the gpu2 Rewind 8k run's mains on the same prompts) read +0.063 (steps 0-4) and +0.091
  (5-9) and stopped the run at step 10 (resumable/step_0010 kept).
- Not drift: the reference is biased low for this pool, which was selected for prompts that very run solved.
  Its turn share on clean prompts is 0.405 (first 80: 0.394; reference success 0.85-0.88), vs 0.433 on mix1k's
  prompts and 0.448 on the hard pool's; successful trajectories turn 0.38 vs 0.51 for failures. It also met these rows
  after 1k+ prompts of training. The run's own R2R turn share (0.486, 0.455) matches every run from SFR at steps 0-9
  (mix1k 0.461/0.483, hard1k 0.489/0.480, mix1k seed 2 0.489/0.484) and is falling.
- Resumed with `svln/tools/clean2k/clean2k_resume.sh` (same frozen tree, config, milestones). The guard now runs
  without --reference-table: same-window comparison with mix1k's own run from SFR (R2R mains, extended past step 124
  at ~0.466). It re-read windows 0-4 +0.025 and 5-9 -0.029. Old guard log: rewind_sfr_clean2k.drift_guard.table_ref.log.
- Lesson: a same-prompt reference table must not come from the run whose outcomes selected the pool.

## 2026-11-12 10:30: mix1k seed 2 = 67.70; amendment before the clean-2k 1k result (gpu2 at step 95)

- rewind_sfr_mix1k_seed2_s125 (mix1k's prompts, config and code; only run.seed 20261111): R2R val_unseen SR 67.70 /
  SPL 60.47, vs mix1k 71.94 (-4.24, McNemar p 7e-5: the paired test measures evaluation noise for two FIXED models,
  not training-seed noise). Same recipe, same prompts: 71.94 (local, seed 1), 70.15 (gpu2, seed 1), 67.70 (local,
  seed 2): mean 69.93, sd 2.13. hard1k (67.70) is inside this spread; its loss vs mix1k is not evidence against the pool.
- Training signal of seed 2 is the same as mix1k's (train success 0.742 vs 0.743, 4/4 groups 0.43 vs 0.44, 1/4 0.10 vs
  0.12; seed2_training_signal.json), yet it scored as hard1k: the group-mix differences in why_pool.md do not predict
  val SR. Seed 2 differs in turning: late-training turn share 0.432 vs 0.484, val 0.422 vs 0.455, reached < 1 m 1,072
  vs 1,126, oracle success 1,372 vs 1,445; failed-after-reaching 5.9 % (mix1k 5.0, hard1k 8.6).
- Amended reading of clean-2k at 1k (written before its result): baseline = 69.93 (3 draws); one new draw minus that
  mean has sd ~2.5. >= 72.9: evidence the pool helps; <= 65.5: stop (2 sd below); between: no detectable effect,
  continue to 2k. A pool comparison smaller than ~5 pp needs more seeds per arm.

## 2026-11-12 17:03: clean-2k at 1k (step 125) = 68.62 SR (gpu2, R2R val_unseen, verified 0 problems)

- SFR 62.59 -> 68.62 (+6.04, SPL +6.02). Against the gpu2 Rewind 1k on mix1k's prompts (same machine): -1.52 (p 0.17).
  Against mix1k (local): -3.32. Against the recipe's level on mix1k's prompts (mean of 3 draws, 69.93): -1.3.
- Amended rule (one new draw minus the 3-draw mean has sd ~2.5): 65.5 < 68.62 < 72.9 -> no detectable effect of the
  clean pool. The run stays paused at 125 (user: "Stop clean 2k after 1k"); gpu2 moves on to STABLE-V1 seed B.
