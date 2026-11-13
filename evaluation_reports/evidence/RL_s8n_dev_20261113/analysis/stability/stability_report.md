# Why Rewind RL from SFR is unstable, and how to make it stable (2026-11-12)

User: "Deeply analysis why the running is not stable for me. Then Deeply research/propose a solution to fix the
promblem for me. I want the SR is stable around 71-72."

## 1. The symptom

| same recipe (Rewind, mix1k config), same 1,000 prompts in the same order, from SFR | R2R val_unseen SR |
|---|---|
| mix1k, seed 20261010, local | 71.94 |
| gpu2 Rewind 1k, seed 20261010, gpu2 (nondeterministic rerun) | 70.15 |
| mix1k seed 2, seed 20261111, local | 67.70 |
| mean +- sd | 69.93 +- 2.13 |

The seed changes only the sampling (prompt order is fixed: `data.shuffle: false`).

## 2. Diagnosis

### 2.1 The two models really differ on val_unseen (not evaluation noise)
Paired over the same 1,839 episodes, 229 episodes are solved only by seed 1 and 151 only by seed 2. The 4.24 pp gap is
about 4 standard errors of evaluation noise (McNemar p 7e-5). A paired test only covers evaluation noise for two fixed
models, so single-run "significant" differences between recipes say nothing about training noise.

### 2.2 The training reward cannot tell them apart (underspecification)
| | train success (all) | steps 100-124 | 4/4 groups | 1/4 groups |
|---|---|---|---|---|
| seed 1 (71.94) | 0.743 | 0.769 | 0.44 | 0.12 |
| seed 2 (67.70) | 0.742 | 0.766 | 0.43 | 0.10 |

### 2.3 What differs is whether the agent finds the goal area in unseen scenes
Across 16 RL checkpoints (`val_behaviour_all.txt`), val SR correlates with oracle success +0.93,
with "came within 1 m of the goal" +0.85 and with the share of turn actions +0.82. It does not correlate with episode
length (-0.09). The two best models turn the most (mix1k 0.455; ActiveVLN 6k 0.585). Seed 2 vs seed 1: turn share
0.422 vs 0.455, came within 1 m 1,072 vs 1,126, oracle success 1,372 vs 1,445.

### 2.4 The seeds split late, along a direction the reward does not see
On the same prompts at each step, turn share (seed 1 - seed 2) is about 0 for steps 0-89. It is +0.033 for steps
90-99, +0.064 for 100-109 (10/10 steps) and +0.056 for 110-119 (10/10 steps). Training success did not improve with
it (-0.03). Seed 1's final checkpoint caught a late, reward-neutral rise in looking around, which pays off only in
unseen scenes.

### 2.5 About 90% of each run's weight change is seed noise
Cosine between weight changes from SFR (`deltas.py`, all 4.84 B changed parameters):
- seed 1 vs seed 2, same recipe and prompts: **0.096**;
- different recipes with the same seed and prompt order: 0.19-0.27 (mix1k vs rewind_path 0.258, vs ActiveVLN
  0.193; rewind_self vs grpo_sfr 0.269);
- all other pairs: 0.05-0.12.

Sampling noise therefore shapes a run's weights more than its data or recipe does. All 8 runs moved the same distance
(norm 0.61-0.63): Adam with a constant learning rate moves a fixed amount per step, whatever the signal.

### 2.6 Why each update is mostly noise (the recipe's mechanics)
1. **No trust region acts.** There is one on-policy update per step, so the ratio is 1 and the clip fraction is
   exactly 0 at every step. There is no KL term and no entropy term. Each step moves the policy by about lr, in
   whatever direction 32 trajectories point.
2. **Small batch.** 8 prompts x 4 samples = 32 main trajectories per update. Published LLM-RL recipes use 10-100x
   more.
3. **Group-std normalisation amplifies near-ties** (`advantage_noise.txt`). 44% of groups are all-success. Their reward
   std is 1.4 (stop distance differs by ~0.3 m), against 6.2 for mixed groups, yet they carry 43% of all advantage
   mass. With batch-level scaling they would carry about a third of that (|adv| per group 2.95 -> 1.06). So ~40% of
   every update ranks four successful trajectories by small stop-distance differences, and pushes everything they did
   (route, turning) with full weight.
4. **Constant learning rate to the end.** The last steps are as large as the first, so the final checkpoint is
   wherever the random walk is at step 125. The gpu2 8k run's checkpoints at 2k-6k sit flat at 70.2-71.0: a noise
   ball around the recipe's level.
5. **bf16 resolution.** lr 1e-6 x 125 steps is at bf16 precision. In the exports, 72% of weights are unchanged, 20%
   moved exactly 1 ulp and 7% moved >= 3 ulps. Whether a weight flips depends on whether its accumulated fp32 update
   crossed half an ulp, which can amplify small differences between runs. This is a likely contributor, not proven.
6. **Nondeterminism.** The same seed on another machine gave 70.15, so fixing the seed cannot reproduce 71.94.

Note: the code ignores the config key `norm_adv_by_std_in_grpo`. `algos.grpo_advantages` always divides by the group
std, and there is no LR scheduler.

## 3. What makes the SR stable at 71-72

The recipe's mean is 69.9 and its sd 2.1. "Stable at 71-72" needs the sd down to about 0.5 and the mean up by about
1.5. Picking the best of N seeds does neither (it is 69.9 + 0.85 sd for N = 3) and inflates the val_unseen number.

| # | change | targets | cost | evidence |
|---|---|---|---|---|
| P1 | **average the weights of K independent runs** (model soup) | the seed-specific 90% of each update (shrinks by sqrt K); averaged fine-tunes from one init generalise better under shift | K x training; no code | model soups 2203.05482, WiSE-FT 2109.01903, DiWA 2205.09739, rewarded soups 2306.04488, WARP 2406.16768 (merging up to 5 RL policies beat each one) |
| P2 | **tail averaging within a run**: export at steps 85/95/105/115/125, average the exports | late reward-neutral swings such as 2.4 | free (config `export_steps`) | SWA/LAWA 2306.03241; averaging replaces LR cooldown 2405.18392 |
| P3a | **batch-level reward std** for main advantages (not group std) | the near-tie amplification of 2.6.3 | small code change | Dr. GRPO 2503.20783; ScaleRL 2510.13786; "Tricks or Traps" 2508.08221 |
| P3b | **linear LR decay to 0** over the last ~40% of steps | final-step noise of 2.6.4 | small code change | 2405.18392, LAWA |
| P3c | 16 prompts per step (same 1,000 prompts, 63 steps) | gradient noise per step | config | ScaleRL (larger batch -> higher final performance) |
| P3d | KL or EMA anchor | behaviour drift (turning) | needs a reference copy; GPU memory is tight | WARP (EMA anchor), ProRL 2505.24864; mixed evidence |
| P4 | report mean +- sd over >= 3 seeds plus the soup; no best-seed picking on val_unseen | honest numbers | -- | 2504.07086 (RL gains within seed noise), D'Amour 2011.03395 |

### 3.1 Test running now (no training needed)
- `soups/soup_mix1k_seed12` = mean(seed 1, seed 2);
- `soups/soup_rewind4` = mean(seed 1, seed 2, rewind_path_sfr_mix1k (69.66), hard1k (67.70)).

Both are evaluated on R2R val_unseen (`soups/soup_evals.log`). If the 2-seed soup reaches >= 71 (its ingredients
average 69.8), averaging is the lever here.

### 3.2 Proposed next round
Implement P3a + P3b, set P2's `export_steps`, and train 2-3 seeds in parallel (local and gpu2). For each seed, report
the last export and the tail average. Then report the cross-seed soup. Adopt the recipe whose seeds land within about
+-0.5 and whose soup is >= 71.

## 4. Training-time fixes (user: "Is there any way to fix it in the training? so make sure the infer result will stable?")

Turning push per step = sum over trajectories of advantage x turn share (`turning_push.txt`):
- **Mixed groups push hard against turning in both seeds**: -49 (steps 0-89) and -15 to -18 (steps 90-124). In familiar
  training scenes, failed attempts are the lost ones that search around, so every turn in them is penalised. On unseen
  scenes, the models that search more find the goal more (section 2.3). The reward therefore trains the policy away
  from the behaviour that generalises. The upward pull most likely comes from rescue imitation (expert takeovers are
  62% turns). Where this tug-of-war ends is seed luck.
- All-success (near-tie) groups push turning only weakly (-1 to -2). Batch-std scaling halves their per-step noise
  (sd 0.20 -> 0.09), but they are not what moved turning. P3a is a noise cut, not the fix for this drift.

Training-time recipe to test with 2 seeds ("stable-v1"):
1. In-run weight averaging: an fp32 EMA / uniform average of the weights over the second half of the run, exported as
   the model (removes late swings such as seed 1's steps 90-124).
2. Linear LR decay to 0 over the last 50 steps.
3. Batch-level reward std for advantages.
4. Optional second arm: an anchor (KL to SFR, or a weight-space pull to an EMA anchor) to stop reward-neutral drift;
   G = 8 to halve sampling noise (2x rollout time).

Success criterion: the 2 seeds within 1 pp of each other, both >= 70.5, weight-change cosine well above 0.10.

## 5. STABLE-V1 launched (2026-11-12 ~10:40; user chose option 1 and "Stop clean 2k after 1k")

- Code (dev tree, uncommitted; defaults unchanged for every existing config):
  - `algos.grpo_advantages(..., scale=)` and `algorithm.adv_scale: group | batch`. Batch = (R - group mean) / the
    step's main-reward std, floor 0.05 R_max, the same scale the rescue branches already used.
  - `algos.lr_at` and `optim.lr_schedule: constant | linear_tail` with `optim.decay_steps`. Set on the ZeRO optimizer's
    param groups before each update; CPU Adam reads group['lr'].
  - Tests: tests/rl/test_algos.py (+2), passing locally and on gpu2. gpu2's train_grpo.py differs from local only in
    its DATA_ROOT default, as before.
- Configs: `rewind_grpo_sfr_mix1k_stable_a.yaml` (seed 20261212, local) and `..._stable_b_gpu2.yaml` (seed 20261313,
  gpu2 GPU 3, 6 helpers). Exports at steps 65, 70, ..., 125; the model is their uniform average (exports/avg_065_125).
- `stable_check.py` verifies one batch advantage scale per step (it fails, as it should, on mix1k seed 2) and the LR
  schedule from metrics.jsonl.
- Local: `stable_a_chain.sh` waits for the soup evaluations, then hard2k_chain.sh (smoke, 125 steps, step_0125 eval),
  then the average and its eval.
- gpu2: `svln/tools/stable/stable_b_chain.sh`. Clean 2k pauses at 125 and its 1k export is evaluated; the run stays
  paused with everything kept. Then seed B: smoke with stable_check, 125 steps, average, evals of avg and s125.

## 2026-11-13: STABLE-V1 seed A (local, mix1k prompts, 125 updates) - R2R val_unseen

| model | SR | SPL | OS | NE | turn share | path length |
|---|---|---|---|---|---|---|
| mix1k run 1 (old recipe) | 71.94 | 63.65 | 78.58 | 3.32 | 45.5% | 11.51 |
| mix1k run 3 (old recipe) | 67.70 | 60.47 | 74.61 | 3.78 | 42.2% | 11.00 |
| STABLE-V1 seed A, update 125 | 69.71 | 63.97 | 75.75 | 3.65 | 37.6% | 10.49 |
| STABLE-V1 seed A, average of updates 65..125 | 67.97 | 62.51 | 74.71 | 3.88 | 37.3% | 10.71 |

- Average vs its own endpoint: -1.74 SR (86 vs 54 discordant episodes, McNemar p 0.009): the tail average hurts. Drop it.
- Endpoint 69.71 = the old recipe's mean (69.93). Highest SPL of all runs, but the fewest turns and lower OS.
- Training success is the highest of the three runs (78.5-79.8% after update 50 vs 75-79%), so the recipe fits the
  training prompts better while the val policy looks around less; turn share on the guard's 20 prompts fell 0.02-0.06
  below the reference from update 60 on. Same pattern as before: val SR tracks OS/turning, not train success.
- The run-end check reported a false FAILED at update 44 (two advantages of ~1e-5 whose float32 rounding gave a scale
  of 5.3618 vs 5.3803). stable_check.py now ignores |adv| < 1e-3; re-run: STABLE CHECK OK. Fixed copy pushed to gpu2
  (sha-checked; old copy kept as stable_check.py.bak_20261113).
- Spread cannot be judged from one run; seed B (gpu2) finishes training around 17:00 on 11-13.
