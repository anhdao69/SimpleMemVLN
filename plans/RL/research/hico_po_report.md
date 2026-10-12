# HiCo-PO for SimpleMemVLN — research report

Status: **IN PROGRESS** (sections marked *pending* are filled as runs finish).
Branch `RL_research` (from `RL_baseline` @ `96eedf1`). Companion notes:
`plans/RL/research/literature_update.md` (verified literature and novelty assessment),
`plans/RL/activevln_grpo_baseline_report.md` (the GRPO baseline this work must beat).

---

## 0. Summary of the setting

| | Value |
|---|---|
| Policy | SimpleMemVLN Window8 text policy, Qwen3.5-4B, epoch-2 SFT (`4391b1f6…`) |
| RL data | first 1,000 rows of ActiveVLN's fixed-order R2R 4K list (125 steps × 8 instructions), identical for every run |
| Baseline to beat | ActiveVLN GRPO, same data/steps: **R2R val_unseen SR 56.88 / SPL 50.45** (SFT: 51.55 / 47.64) |
| Evaluation | unchanged evaluator, Habitat 0.2.4, val_unseen 1,839 episodes, greedy, 500-action cap |
| Model selection | only on a 300-episode scene-stratified **val_seen** subset (`data/rl/r2r_val_seen_subset300.json`, seed 20261009); val_unseen is used once per final checkpoint |

---

## 1. Critical evaluation of `RL_technical.md` (and `RL_survey.md` §7)

What holds up:

1. **The v2 estimator is correct.** At a fixed history, E_a[(Q_S − b)∇σ + 1[a≠S](Q_a − b)∇log π(a)] = Σ_a Q_a ∇π(a). Verified here by exact enumeration of a toy POMDP with the *production* coefficient code (`tests/rl/test_hico_estimator.py`): relative error < 1e-6 for v2a, v2 and v3 under zero, constant, state and deliberately poor baselines and under G = 2 RLOO; two biased variants (double-counting STOP; crediting STOP with the episode return) fail by > 1e-2.
2. **The v1.1 → v1.2 correction is right.** A critic must only be a *baseline*; the analytic branch needs no continuation value at the stop step.
3. **The σ trie-shape condition (G3) holds for this tokenizer**: STOP's first token (id 48964) is unique among the four action strings, so σ_t = π_T(first token = STOP | h_t) is available in the forward that already scores the taken action.

What needed correction or refinement:

1. **It is variance reduction, not a new objective.** v2/v3 have the same expected gradient (same optimum) as REINFORCE/RLOO and, up to normalisation, GRPO. Any gain must come from lower-variance, more direct STOP credit per sample. The plan's framing ("exact STOP credit closes OS−SR") must be stated as a sample-efficiency hypothesis and tested as such (SNR probe, §6).
2. **Saturation.** ∇σ = σ(1−σ)∇logit. Where σ ≈ 0 near the goal the analytic term is as small as REINFORCE's; "no need to sample STOP" does not amplify a saturated logit. We therefore measure σ inside 3 m on-policy before attributing any effect to STOP credit (first smoke: mean σ = 0.065 at d < 3 m with 94 % of those decisions not stopping — not saturated).
3. **Full-vocabulary sampling, not the trie.** The GRPO baseline (faithful to ActiveVLN) samples the full vocabulary; we keep that so the only change versus GRPO is the estimator. σ is the full-vocabulary first-token probability; an invalid continuation after STOP (p ≈ 1 − p_eos) is treated as part of the STOP branch with value ≈ Q_S (p_eos|STOP is ≈ 1; logged).
4. **Reward.** `RL_technical.md` changes the reward to a success-gated SPL-like Q_S (v1) *before* changing the estimator. To isolate the algorithm from reward engineering we keep **ActiveVLN's reward** (Q_S(d) = 15·(1 − d/3)·1[d<3], exactly the stop return under that reward) in every HiCo run.
5. **Dynamic sampling.** Required off for unbiasedness (outcome-filtered groups). This also means HiCo uses **fewer environment interactions** than GRPO (4,000 vs 4,868 rollouts on the same 1K data) — reported, not hidden.
6. **Critic model.** GBDT is unavailable in the training env; we use a 2×64 tanh MLP on CPU (survey §7.4 allows it), fit only on the previous 4 RL steps (≥ 2,000 rows), predictions clipped to [0, R_max]. Rows of the current step never enter the fit (unit-tested), so b is independent of the current actions. Before enough rows exist (first RL step) the v3 run falls back to RLOO.
7. **Naming and claims** (literature update §7): drop "hindsight" (collides with HER/HCA/HCAPO); present the STOP branch as "integrating the STOP branch analytically because its return is a known function of simulator state" — an exact special case of all-action/expected policy gradients (Sutton 2000; Allen, Asadi et al. 2017; Ciosek & Whiteson 2018) and of optimal-stopping objectives, not a new theorem. Claim 3 (RL through GDN memory) is partly anticipated by LANav (2607.18794) and needs memory ablations to count.
8. **Survey §7.4 on Baisero & Amato** is imprecise: their bias result is about V(s) *replacing* a history value; a pure subtracted state baseline with Monte-Carlo returns is unbiased (Greensmith et al. 2004).

---

## 2. Method (refined)

Episode i, decisions t = 0..T_i−1, terminal-only reward R_i (ActiveVLN), γ = 1, tempered policy π_T (T = 1.2).
σ_{i,t} = π_T(first response token = STOP | h_{i,t}); Q_S(i,t) = 15·(1 − d_{i,t}/3)·1[d_{i,t} < 3] with d the geodesic
distance at the pose of decision t (exact: STOP neither moves the agent nor continues the episode).

**Loss (one on-policy update per RL step, μ = 1):**

  L = −(1/N_dec) Σ_i Σ_t [ ((Q_S(i,t) − b_{i,t})/s) · σ_{i,t} + 1[a_{i,t} ≠ STOP] · ((R_i − b_{i,t})/s) · log π_T(a_{i,t} | h_{i,t}) ]

- N_dec: number of decisions of all episodes in the RL step (all ranks); s = max(std of R over the step, 0.05·R_max).
- **v3 (main):** b_{i,t} = b(z_{i,t}), a geometric state baseline (features: d, d/d0, ℓ/d0, t/N\*, 1[d<3], cos/sin of the bearing error to the shortest path, d0, min_{k≤t} d_k/d0, missing-bearing flag), fit out-of-sample.
- **v2 (ablation):** b_{i,t} = RLOO mean of the other G−1 episodes of the instruction.
- **v2a (ablation):** RLOO and no STOP branch: c = (R − b)/s on log π of every action including STOP.
- Everything else equals the GRPO run: same checkpoint, data order, G = 4, T = 1.2, LR 1e-6 constant, AdamW, clip-norm 1.0, ZeRO-2, vision frozen, budgets.

Properties: unbiased for the tempered-policy gradient given on-policy unfiltered rollouts and an action-independent baseline (enumeration test); all-fail groups still carry signal (analytic terms at every d < 3 m state they passed through); no value head on images; the only extra compute is a CPU MLP fit per step.

---

## 3. Implementation

| File | Role |
|---|---|
| `src/qwen_vl/rl/hico.py` | stop values, RLOO, batch scale, coefficient rule, geometric features, out-of-sample MLP baseline, per-decision log π / σ extraction |
| `src/qwen_vl/rl/train_grpo.py` | `algorithm.estimator ∈ {grpo, rloo, rloo_stop, critic_stop}`; GRPO path untouched; HiCo coefficients, critic fit (rank 0, broadcast), critic state in checkpoints, metrics |
| `src/qwen_vl/rl/sampling.py`, `rollout.py`, `env_server.py` | σ_t logged at every decision (same random draws as before); per-decision geodesic distance, path length and route bearing (Habitat pathfinder) |
| `configs/rl/hico_v{3,2,2a}_*.yaml` | matched to `activevln_grpo_r2r1k.yaml`; only the estimator block differs (and dynamic sampling off) |
| `tests/rl/test_hico_estimator.py` | expected-gradient enumeration, loss gradient, coefficients, critic |

Validation: 28 RL unit tests; real-model smoke (σ rollout/trainer mismatch 3e-4 mean); exact resume including the critic state (re-run step identical up to bf16 grad-norm noise).

---

## 4. Experiments

### 4.1 Pilot (val_seen-300 only; no val_unseen tuning)

Same evaluator and protocol as the main comparison, on the 300-episode scene-stratified val_seen subset.
Gate: HiCo-PO v3 at step 50 must not be worse than GRPO at step 50.

| Checkpoint | SR | SPL | NE (m) | OSR | Forced STOP | Mean actions |
|---|---:|---:|---:|---:|---:|---:|
| SFT epoch 2 | 56.33 | 53.06 | 4.95 | 65.00 | 14 | 86.2 |
| GRPO @ 50 | 52.00 | 46.60 | 5.62 | 69.33 | 23 | 105.8 |
| GRPO @ 125 | 63.67 | 58.04 | 4.29 | 75.00 | 15 | 88.3 |
| **HiCo-PO v3 @ 50** | **58.33** | **54.11** | 4.97 | 69.33 | 17 | 92.5 |

GRPO dips below the SFT policy at step 50 (−4.3 SR, +20 % actions, more cap hits) before recovering by
step 125; HiCo-PO v3 at the same step is +6.3 SR above GRPO and +2.0 above SFT. The gate passed and the
v3 run continued to step 125. (n = 300; SE ≈ 2.9 SR points — a sanity gate, not a claim.)
Files: `plans/RL/research/results/pilot_valseen300_*.json`.

## 5. Results (R2R val_unseen, 1,839 episodes, identical protocol)

### 5.1 HiCo-PO v3 (analytic STOP + geometric baseline, RLOO-family, no dynamic sampling)

| Model | SR | SPL | NE (m) | OSR | TL (m) | Actions | Forced STOP | Rollouts used |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT epoch 2 | 51.55 | 47.64 | 5.80 | 59.27 | 9.89 | 91.3 | 114 | — |
| ActiveVLN GRPO (1K, 125 steps) | **56.88** | **50.45** | **5.31** | **64.87** | 10.63 | 96.6 | 133 | 4,868 |
| HiCo-PO v3 (1K, 125 steps) | 54.87 | 49.88 | 5.43 | 62.70 | 10.30 | 97.6 | 140 | 4,000 |

Paired statistics (same 1,839 episodes):
- **v3 vs SFT:** ΔSR +3.32 (95 % CI [+1.09, +5.60]), McNemar p = 0.0046; ΔSPL +2.24 (CI [+0.21, +4.28]).
- **v3 vs GRPO:** ΔSR −2.01 (95 % CI [−4.02, 0.00]), McNemar p = 0.061 (166 gained, 203 lost); ΔSPL −0.57
  (CI [−2.34, +1.26]).

**HiCo-PO v3 does not outperform GRPO.** It improves the SFT policy significantly and is at best on par
with GRPO, with a 2-point deficit that is borderline significant.

Failure analysis vs GRPO (`results/hico_v3_step125_failure_analysis_vs_grpo.json`): the deficit is
*navigational* — lower OSR (62.7 vs 64.9), more "far" failures among the flips (lost 89 vs gained 71),
slightly more near misses (157 vs 140 stops at 3–5 m); STOP behaviour ("passed the goal, stopped
outside 3 m": 141 vs 142) is unchanged. Together with §6 this indicates that the gains HiCo targets (STOP
variance, baseline variance) are small, while it gave up two things GRPO has: dynamic sampling (22 %
more rollouts, concentrated on hard instructions; HiCo's 596 all-fail-group episodes carried almost no
movement signal) and per-group normalisation.

### 5.2 Refinement: GRPO + analytic STOP branch (`grpo_stop`) — *running*

The decisive test of the core claim is whether exact STOP credit helps **on top of the stronger GRPO
estimator**. `grpo_stop` keeps ActiveVLN GRPO exactly (group-normalised advantages, dynamic sampling,
micro-batch token-mean, same data and hyperparameters) and only replaces the sampled STOP branch at every
decision by its exact value under GRPO's own group baseline and scale:
c_logpi = A_i for non-STOP decisions (0 at a taken STOP), c_sigma(t) = (Q_S(t) − mean_g)/max(std_g + ε, 0.05·R_max).
With the group statistics held fixed this is the expectation of GRPO's own STOP-branch term (the
enumeration test with a constant baseline and scale covers it); it also gives exact STOP signal in
all-fail and all-success groups, where GRPO's own terms vanish or are amplified.

## 6. Diagnostics (STOP credit, SNR, critic)

### 6.1 Estimator probe on identical rollouts (HiCo v3 run, steps 0–49, 1,600 episodes)

`scripts/rl/stop_credit_probe.py`: per-episode gradient w.r.t. the STOP first-token logit, summed over
decisions with d < 3 m, for each estimator computed on the **same** logged rollouts
(`results/stop_probe_hico_v3_steps0-49.json`).

| Estimator | mean | std | fraction non-zero |
|---|---:|---:|---:|
| GRPO (group-normalised) | −0.029 | 0.432 | 0.61 |
| RLOO (v2a) | −0.022 | 0.456 | 0.61 |
| RLOO + analytic STOP (v2) | −0.013 | **0.365** | 0.63 |
| critic + analytic STOP (v3) | −0.008 | 0.383 | 0.63 |

- The analytic STOP branch lowers the STOP-logit gradient std by **20 %** relative to RLOO on identical
  rollouts; the geometric baseline does not lower it further. Means agree within noise (±0.02), as
  unbiasedness requires.
- The expected STOP-logit gradient inside 3 m is slightly **negative**: under ActiveVLN's reward
  (15·(1 − d/3)) moving closer inside the radius still pays, so approach states dominate the few
  "should have stopped" states.
- Passed-the-goal-but-failed rollouts (98): median max σ inside 3 m = **0.049**; 27 % never exceed 0.01,
  44 % exceed 0.1. Much of this failure mode is recognition (the policy never wanted to stop), which no
  credit-assignment estimator can fully fix; successful rollouts reach median max σ 0.53.
- Geometric baseline: held-out R² 0.52–0.81 over steps (logged per step) — geometry explains most of the
  return variance, but per-episode return noise (sampled movement decisions) dominates the STOP-logit
  gradient variance.

### 6.2 Full run (steps 0–124, 4,000 episodes) and a GRPO normalisation bias

| Estimator (same rollouts) | mean STOP-logit grad (d < 3 m) | std | |mean|/s.e. |
|---|---:|---:|---:|
| GRPO | **−0.0189** | 0.396 | **3.0** |
| RLOO (v2a) | −0.0006 | 0.422 | 0.09 |
| v2 (analytic STOP) | +0.0035 | 0.330 | 0.67 |
| v3 (analytic STOP + geometric baseline) | +0.0086 | 0.325 | 1.68 |

- HiCo lowers the STOP-logit gradient std by 22–23 % vs RLOO on identical rollouts.
- The unbiased estimators agree that the true expected STOP gradient inside 3 m is ≈ 0, but **GRPO's
  estimate is significantly negative** (3 standard errors). Decomposition by group type:

  | Group type | Episodes | mean group reward std | GRPO STOP push (sum) | RLOO/s (unbiased) |
  |---|---:|---:|---:|---:|
  | mixed outcome | 2,108 | 5.64 | +61.0 | +54.4 |
  | all success | 1,296 | 2.02 | **−136.7** | −56.9 |
  | all fail | 596 | 0 | 0 | 0 |

  **Mechanism.** GRPO divides by the *per-group* reward std. In all-success groups the only reward variation
  is the closeness term 15·(1 − d/3), whose low spread makes the normalised advantages large, so "keep
  approaching, do not stop yet" is amplified 2.4×. Net, GRPO pushes the STOP logit near the goal down
  (−75.7 vs −2.5 unbiased). This matches GRPO's observed behaviour: more actions, more cap hits, an
  unchanged "passed the goal" failure count, and the mid-training val_seen dip (§4.1). It is the per-group
  std normalisation critique of Dr. GRPO (Liu et al. 2025), made concrete for stopping. HiCo-PO uses one
  batch-level scale, which removes it; the v2a ablation isolates exactly this change.

## 7. Failure analysis — *pending*

## 8. Future work and publication roadmap — *pending*
