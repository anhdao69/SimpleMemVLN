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

### 5.2 Refinement: GRPO + analytic STOP branch (`grpo_stop`)

The decisive test of the core claim is whether exact STOP credit helps **on top of the stronger GRPO
estimator**. `grpo_stop` keeps ActiveVLN GRPO exactly (group-normalised advantages, dynamic sampling,
micro-batch token-mean, same data and hyperparameters) and only replaces the sampled STOP branch at every
decision by its exact value under GRPO's own group baseline and scale:
c_logpi = A_i for non-STOP decisions (0 at a taken STOP), c_sigma(t) = (Q_S(t) − mean_g)/max(std_g + ε, 0.05·R_max).
With the group statistics held fixed this is the expectation of GRPO's own STOP-branch term (the
enumeration test with a constant baseline and scale covers it); it also gives exact STOP signal in
all-fail and all-success groups, where GRPO's own terms vanish or are amplified.

**Result (val_unseen, 1,839 episodes):** SR 54.98, SPL 49.71, NE 5.50, OSR 62.48, TL 10.33, 148 forced stops.

| Comparison | ΔSR | 95 % CI (SR) | McNemar p | ΔSPL | ΔOSR |
|---|---:|---:|---:|---:|---:|
| GRPO+STOP vs SFT | +3.43 | [+1.25, +5.60] | 0.0027 | +2.06 | +3.21 |
| GRPO+STOP vs GRPO | −1.90 | [−3.81, 0.00] | 0.058 | −0.75 | −2.39 |
| GRPO+STOP vs HiCo-PO v3 | +0.11 | [−1.96, +2.07] | 0.96 | −0.18 | −0.22 |

**GRPO+STOP does not outperform GRPO either.** Failure taxonomy vs GRPO: "passed the goal, stopped
outside 3 m" 136 vs 142 (the targeted failure, marginally fewer), but "never within 3 m" 482 vs 466, cap
hits 143 vs 131 and near misses 67 vs 54 — again the deficit is navigational, not STOP. During training it
was *ahead* of GRPO on the training prompts (steps 40–84: train SR 0.640 vs 0.613, OSR 0.715 vs 0.691), so
the val_unseen gap is not explained by slower learning.

### 5.3 Is the 2-point gap a method effect? — training-seed control

A second GRPO run differs from the first only in the seed (`configs/rl/activevln_grpo_r2r1k_seed2.yaml`;
same data order, checkpoint and hyperparameters).

| Model (val_unseen, 1,839) | SR | SPL | NE | OSR | forced STOP |
|---|---:|---:|---:|---:|---:|
| GRPO seed 1 | 56.88 | 50.45 | 5.31 | 64.87 | 133 |
| GRPO seed 2 | 57.37 | 51.82 | 5.30 | 64.22 | 86 |
| GRPO mean of 2 seeds | 57.13 | 51.14 | 5.31 | 64.55 | — |

Paired against GRPO seed 2:

| Model | ΔSR | 95 % CI | McNemar p | ΔSPL | ΔOSR |
|---|---:|---:|---:|---:|---:|
| GRPO seed 1 | −0.49 | [−2.56, +1.52] | 0.68 | −1.37 | +0.65 |
| HiCo-PO v3 | −2.50 | [−4.57, −0.38] | **0.023** | −1.94 | −1.52 |
| GRPO+STOP | −2.39 | [−4.40, −0.38] | **0.024** | −2.11 | −1.74 |

(Seed 2 vs SFT: +5.82 SR, p = 1e-7.) **GRPO is reproducible across training seeds, within half a point
of SR.** Both STOP-credit variants are about 2.4 points below each GRPO seed, and significantly so against
seed 2. The deficit is a method effect, not seed noise: at this scale, exact STOP credit (alone or on top
of GRPO) does not help and slightly hurts navigation. The training curves agree: on identical prompts the
two GRPO seeds track each other within 1.3 points in every phase, while GRPO+STOP falls behind in the last
40 steps (train SR 0.623 vs 0.647 / 0.650).

### 5.4 Second refinement: GRPO + forked sibling credit (GRPO+F, `grpo_fork`)

**Motivation.** §7 shows the residual gap is navigation: "never within 3 m" and cap hits, with STOP
behaviour unchanged by exact STOP credit. With a terminal-only reward, GRPO credits every decision of a
trajectory with the same group advantage, so the credit for an individual movement decision comes only
from averaging over trajectories. This is the plan's fork extension (+F), refined to keep GRPO intact.

**Estimator.** Per main trajectory i (GRPO group g, advantage A_i, decisions 0..n−1):

1. *Causal fork placement.* Before the episode, a target τ is drawn uniformly from [1, N*) with an
   independent generator. The fork point t is the first decision in [τ, τ + 8] whose first-token
   entropy H(π_T(·|h_t)) is at least the previous RL step's median. The rule uses only h_t and
   pre-drawn randomness, never a_t or later events.
2. *Forking.* The streaming memory before decision t (GDN recurrent + conv state, Window8 KV, position
   ledger) is cloned in O(state). After the main episode, the simulator prefix a_{0:t} is replayed
   deterministically (pose and frame verified bit-exactly) and M on-policy continuations are sampled from
   the restored memory.
3. *Credit.* With s_g = max(std_g + ε, 0.05·R_max):
   - main decisions before t keep GRPO's A_i (unchanged);
   - main decisions s ≥ t get (R_i − mean_b R_b)/s_g · 1/(1+M);
   - branch b's own decisions get (R_b − mean of its 1+M−1 siblings incl. the main)/s_g · 1/(1+M);
   - its prefix decisions get 0.

   Branch tokens get the parent's GRPO token weight, and dynamic sampling and micro-batch weights are
   GRPO's.

**Why it is unbiased (for a fixed scale and root baseline).** Given h_t, the main suffix and the M branch
suffixes are i.i.d. draws of the same conditional process. The sibling baseline of each is independent
of its own actions, so each suffix term is an unbiased estimate of the suffix part of ∇J. Averaging the
1 + M copies keeps that expectation, and only the variance changes. Without the 1/(1+M) averaging the
estimator is biased: the subtree after a fork would be counted 1 + M times. With several fork points the
weights multiply. `tests/rl/test_hico_estimator.py` verifies E[estimator] = ∇J by exact enumeration
(toy POMDP, two nested forks, M ∈ {1, 2}) and shows the un-averaged variant is biased.
`tests/rl/test_fork_rollout.py` checks three things on the production `run_episode`:
- the main trajectory is bit-identical with and without forking;
- branches start from the same frame and pose;
- teacher-forcing main prefix + branch suffix through a fresh session reproduces the branch log-probs
  exactly.

**What it changes vs GRPO.** For decisions after the fork, the baseline is the value of the *same
history* (estimated by siblings), not of the instruction. A movement choice at an uncertain decision is
credited by what happened to an alternative continuation from the same point. In all-fail and
all-success GRPO groups, where GRPO's own signal is zero, a sibling that succeeds or fails differently
still provides credit.

**Cost.** K = 1 fork and M = 1 branch per main trajectory. With forks spread uniformly over N*, a branch
averages roughly the second half of an episode, i.e. about +50–70 % policy steps, plus cheap simulator
replays and one more training sequence per branch. Branch and replay steps are logged separately
(`sampled_branch_env_steps`, `sampled_replay_env_steps`, discarded re-rolls included). Two comparisons:

- matched RL steps (125);
- matched environment interactions: the GRPO+F checkpoint whose cumulative policy steps equal GRPO's at
  step 125. Exports exist every 25 steps.

**Queue.** After the seed-2 GRPO run releases the GPUs, `scripts/rl/smoke_then_queue.sh` runs a 2-step
real-model smoke (`scripts/rl/check_fork_smoke.py`: finite gradients, log-prob agreement, forks present
and recorded). Only if the smoke passes does it start the full `hico_grpo_fork_r2r1k` run, followed by
v2a.

**Real-model smoke (2 steps, 4 prompts × G = 2, budget 60) passed:**
- 10 branches;
- trainer/rollout log-prob mismatch 0.0048 / 0.0031 (GRPO+STOP smoke: 0.0042);
- finite gradients (3.98 / 2.21);
- simulator replays bit-exact.

The full run `hico_grpo_fork_r2r1k` started 2026-10-13 11:26 UTC. Step 0:
- 0.94 forks per main trajectory;
- 73 % of branches carry a nonzero sibling advantage;
- the branch's first action equals the main's in 77 % of forks;
- branch policy steps +59 % over main steps;
- step time ≈ 400 s vs ≈ 300 s for GRPO;
- host peak 15.5 GB (cap 32 GB).

**Training (125 steps, completed 2026-10-14 07:16 UTC).**
- One restart: at step 59 another user's vLLM job took about 30–40 GB on GPU 1, and our rank 1 hit CUDA
  OOM in backward. The supervisor resumed exactly from the step-50 checkpoint, and steps 51–58 were
  re-run.
- 4,000 main rollouts plus 1,020 discarded by dynamic sampling (GRPO: 868 / 912). 6,161 training
  sequences, against GRPO's 3,796.
- Environment steps: 261 k main + 173 k branch + 135 k replay = **570 k, 2.2× GRPO**. Replays are
  simulator-only (no policy forward), so policy forward steps are 434 k (1.66×).
- Step time 485 s vs 272–289 s for GRPO. Part of the difference is the external GPU contention after
  step 59.
- Gradient norm (pre-clip) mean 3.77, max 16.1 (GRPO 2.36–2.59 / 3.5–3.6). Clipping at 1.0 bounds the
  update; there was no divergence.
- Across training, forks per main trajectory were 0.88–0.91 and branch success 0.59–0.66. The share of
  branches whose first action equals the main's rose from 0.61 to 0.77 as the policy sharpened.

**Results at step 125 (R2R val_unseen, 1,839 episodes):**

| Policy | SR | SPL | NE | OSR | cap hits | never ≤ 3 m | passed goal |
|---|---:|---:|---:|---:|---:|---:|---:|
| GRPO s1 | 56.88 | 50.45 | 5.31 | 64.87 | 131 | 466 | 142 |
| GRPO s2 | 57.37 | 51.82 | 5.30 | 64.22 | 84 | 495 | 125 |
| GRPO+STOP | 54.98 | 49.71 | 5.50 | 62.48 | 143 | 482 | 136 |
| **GRPO+F** | **55.57** | **50.59** | **5.35** | **62.37** | 122 | 497 | 122 |

Paired tests (`results/grpo_fork_step125_vs_*_comparison.json`):

| GRPO+F vs | ΔSR | 95 % CI | McNemar p |
|---|---:|---|---:|
| GRPO s1 | −1.3 | [−3.3, +0.7] | 0.22 |
| GRPO s2 | −1.8 | [−3.8, +0.2] | 0.087 |
| GRPO+STOP | +0.6 | [−1.4, +2.6] | 0.59 |
| SFT | +4.0 | [+2.0, +6.1] | 0.0002 |

**GRPO+F does not beat GRPO.** It is numerically 1.3–1.8 SR below both GRPO seeds (not significant) at
2.2× the environment steps.

**Matched environment steps.** The GRPO+F step-50 checkpoint (≈ 237 k env steps incl. branches and
replays, vs GRPO's 261 k at step 125) reaches SR 50.19, SPL 45.54, NE 6.23, OSR 60.09.
- vs GRPO s1 / s2 at step 125: −6.7 / −7.2 SR (p < 1e-8).
- vs SFT: −1.4 SR (95 % CI [−3.5, +0.7], p = 0.23).

So at matched simulator interaction GRPO+F is clearly worse. Part of this is a mid-training dip that
GRPO shares: on the val_seen-300 pilot, GRPO went 56.3 (SFT) → 52.0 (step 50) → 63.7 (step 125). A
GRPO s1 step-50 val_unseen evaluation is queued for a matched-RL-step comparison.

**Where it differs** (vs GRPO s2, `results/grpo_fork_step125_failure_analysis_vs_grpo_s2.json`):

| Reference length (actions) | n | GRPO s2 | GRPO+F |
|---|---:|---:|---:|
| < 40 | 216 | 67.6 | 64.4 |
| 40–59 | 891 | 64.4 | 59.9 |
| 60–79 | 597 | 48.2 | 50.1 |
| ≥ 80 | 135 | 34.8 | 37.0 |

GRPO+F gains on long episodes (60+ actions: +1.9 / +2.2; ≥ 10 m geodesic: +0.5 / +1.8) and loses on
short and medium ones (−3.2 / −4.5). The long-horizon gain is in the direction the per-decision credit
argument predicts, but no bin is significant on its own. The "never within 3 m" count, the failure
mode GRPO+F was meant to reduce, is unchanged (497 vs 466 / 495). Its beyond-window revisit rate is the
lowest of all policies (6.4 %, §7.5).

**Diagnosis: group-std amplification of sibling contrasts.** Replaying the saved coefficients of all
125 steps (1,000 groups):

| Group type | groups | mean \|branch coef.\| | root gradient mass | fork gradient mass |
|---|---:|---:|---:|---:|
| mixed | 613 | 0.28 | 65 k | 54 k |
| all-success | 330 | 0.81 | 29 k | **63 k** |
| all-fail (after 2 re-rolls) | 57 | 0.38 | 0 | 9 k |

(Gradient mass = Σ |coefficient| × decisions.)

- In mixed groups the fork terms behave as intended. The main suffix's credit has the opposite sign to
  its GRPO advantage in 32 % of forks, so credit is genuinely reassigned within the trajectory.
- In all-success groups the reward differences are small final-distance differences (R = 15(1 − d/3)).
  Divided by the small group std, they become the largest coefficients in the batch. The fork terms
  then carry 2.2× the root gradient mass there and are the source of the gradient-norm spikes (two of
  the three largest, 16.1 and 11.9, occur at steps where half the groups are all-success).
- The net effect is that GRPO+F spends a large share of each update on fine-tuning final distance on
  easy instructions. This coincides with its losses on short episodes.

The estimator stays unbiased for any fixed positive scale. The issue is the per-group scale: it weights
easy groups' tiny reward differences as heavily as success/failure contrasts (the Dr. GRPO critique,
amplified by forking).

### 5.5 Third refinement: batch-scaled sibling credit (GRPO+F-B) — *queued*

The only change from GRPO+F is `fork_scale: batch`: sibling contrasts (main suffix and branches) are
divided by one scale per RL step, s = max(std of all main rewards of the step, 0.05·R_max), instead of
s_g. Root (pre-fork) decisions keep GRPO's A_i exactly. Unbiasedness is unchanged, since s is a common
positive step size.

Replaying the saved GRPO+F coefficients with this scale (mean s = 5.7, range 4.1–6.6):
- all-success fork mass 63 k → 17 k;
- mixed 54 k → 50 k;
- all-fail 9 k → 1 k.

So the change removes the amplification and keeps the intended credit. Code:
`train_grpo.fork_prepare(step_scale=…)`. Test: `tests/rl/test_fork_rollout.py::test_fork_prepare_batch_scale`.
Config: `configs/rl/hico_grpo_fork_bs_r2r1k.yaml`, same seed as GRPO s1.

It runs after the GRPO+F step-50 (matched env steps) evaluation: a 2-step smoke, 125 steps and
val_unseen evaluation. The GRPO+F step-75 (matched wall-clock) evaluation and v2a follow.

**In training (steps 0–74, measured on the saved rollouts):** all-success fork mass 35.1 k (GRPO+F) →
9.3 k (GRPO+F-B); mixed 30.8 k → 29.8 k; all-fail 8.2 k → 0.2 k. Root GRPO mass is unchanged
(56 k vs 56 k). Gradient norm (pre-clip) over steps 0–50 is 1.97 mean / 3.37 max, against GRPO+F's
3.39 / 10.4 and GRPO's 2.4 / 3.6. Training-rollout success over steps 0–50 is identical to GRPO+F
(0.621). There is no val_unseen result yet.

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

**Replication on a second rollout distribution, and a correction.** The probe repeated on the 4,000 logged
GRPO+STOP rollouts (a different policy trajectory, with dynamic sampling)
(`results/stop_credit_probe_grpo_stop_rollouts.json`, `results/stop_probe_paired_grpo_vs_rloo.json`;
paired per-episode differences GRPO − RLOO on identical rollouts):

| Rollouts | all: GRPO / RLOO sum | paired diff (t) | all-success: GRPO / RLOO sum | paired diff (t) | mixed: paired t |
|---|---:|---:|---:|---:|---:|
| HiCo-PO v3 run | −75.7 / −2.5 | −0.018 (**−5.6**) | −136.7 / −56.9 | −0.062 (**−8.3**) | +0.75 |
| GRPO+STOP run | +49.9 / +97.1 | −0.012 (**−3.5**) | −42.1 / +1.5 | −0.036 (**−4.6**) | −0.38 |

The *relative* finding replicates: on both rollout sets GRPO gives the near-goal STOP logit significantly
less credit than the unbiased estimator, and the gap comes entirely from all-success groups. The *absolute*
statement above ("GRPO's estimate is negative, the true gradient ≈ 0") does **not** generalise: on the
GRPO+STOP rollouts the true near-goal STOP gradient is positive and GRPO's estimate is positive too, about
half as large. So GRPO under-credits stopping rather than pushing it down.

**Behavioural relevance is not established.** GRPO+STOP removes this under-crediting (its STOP branch uses
exact values; all-success groups get the floored scale). Yet its "passed the goal, stopped outside 3 m"
count is essentially GRPO's (136 vs 142), and it does not beat GRPO (§5.2). At this scale (1K prompts, 125
steps) the STOP under-crediting is a real estimator property, but it is not what limits success.

## 7. Failure analysis, cost and stability

All numbers on R2R val_unseen (1,839 episodes); files `results/*_failure_analysis_*.json`.

### 7.1 Outcome taxonomy

| Policy | SR | OSR | median NE (m) | actions | cap hits | stop at 3–5 m | stop at ≥ 5 m | STOP share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT | 51.55 | 59.27 | 2.75 | 91.3 | 114 | 167 | 614 | 13.0 % |
| GRPO | 56.88 | 64.87 | 2.10 | 96.6 | 133 | 140 | 522 | 9.5 % |
| HiCo-PO v3 | 54.87 | 62.70 | 2.33 | 97.6 | 140 | 157 | 539 | 9.3 % |
| GRPO+STOP | 54.98 | 62.48 | 2.32 | 98.2 | 148 | 150 | 535 | 8.6 % |

- **What RL fixes (SFT → GRPO, 275 gained / 177 lost):** mostly premature stops far from the goal
  (614 → 522 stops at ≥ 5 m; 145 of the gains were "far" failures under SFT). OSR rises by 5.6 points,
  so RL mainly makes the agent *reach* the goal region more often. "Passed the goal" failures barely move.
- **All three RL policies share the same failure mix.** Each RL run takes slightly longer episodes and
  hits the cap more often than SFT. The two HiCo-family policies trail GRPO on OSR (−2.2 / −2.4), which is
  the whole SR gap: their lost episodes vs GRPO are mostly "far" (89 / 83) rather than "passed the goal"
  (63 / 55).
- **Per-episode churn is large.** Between any two RL policies about 320–370 of 1,839 episodes flip
  (GRPO vs v3: 166 gained / 203 lost; v3 vs GRPO+STOP: 177 / 179). Net differences of ±35 episodes
  (≈ 2 SR) are therefore small relative to the churn, which is why the training-seed control (§5.3) is
  needed before attributing them to the estimator.

### 7.2 By difficulty (SR, %)

| Reference length (actions) | n | SFT | GRPO | v3 | GRPO+STOP |
|---|---:|---:|---:|---:|---:|
| < 40 | 216 | 62.0 | 66.2 | 63.4 | 63.0 |
| 40–59 | 891 | 56.0 | 63.5 | 60.2 | 60.5 |
| 60–79 | 597 | 45.6 | 48.2 | 48.7 | 47.2 |
| ≥ 80 | 135 | 31.9 | 36.3 | 33.3 | 40.0 |

| Geodesic start–goal (m) | n | SFT | GRPO | v3 | GRPO+STOP |
|---|---:|---:|---:|---:|---:|
| < 7 | 477 | 55.4 | 61.6 | 60.0 | 56.6 |
| 7–10 | 816 | 53.1 | 59.1 | 55.9 | 59.0 |
| 10–13 | 378 | 48.9 | 51.1 | 51.1 | 49.2 |
| ≥ 13 | 168 | 39.3 | 45.8 | 44.0 | 44.0 |

GRPO's advantage is concentrated in short/medium episodes (40–59 actions: +3.0 to +3.3 over both
variants). On the longest episodes GRPO+STOP is best (40.0 vs 36.3, n = 135, not significant). No bin
shows a consistent, interpretable gain from exact STOP credit.

### 7.3 Credit assignment and STOP calibration

On training rollouts, successful episodes put a median max σ of 0.61–0.65 on STOP inside 3 m. For
"passed the goal" failures the median is 0.024–0.045, and 34–42 % never exceed 0.01 (§6.1, §6.2). These
episodes are not near-misses of a well-calibrated stop rule: the policy hardly considers stopping where
it should. A better-weighted STOP gradient cannot fix this, since it scales a probability that is
already ≈ 0. This is consistent with the null effect of both STOP-credit variants on this failure mode
(GRPO 142, v3 141, GRPO+STOP 136).

### 7.4 Cost and stability (125 RL steps, 8 prompts × G = 4)

| Run | rollouts sampled | discarded (dynamic sampling) | env steps (kept) | grad-norm mean / max | restarts |
|---|---:|---:|---:|---:|---:|
| GRPO | 4,868 | 868 | 262.7 k | 2.36 / 3.62 | — |
| HiCo-PO v3 | 4,000 | 0 | 262.7 k | 6.67 / 10.34 | — |
| GRPO+STOP | 4,988 | 988 | 269.8 k | 2.36 / 3.42 | 2 (one manual for a helper change, one unit OOM; exact resume from step 20 / 30) |

HiCo-PO v3 used 18 % fewer rollouts (no dynamic sampling) and the critic adds < 10 s per step on CPU.
Its larger gradient norm reflects the batch-level scale (per-group normalisation in GRPO shrinks the
loss), not instability: clipping at 1.0 is active in all runs and no run diverged. Wall-clock is not
comparable across runs because the GPUs were shared with other users' jobs.

**Gradient clipping is always active.** In every run, all 125 updates have a pre-clip gradient norm
above the clip threshold of 1.0:

| Run | min | median |
|---|---:|---:|
| GRPO s1 | 1.60 | 2.33 |
| GRPO s2 | 1.77 | 2.58 |
| GRPO+STOP | 1.55 | 2.35 |
| HiCo v3 | 3.95 | 6.58 |
| GRPO+F | 1.64 | 3.07 |
| GRPO+F-B (steps 0–76) | 1.28 | 2.02 |

Every update is therefore g/‖g‖ before AdamW. This has two consequences for interpreting §5:

1. Unbiasedness of the raw estimator does not carry over to the applied update. Dividing by a random
   norm already biases it, and AdamW's elementwise normalisation removes the global scale. What
   matters is the *direction* quality of g/‖g‖.
2. Adding terms (the STOP branch, fork siblings) cannot add learning signal at a fixed learning rate. It
   only re-allocates a fixed per-step budget. For GRPO+F, fork terms carried 57 % of the gradient mass (126 k of 220 k, §5.4),
   so the GRPO component got less than half of each update. An extra term helps only
   if it improves the direction more than it dilutes the GRPO signal.

This applies equally to every variant here, and plausibly explains why three different unbiased
additions all landed 1.3–2.5 SR *below* GRPO rather than at it. It also points to the right control for
future estimators: either compare at matched *GRPO-component* step size (e.g. scale the learning rate
by ‖g‖/‖g_GRPO‖), or measure direction quality directly (cosine to a large-batch reference gradient on
held-out training prompts).

### 7.5 Memory behaviour

All estimators train the same streaming-memory policy (full backprop through the Gated DeltaNet state and
Window8 KV), so memory is not a variable between them. What can be measured without new GPU runs is how
the trained policies *use* memory, from the recorded agent positions in the evaluator journals
(`scripts/rl/analyze_memory.py`, `results/memory_analysis_step125.json`). A **beyond-window revisit**
means the agent returns within 0.5 m of a place it visited more than 8 decisions earlier, after moving
≥ 2 m away. Those observations have left the Window8 KV, so only the recurrent state can tell the agent
it has been there.

| Policy | SR | revisit episodes | SR \| revisit | SR \| no revisit | revisit steps / 100 fwd | never ≤ 3 m | … of which revisit | failures with revisit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT | 51.55 | 6.8 % | 21.6 | 53.7 | 1.89 | 40.7 % | 11.6 % | 11.0 % |
| GRPO s1 | 56.88 | 7.7 % | 27.7 | 59.3 | 2.21 | 35.1 % | 14.1 % | 12.9 % |
| GRPO s2 | 57.37 | 8.7 % | 19.4 | 61.0 | 2.92 | 35.8 % | 17.6 % | 16.5 % |
| HiCo v3 | 54.87 | 6.7 % | 21.0 | 57.3 | 1.80 | 37.3 % | 12.8 % | 11.8 % |
| GRPO+STOP | 54.98 | 7.3 % | 20.7 | 57.7 | 2.21 | 37.5 % | 13.6 % | 12.9 % |
| GRPO+F | 55.57 | 6.4 % | 19.7 | 58.0 | 1.93 | 37.6 % | 12.1 % | 11.5 % |

- **Looping is real but minor.** An episode that revisits an out-of-window place succeeds 19–28 % of the
  time, against 54–61 % without. Such episodes are only 7–9 % of the total and 11–17 % of failures.
- **Most navigation failures are wrong-route commitments, not memory loops.** Of the episodes that never
  get within 3 m, only 12–18 % ever revisit; the rest walk a plausible but wrong route without returning.
  Full 360° spins (≥ 24 consecutive turns) and immediate L/R reversals essentially never happen (≤ 0.05 %).
- **RL does not improve memory use.** Both GRPO seeds revisit slightly *more* than SFT (7.7 / 8.7 % vs
  6.8 %), as do the HiCo variants at about the same rate. RL's gains come from reaching the goal region
  more often (OSR), not from avoiding loops. Among SFT→GRPO gained episodes, only 11–13 % had a revisit
  under SFT.
- Consequence for method design: a memory-specific objective (e.g. penalising revisits) could address at
  most ≈ 15 % of failures. The fork estimator (§5.4) instead targets which branch to take at decision
  points, the dominant "never within 3 m" mode. As a side-effect check, GRPO+F has the lowest
  revisit rate (6.4 %) but the same "never within 3 m" rate as the other RL variants.

## 8. Future work and publication roadmap

What the evidence so far supports, and what a CVPR/ICML submission would still need:

1. **The GRPO per-group normalisation bias on STOP (§6.2)** is a measured, mechanistic finding: on
   two independent rollout sets GRPO gives the near-goal STOP logit significantly less credit than the
   unbiased estimator (paired t −5.6 and −3.5), driven entirely by all-success groups. Removing it
   (GRPO+STOP) did not change behaviour or SR at this scale, so on its own it is an analysis
   observation (a concrete case of the Dr. GRPO critique), not a headline result.
2. **Exact STOP credit** is a principled, cheap variance reduction (−22 % STOP-logit gradient std), but
   neither on its own (v3) nor on top of GRPO (`grpo_stop`) did it beat GRPO (§5).
3. **Navigation credit is the bottleneck.** After RL the dominant failures are "never within 3 m" (≈ 470)
   and cap hits (≈ 135), not stopping. A baseline only reduces variance; per-step credit for movement
   needs action values. The principled next step is the plan's **fork extension (HiCo-PO+F)**: snapshot the
   simulator pose and the streaming memory (GDN state + Window8 KV, O(1) copy) at high-entropy decisions,
   roll out the policy after each alternative action to the end of the episode, and use these unbiased
   Monte-Carlo action values in an all-action gradient (VinePPO-style; SeekVLN/C2PO-style branching in
   VLN-CE). Cost is extra rollouts, which can be paid for by reducing G (v4) — exactly the trade-off the
   critic was meant to enable.
4. **Recognition failures.** 27–34 % of "passed the goal" rollouts never put σ above 0.01 inside 3 m; no
   estimator fixes perception. Training-time geometric auxiliary heads on on-policy states (survey §7.10,
   P1/P2 heads with exact labels from the simulator) target this directly.
5. **Required controls for a paper:** matched-compute continued SFT/DAgger, a second RL seed for every
   finalist, STOP-bias calibration (B1), a supervised on-policy STOP loss (C1), and the full-backprop vs
   readout-only memory ablation (C2) before any claim about RL through the GDN memory (LANav 2607.18794 is
   close prior art).
6. **Scale:** all runs here use the 1K prefix (125 steps); the full 4K / 500-step schedule is the natural
   next scale for the best estimator.
