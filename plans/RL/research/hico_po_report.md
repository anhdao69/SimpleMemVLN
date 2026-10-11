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

## 4. Experiments — *pending*

## 5. Results — *pending*

## 6. Diagnostics (STOP credit, SNR, critic) — *pending*

## 7. Failure analysis — *pending*

## 8. Future work and publication roadmap — *pending*
