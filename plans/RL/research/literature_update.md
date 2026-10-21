# HiCo-PO: literature update and novelty assessment

*Date: 2026-10-11. Scope: RL fine-tuning of the SimpleMemVLN policy (Qwen3.5-4B, 24 Gated DeltaNet + 8 gated full-attention layers, project-specific Window8 KV eviction with a persistent GDN state) on R2R-CE. This note extends `plans/RL_survey.md` (§1, §3, §6, §7.9, §11) and `plans/RL_technical.md` (§2, §5–§10, App. A/E). It is grounded in the completed baseline in `plans/RL/activevln_grpo_baseline_report.md`: SFT 51.55 SR → ActiveVLN-style GRPO on the 1K R2R prefix 56.88 SR (+5.33, McNemar p = 4.7e-6). In that baseline, "passed within 3 m, then stopped outside" went 141 → 142 and OSR−SR went 7.72 → 7.99.*

**How items were checked.**
- **V**: read on the arXiv abstract or HTML page (or the venue page) for this note.
- **V-abs**: only the abstract was read; method details come from secondary reads and are flagged where they matter.
- **K**: canonical paper cited from knowledge and not re-fetched. These are well-known results; the citation is safe but page numbers were not checked.
- **UNVERIFIED**: no primary source was read. Do not cite these numbers.

All arXiv IDs in RL_survey.md that this note relies on exist and match their topics. Corrections are listed in §8.

---

## 1. Key findings

1. **Claim 1 (analytic STOP branch): the mathematics is not new, but the instantiation appears to be.** In expectation, the estimator equals at least four established constructions:
   - the all-actions form of the policy gradient (Sutton et al. 2000), applied to one action;
   - the option-critic termination gradient (Bacon et al. 2017), with an exact termination value in place of a learned one;
   - the per-time objective of Deep Optimal Stopping (Becker et al. 2019): exact stop payoff, sampled continuation;
   - a REINFORCE control variate whose action-dependent part is exact for STOP (the Q-Prop / Stein control-variate family).

   No VLN, VLN-CE or VLM-navigation RL paper found gives STOP an exact counterfactual or analytic gradient. Every verified VLN-RL paper either broadcasts an outcome scalar, shapes a hand-designed stop reward (Embodied-Navigator), or penalises premature STOP per state against GT labels (GroundingVLN/GEAR). The LLM analogues (S-GRPO, AnytimeReasoner) obtain per-prefix exit values only by **generating extra answers**, and in AnytimeReasoner stopping is an exogenous budget, not a policy action. **Risk: medium.** Phrase it as an estimator that exploits a structural property of embodied navigation, not as a new policy-gradient theorem.
2. **Claim 2 (privileged geometric state baseline): low technical risk, low novelty.**
   - Unbiasedness of a latent-state baseline under Monte-Carlo returns is textbook: Greensmith et al. 2004 §5.2; Wu et al. 2018 §5 even uses a privileged goal-location baseline in a POMDP.
   - The closest 2026 precedents are Le Critique (2608.16739; privileged-value baseline with λ = 1 for LLM RL) and Informed Asymmetric AC (2509.26000, ICML 2026).
   - No VLN-RL paper uses simulator geometry in a value function or baseline; every verified learned critic reads the policy's own features. "First in VLN" is plausible but small. **Risk: high as a headline claim.** Present it as a component whose value is shown only by the G = 2 result (v4) and a measured variance reduction.
3. **Claim 3 (on-policy full-episode RL through a hybrid GDN memory in a VLM navigation policy): partly anticipated.**
   - LANav (2607.18794, July 2026) trains a **Gated DeltaNet / linear-attention navigation policy with DAgger → PPO** (ObjectNav, small SigLIP-fed model, not a VLM). It shows GDN > Transformer > GRU.
   - DD-PPO trains recurrent (LSTM) navigation policies on-policy.
   - MiniMax-M1, Kimi Linear, Ring-linear and HARTS (2608.28158) run RL on hybrid linear-attention LLMs; HARTS backpropagates through linear-attention state handoffs in multi-turn text agents.
   - What remains, as far as we found: a **VLM** VLN policy with a **persistent GDN state** trained by closed-loop RL with **full-episode BPTT**.
   - The ActiveVLN port already shows this works (+5.33 SR). Without a detach/readout-only ablation, this is engineering, not a claim. **Risk: medium-high.**
4. **The baseline result matches the claim-1 motivation, but does not prove credit is the bottleneck.** Outcome GRPO left the passed-goal-then-stopped-outside failures untouched (141 → 142). An unbiased estimator only reduces variance; it cannot fix a perceptual inability to recognise the goal. Before relying on claim 1, measure σ_t on the d < 3 m steps of those 141 episodes (§6.1).
5. **Must-compare internal controls** (§7): RLOO (v2a), STOP-bias calibration (B1), an on-policy supervised STOP loss (C1), and an Embodied-Navigator-style stop-reward shaping. These four directly answer "isn't this just stop supervision or calibration?".

---

## 2. Verification of the VLN/VLN-CE RL papers the survey relies on

Mode codes: **Online-CL** = closed-loop multi-turn rollouts in a simulator; **Per-state** = offline or per-state samples scored against GT/oracle actions or paths. Before→after values are R2R-CE val-unseen SR/SPL.

| arXiv | Paper (first author) | Check | RL mode | Reward / STOP handling | Critic | SR/SPL before → after | Obs |
|---|---|---|---|---|---|---|---|
| 2509.12618 | ActiveVLN (Zekai Zhang, 09/2025) | V | GRPO, Online-CL (Habitat HTTP server), G = 4 | Printed Eq. 8 reads 15·I(succ)·d/3 (typo). The released code uses 15·1[succ]·(1−min(d/3,1)) (verified in RL_technical §2.1). No stop-specific credit | none | 38.5/33.7 → 50.1/43.7 (w/o dynamic sampling 48.1/40.6) | mono RGB, Qwen2.5-VL-3B |
| 2506.17221 | VLN-R1 (Zhangyang Qi, 06/2025) | V | GRPO, Per-state | time-decayed GT-action match; no stop term | none | 7B 24.9/17.5 → 30.2/21.8 | mono RGB video |
| 2603.09740 | SACA (Haoyuan Li, 03/2026) | V | GRPO + step-aware contrastive loss; suffix resampling (online implied) | outcome on STOP + auditor process score; positive actions from the shortest-path signal (privileged) | none | SFT 52.8 → GRPO 54.1 → SACA 60.3/55.1 | mono RGB, LLaVA-Video-8B |
| 2602.06356 | BudVLN (Gang He, 02/2026) | V | GRPO on probe successes + oracle SFT on failures | 1(succ)(2 + λ·SPL) − C·d_remain; premature STOP is a failure trigger only | none | 57.0/50.5 → 57.6/51.1 (rectification only 57.2/50.1) | mono RGB |
| 2609.37353 | SeekVLN / C2PO (Zhimin Wang, 09/2026) | V | **PPO**, decision level, γ = λ = 1; **clones sim state + history + context** for SEEK-vs-NAV branches | counterfactual Δd reward on SEEK decisions only; outcome 1 + 0.2·SPL / −0.5 | separate SFT-initialised VLM value model (policy-style inputs) | 61.0/55.9 → 67.5/61.4 (RL step +6.5) | mono RGB, Aux-Think base |
| 2608.17512 | Embodied-Navigator; method TAMP-Nav (Hongyan Feng, 08/2026) | V | GRPO, Online-CL, G = 8 trajectories × M = 4 candidate actions per step | local reward includes **0.8·r_stop** on geodesic d_t: +1 if STOP at d < 1.5; −1 if STOP at d > 3; ±0.5 in the 1.5–2.5 m band. A = z(R_global) + z(R_local) | none | 55.7/50.3 → 66.2/58.8 | **4 views (360°) + GT depth**, Qwen2.5-VL-7B |
| 2608.30935 | LightNav-0 (Shaoan Wang, 08/2026) | V | GRPO-style, Online-CL, G = 8, KL 0.01 | (1 + nDTW)·1[succ] + Gaussian distance term − timeout penalty | none | final 68.5/62.8; **no before/after-RL ablation** | mono RGB, Qwen3-VL-4B |
| 2606.03682 | GN0 (Xinhai Li, 06/2026) | V | SFT → DAgger → DAPO; trajectory prediction scored against GT (effectively per-state) | no success term; STOP rule only in DAgger labels | none | VLN-CE final 67.7/63.4; stage ablation only on GN-Bench | single-view RGB (+BEV option) |
| 2602.00222 | MapDream (Guoxin Lian, 01/2026) | V | GRPO, Per-state | longest correct prefix + format | none | ablation 42.2 → 45.6; headline 59.8/54.4 (gap unexplained) | mono RGB |
| 2509.10884 | Nav-R1 (Qingxiang Liu, 09/2025) | V | GRPO, Per-state (path vs GT path) | format + understanding + path/endpoint similarity | none | no RL ablation; 72.5/68.8 | RGB-D + point cloud |
| 2512.20940 | ETP-R1 (Shuhao Ye, 12/2025) | V | GRPO, Online-CL on a topological graph | 1(d < 1.5) + SPL − d/6 | none | DAgger 63/54 → 65/56 | pano RGB-D, ~0.5B graph model |
| 2609.03906 | Macro-action PPO (Shuhao Ye, 09/2026) | V | PPO on frontier macro-actions | terminal Success + SPL + nDTW | action-aware value head on the **policy's shared backbone** | DAgger 65.1 → GRPO 66.8 / **PPO 68.1/57.3** | pano RGB-D, 0.5B |
| 2607.20785 | Robostral Navigate (Bounhar, 07/2026) | V | CISPO + group advantage, async online, 35K hard tasks | −max(2, d_geo): flat inside 2 m, so STOP is pushed implicitly | none stated | 73.40 → 77.43 SR | mono RGB, 8B (closed) |
| 2607.13461 | JOP-VLN (Qingrong He, 07/2026) | V | GRPO + IL mixing (CHORD), per-state at DAgger states | action-chunk match + pose terms | none | DAgger 68.1 → pure GRPO 67.5 → 69.9/64.9 | mono RGB, Qwen3-VL-8B |
| 2609.18581 | GroundingVLN / GEAR (Kailing Li, 09/2026) | V | GRPO, Per-state (16K anchors) | **premature STOP = −1** against the GT label; P_stop = 0.5 | none | 57.2/49.5 → 69.9/64.1 | stitched L/F/R **RGB-D**, Qwen3.5-4B |
| 2511.17097 | Progress-Think (Shuo Wang, 11/2025) | V | GRPO, Per-state | prefix match + progress-length | none | ablation 41.4 → 43.8; headline 60.1/53.6 | mono RGB |
| 2506.09839 | **OctoNav** (Chen Gao, 06/2025); model OctoNav-R1 | V | per-state GRPO → online A2C | A2C: +1 at goal, else −Δd | **linear head on pooled policy hidden states** | no R2R-CE; OctoNav-Bench 17.0 → 19.4 | RGB |
| 2602.12351 | LongNav-R1 (Yue Hu, 02/2026, RSS'26) | V | Online multi-turn, **ObjectNav**; HAPO | dense −Δgeo, terminal 2.5·SPL; LOO kernel-regression baseline over time | critic-free (reports its PPO value head was unstable) | ObjectNav 64.3 → 73.0 | RGB, Qwen3-VL-2B |
| 2603.15370 / 2609.00920 | NavGRPO / VerNav | V | discrete R2R only | — | — | not R2R-CE | pano |

**What this means for HiCo-PO.**
- Only five verified papers are **online closed-loop RL of a monocular VLM on R2R-CE**: ActiveVLN, SACA, BudVLN, SeekVLN and LightNav-0. Robostral (closed 8B) is a sixth, but has no code.
- Embodied-Navigator is multi-view + depth. GEAR, GN0, Nav-R1, VLN-R1, MapDream, Progress-Think and JOP-VLN are per-state. None of these is a protocol-matched RL comparator.
- **No verified VLN-RL paper gives STOP counterfactual or analytic credit.** SeekVLN's counterfactual branch applies to SEEK only. Embodied-Navigator's r_stop is a hand-set step reward on the *executed* action, and its band design rewards "not moving away" in 1.5–2.5 m rather than valuing the option to stop.
- **No verified VLN-RL critic uses privileged geometry.** The critics read OctoNav's pooled hidden states, Macro-PPO's shared backbone, and SeekVLN's SFT-initialised VLM.
- **The PPO-vs-GRPO evidence on navigation is mixed.** Macro-PPO reports PPO > GRPO by 1.3 SR. LongNav-R1 reports an unstable PPO value head. The survey's lesson 6 ("PPO with a critic beats GRPO") should be softened accordingly.

---

## 3. Prior art for claim 1: exact or analytic credit for a terminal STOP action

### 3.1 What the estimator is

From RL_technical §7.3, per visited decision:

ĝ_t = (Q_S(t) − b)·∇σ_t + 1[a_t ≠ S]·(R − b)·∇log π(a_t | h_t)

Its expectation over a_t is Σ_a Q(h_t, a)∇π(a | h_t). Appendix E's enumeration confirms the estimator is unbiased (error 2.8e−10).

Three equivalent readings follow. Each maps the method onto existing work.

- **A control variate with an exact action-dependent part.** Use c(a) = Q_S·1[a = S] + b·1[a ≠ S] as an action-dependent baseline and add back its analytic expectation Σ_a π(a)c(a)∇log π(a). The result is exactly ĝ_t: when a = S the sampled term vanishes because R = Q_S. This is the structure of Q-Prop and the Stein/action-dependent control-variate family. Tucker et al. (2018) showed that the gains of *learned* action-dependent baselines were largely a mirage, because the learned Q is inaccurate. Here Q_S is exact, which is the one setting in which their critique does not apply.
- **The all-actions policy gradient restricted to one action.** Sutton et al. (2000) state the theorem in all-actions form, Σ_a ∇π(a|s)Q(s,a). Mean Actor-Critic (Allen, Asadi et al., arXiv 2017) and Expected Policy Gradients (Ciosek & Whiteson, AAAI 2018) use it with a *learned* Q over all actions. Nauman & Cygan (NeurIPS-W 2022 / ICML 2023) analyse when all-action and many-action sampling reduces variance. HiCo-PO integrates **only the STOP branch**, and its Q is **exact**, so no critic is needed for that branch.
- **An optimal-stopping or termination gradient.** In expectation the stop coefficient is (Q_S − Q_C)·∇σ_t, where Q_C is the continuation value.
  - This is the option-critic termination-gradient form, ∝ −∇β·(Q_Ω − V_Ω) (Bacon, Harb & Precup, AAAI 2017), with an exact termination value in place of a learned one.
  - It is also the per-time objective of Deep Optimal Stopping (Becker, Cheridito & Jentzen, JMLR 2019): E[g(n, X_n)·F(X_n) + g(τ_{n+1}, X_{τ_{n+1}})·(1 − F(X_n))], with exact stop payoff g and a sampled continuation payoff (verified on ar5iv 1804.05394).

  **The key difference** is that in classical optimal stopping the state process is exogenous: stopping does not affect X, so the continuation can be evaluated on every path, and training is backward induction. In navigation the trajectory is generated by the same policy's movement actions. HiCo-PO therefore keeps a REINFORCE term for movements and trains all decisions jointly, on-policy.

### 3.2 Termination and early-exit learning in LLMs and VLN

| Work | Check | What it does | Same as HiCo-PO | Different |
|---|---|---|---|---|
| S-GRPO (Dai et al., 2505.07686) | V | One reasoning trace; exits are **forced** at several serial positions and an answer is decoded at each; correct earlier exits get decaying higher rewards | value of stopping at many prefixes of one rollout | each exit needs extra **generation** and a verifier; the reward is shaped for brevity; not an unbiased gradient of the original objective |
| AnytimeReasoner / BRPO (Qi et al., 2505.13438, NeurIPS'25) | V | Truncate thinking at budgets b_j ~ p_B; summarise and verify at each; per-token return Σ_{j≥j_t} P_j·r(z_{≤b_j}); BRPO baseline interpolates prefix scores and the group mean | dense credit from counterfactual "stop here" outcomes | **stopping is an exogenous budget prior, not a policy action**; prefix values cost extra decoding |
| MRT (Qu et al., 2503.07572) | V-abs | Dense "progress" bonus = change in the likelihood of eventual success per block | per-step value-of-stopping style bonus | estimated, not exact; shaping changes the objective |
| ESTAR (2602.10004), Learning When to Stop (2511.21581), PUMA (2605.17672) | V-abs | Learned stop tokens or stop heads trained with RL and compute-aware rewards | learned stop decision | rewards and gradients unspecified in abstracts; no exact counterfactual |
| DASH (2607.00482) | V-abs (subagent) | segment credit for "when to stop" in reasoning | stop credit | text, estimated |
| L2Stop (Xiang et al., EMNLP-F 2020, 2009.13112) | V | Separate STOP policy module (urban VLN, Touchdown); +6.89 SED | treats STOP specially | supervised, pre-VLM |
| Embodied-Navigator r_stop (2608.17512) | V | hand-set stop reward from geodesic d_t on sampled candidate actions | geometry-based stop signal at every step | shaped reward on executed or sampled candidates; changes the objective; never values *missed* stop opportunities |
| GEAR (2609.18581) | V | −1 for a premature STOP against the GT label | stop-specific term | per-state imitation-style |
| Robostral (2607.20785) | V | −max(2, d_geo) flattens the reward inside 2 m | implicitly favours stopping near goal | outcome shaping only |
| CVT-RL (2606.05263) | V-abs | counterfactual interventions on language-agent actions, doubly-robust estimator; ALFWorld, ScienceWorld | counterfactual credit | sampled continuations from a reference policy; explicitly not an exact policy-gradient effect |

Naming collision: "hindsight" is used by HER (goal relabelling), Hindsight Credit Assignment (Harutyunyan et al., NeurIPS 2019), HCAPO (2603.08754) and Φ-Nav (instruction relabelling). **Do not call the method "hindsight stop".** Use "analytic STOP branch" or "exact STOP marginalisation".

### 3.3 Honest technical caveats for claim 1 (reviewers will find these)

1. **It is variance reduction only.** The expected gradient equals GRPO/RLOO's (same objective, same optimum). Any SR gain must come from better per-sample signal on near-goal STOP decisions.
   - Appendix E: with a constant baseline, total variance fell 0.142 → 0.092; with a good state baseline the gain was negligible (0.0714 vs 0.0715).
   - Report the **variance or SNR restricted to the STOP logit at d < 3 m steps**, not total variance.
2. **Saturation.** ∇σ = σ(1 − σ)∇(logit). If σ ≈ 1e-3 near the goal, the analytic term is tiny in logit space, just as REINFORCE's is. "Does not need to sample STOP" is true, but the analytic term does not amplify a saturated logit. A logit-space or natural-gradient scaling would change the estimator and needs its own justification.
3. **Perception versus credit.** If the 141 passed-goal failures have σ_t ≈ 0 at every d < 3 m step *and* the frames are indistinguishable from non-goal frames, better credit cannot help.
4. **The "it is just stop supervision" objection.** With RLOO b, the coefficient is (Q_S − b) > 0 inside 3 m when Q_S > b, and −b outside. This looks like reward-weighted stop labels. The defence is that the movement terms carry the continuation trade-off (the log(1 − σ) part of log π), so the *sum* is the true gradient. A supervised BCE stop loss (C1) is not, and can over-stop when the policy would reach a closer point. Prove it empirically with C1 and B1.
5. **Conditions to state:** on-policy and unfiltered (no dynamic sampling); the baseline independent of the episode's own actions; rollout/trainer parity (the measured mismatch was ~0.004 nats/token); the trie-shaped σ. The batch scale s includes the episode's own R, so unbiasedness holds conditional on s.

---

## 4. Prior art for claim 2: privileged state baselines

| Work | Check | Exact claim | Relevance |
|---|---|---|---|
| Greensmith, Bartlett & Baxter, JMLR 5 (2004) | V | §5.2: a baseline on the **underlying state** of a POMDP integrates to zero, so the estimate stays unbiased; it requires state knowledge, so §5.3 moves to observation baselines. The analysis assumes reactive policies | **Primary citation** for unbiasedness of b(s_t). For a history-dependent policy the same one-line argument holds: a_t ~ π(·\|h_t) and b does not depend on a_t |
| Wu et al., action-dependent factorized baselines (ICLR 2018, 1803.07246) | V | §5: in blind peg insertion the baseline gets the hole location that the policy lacks, and learning is faster | direct empirical precedent for a privileged-information baseline |
| Weaver & Tao (UAI 2001) | V | a baseline reduces variance without bias; the optimal constant baseline is the average reward | classic |
| Baisero & Amato, AAMAS 2022 (2105.11674) | V | V(s) is generally ill-defined or a biased estimate of V(h) (Thm 4.1–4.3), even when trained on MC returns; V(h, s) is unbiased (Thm 5.1–5.2). The paper does **not** analyse a state-only *baseline* | explains why HiCo-PO uses geometry **only as a subtracted baseline with MC returns**, never as a bootstrap target or Q estimate |
| Informed Asymmetric AC (Ebi, Ernst, Böhm, Lambrechts; 2509.26000; ICML 2026 per arXiv) | V | V(h, i) with a privileged signal i gives an unbiased PG; adds tests for signal informativeness | closest theory for "a few chosen privileged signals" |
| Lambrechts, Ernst & Mahajan, ICML 2025 (2501.19116) | V | asymmetric critics remove aliasing error terms (linear FA bound) | theory for why privileged values help |
| Le Critique (Venkatraman et al., 2608.16739, Aug 2026) | V | privileged V(h, z) for LLM RL; Eq. 5: unbiased if z ⟂ the current token given h; main runs λ = 1 (A = R − V); TETHER interpolates group-relative and value baselines | **closest recent precedent**: a privileged MC baseline. TETHER is a natural extra ablation (geometry baseline blended with RLOO) |
| SWEET-RL (2503.15478), BPCO (2608.23566) | V | critics with training-time privileged information for multi-turn LLM agents | same idea, text domain |
| Pinto et al. (1710.06542); OpenAI in-hand manipulation (1808.00177); DreamWaQ (2301.10602); RAFT (2608.22976) | V | asymmetric PPO critics with simulator state | standard in robotics; these are bootstrapped critics, not MC baselines |
| GAE (Schulman et al., 1506.02438) §6.1 | V | advantages use the old V, then V is refit; "additional bias would have been introduced if we updated the value function first" | citation for the strictly out-of-sample fit |
| rllab LinearFeatureBaseline (Duan et al., 1604.06778) | V (features) | linear baseline on state and time polynomials | precedent for small hand-feature baselines. Whether rllab fits before or after predicting is UNVERIFIED |
| Lyu et al., JAIR 2023 (2408.14597) | V | state-based critics can add bias **and variance** compared with history-based ones | variance reduction is not guaranteed; measure it |
| COMA (1705.08926) | V | centralised critic with a counterfactual baseline | privileged-information baseline in MARL |
| VLD (2512.07976) | V | geometric distance-to-goal is fed to the **actor** in training and replaced by a learned distance at deployment | geodesic as privileged signal in navigation, but not as a baseline |

**In VLN specifically.** Classical VLN RL (RCM, EnvDrop; K, not re-fetched) and recent work (OctoNav A2C, Macro-PPO, SeekVLN) learn critics on policy features. Geodesic geometry appears in *rewards* (Embodied-Navigator, BudVLN, LongNav-R1), *positives* (SACA) and *oracle labels* (DAgger family), never as a baseline. That is the defensible "difference".

**Conditions to keep in the paper:**
- The features z_t must be measured **before** a_t: no d_{t+1}, no final path length.
- The fit uses previous RL steps only.
- The target is R on every visited step.
- The baseline is never used to bootstrap. Plan v1.2 already satisfies all four. RL_survey §7.4 still mentions λ = 0.95 with a history-state critic; keep that as an optional ablation only.

---

## 5. Prior art for claim 3: RL through recurrent, linear-attention or SSM memory

| Work | Check | Memory / training | Distance from claim 3 |
|---|---|---|---|
| **LANav** (Zhang et al., 2607.18794, Jul 2026) | V (HTML) | 4-layer d = 512 policy on frozen SigLIP features; compares Linear Transformer, DeltaNet, **Gated DeltaNet** and WSLA; **DAgger 150M steps → PPO 100M steps**; HM3D-OVON avg SR: GRU 26.7, Transformer 30.1, GDN 34.6, WSLA 36.4 | **closest prior**: an RL-trained GDN navigation policy. Differences: ObjectNav, not VLN; small, not a VLM; rollout length = context length (BPTT details not stated); no language instructions |
| DD-PPO (Wijmans et al., 1911.00357) | V-abs; LSTM and ICLR'20 from K | on-policy PPO with a recurrent LSTM state, 2.5B frames, PointNav | on-policy recurrent navigation RL is old |
| PoliFormer (2406.20083, CoRL'24) | V | DD-PPO with a causal transformer decoder and KV-cache; rollout length grows 32 → 128 | embodied on-policy memory RL; softmax attention; not full-episode |
| ReLIC (2410.02751), Memo (2510.19732, NeurIPS'25) | V | 64k-step in-context RL; summary-token memory trained with RL | long-memory embodied RL; not a VLM; not GDN |
| AGaLiTe (2310.15719, TMLR), S5 ICRL (2303.03982), RLBenchNet (2505.15040) | V | gated linear-attention, SSM and Mamba-2 recurrent states in online RL | small-model precedents |
| POPGym (2303.01859, ICLR'23) | V | GRU best overall; older linear transformers underperform as RL memory | caution; gating and delta rules are the counter-argument (LANav, AGaLiTe) |
| MiniMax-M1 (2506.13585) | V | RL on a lightning-attention hybrid; train/infer probability mismatch fixed with an FP32 LM head (correlation ~0.9x → 0.99x); CISPO | must-cite numerics |
| Kimi Linear (2510.26692), Ring-linear (2510.19338) | V | hybrid (gated-delta-family) linear-attention LLMs trained with RL; stability credited to train/infer operator alignment | RL of hybrid linear attention is established for text |
| HARTS (2608.28158, Aug 2026) | V (subagent) | agentic RL for hybrid-attention models over rollout trees; chunk-boundary state recovery; "gradients propagate through differentiable state handoffs" | **closest systems precedent** for BPTT through linear-attention state in multi-turn RL (text) |
| KV-streams (2609.35750), MEM1 (2506.15841), MemAgent (2507.02259, ICLR'26) | V | carried caches or rewritten text memory as recurrent state under agentic RL | text-agent analogues |
| Rollout/training mismatch, TIS (Yao et al., NeurIPS'25 / blog) | partly V | kernel differences make "on-policy" RL off-policy; truncated IS fix | GDN recurrent decode vs chunkwise training is a specific source; already measured here (~0.004 nats/token) |
| StreamVLN (2507.05240), MemoryVLA (2508.19236), RoboMamba (2406.04339), μVLA (2606.12497), AVA-VLA (2511.18960) | V | sliding-window, memory-bank, Mamba or recurrent-token VLN/VLA models | trained with IL or SFT only; no RL |

**Architecture wording.** The Qwen3.5-4B model card describes 8 × (3 Gated DeltaNet + 1 gated full-attention) layers and no sliding window. "Window8" (instruction prefix + 8 most recent observation/action groups kept in the full-attention KV) is **SimpleMemVLN's own design**. The paper must not describe it as a property of the base model.

**Finding.** Searches by both subagents (~40 queries) found **no VLM-based VLN policy trained with RL whose memory is a recurrent, linear-attention or SSM state**. The VLN-RL works re-encode the history (ActiveVLN keeps up to 200 images in context) or use windows with GRPO. BudVLN is the only streaming-VLN + RL hybrid, and its memory is softmax KV.

---

## 6. Additional recent work (2024–2026) the survey under-covers

### 6.1 GRPO variants and critiques (relevance: baselines and normalisation)

| Work | Check | Point for this project |
|---|---|---|
| GRPO (2402.03300), DAPO (2503.14476) | V | DAPO's clip-higher and dynamic sampling. Dynamic sampling **biases** the estimator, so keep it out of the unbiased ladder (as planned; v0-DS is a control) |
| Dr.GRPO (2503.20783) | V | std and length normalisation are biased. Failing navigation episodes are longer, so per-episode 1/T normalisation under-penalises them; the plan's per-action global normalisation avoids this |
| RLOO (2402.14740) | V | unbiased leave-one-out baseline. **v2a must use it** (planned) |
| REINFORCE++ (2501.03262; v9 title "...Global Advantage Normalization") | V | global-batch normalisation; argues per-prompt std is biased |
| Why GRPO Needs Normalization (2601.23135) | V | std normalisation acts as an adaptive step size; gains at 1.5B, statistically indistinguishable at 7B. **Supports keeping a single batch scale s** |
| Rethinking RL deep-research agents (2510.15862) | V | on-policy RLOO > GRPO for a multi-turn agent |
| GSPO (2507.18071), CISPO (in 2506.13585) | V | sequence-level ratio and clipped IS weights; with μ = 1 (no ratio) mostly irrelevant |
| VinePPO (2410.01679, ICML'25) | V | MC values from intermediate-state resets; PPO critics barely beat random. Precedent for a simulator-fork extension (+F) |
| PRIME (2502.01456) | V | implicit PRM from outcome labels |
| Yue et al., pass@k (2504.13837, NeurIPS'25); ProRL (2505.24864) | V | RLVR sharpens rather than expands; ProRL counters with reference resets. Report success@k of the base as a headroom diagnostic |

### 6.2 Credit assignment for multi-turn agents

| Work | Check | Mechanism | Fit for 60–100-step continuous navigation |
|---|---|---|---|
| GiGPO (2505.10978, NeurIPS'25) | V | step advantage over repeated "anchor states" | needs identical states, rare in continuous poses; quantised-pose variant possible; ECPO (2606.05885) shows small-count anchors are noisy |
| MT-GRPO / turn-level credit (2505.11821; first author and title changed across versions) | V | per-turn group-normalised advantages | rollouts grow with turns; infeasible here |
| SPA-RL (2505.20732), HCAPO (2603.08754), GACA (2609.12424), TRACE (2607.13988), EFCA (2608.08255), SLEA-RL (2603.18079) | V/V-abs | learned or LLM-estimated step values, criticality weighting, TD of the gold-answer log-prob | estimated credit; geometry gives exact Q_S for free here |
| ArCHer (2402.19446), Turn-PPO (2512.17008), STEP (2511.13091) | V | turn-level critics; Turn-PPO finds PPO more robust than GRPO for multi-turn | a learned-critic comparator |
| RAGEN/StarPO (2504.20073) | V | "Echo Trap" collapse; keep high-variance prompts | entropy and variance monitoring |
| Tree-GRPO (2509.21240, ICLR'26) | V | shared-prefix trees; within-tree ≈ step-level DPO | fork extension (+F) |
| EMPG (2509.09265) | V | scales step advantages by step entropy | cheap add-on; ALFWorld GRPO 74.8 → 78.5 |
| SeeNav-Agent / SRGPO (2512.02631) | V | step-level grouping with verifiable process rewards; EmbodiedBench-Nav, not R2R | closest embodied step-credit analogue |
| PAV (2410.08146) | V | process reward should be an advantage (progress) under a prover policy | theory behind Q_S − b as a progress-like signal |

### 6.3 On-policy distillation (OPD) and DPO for agents

- **OPD** (Thinking Machines blog, Oct 2025; GKD 2306.13649; MiniLLM 2306.08543) gives a dense per-token reverse-KL signal on student states. Embodied variants:
  - VLA-OPD (2603.26666): about 3× fewer steps than GRPO on LIBERO-Long.
  - FA-OPD (2605.27095).
  - Guided-OPD (2606.15912): teacher interventions annealed to 0.
- **OPSD** (2601.18734, "Self-Distilled Reasoner"): its best divergence is *forward* KL. Do not merge it with reverse-KL OPD.
- **Rethinking OPD** (2604.13016): OPD needs a compatible and genuinely stronger teacher.
- **No OPD paper on VLN or R2R was found.** A geometric-teacher OPD (survey "+G") would be new but suffers the imitation gap (ADVISOR 2007.12173).
- **DPO for agents:** ETO (2403.02502), IPR (2406.11176; step-level MC rewards → DPO), Agent Q (2408.07199), Step-DPO (2406.18629). For VLN, only PEAP-LLM (2505.07500, REVERIE) and aerial-VLN DPO works (2608.07557, 2610.10635; abstracts only) were found. No DPO on R2R-CE.

### 6.4 Entropy and exploration (LLM and VLA)

- Entropy mechanism (2505.22617): ΔH ∝ −Cov(log π, A); Clip-Cov and KL-Cov act on about 0.02–0.2% of tokens.
- 80/20 forking tokens (2506.01939): with 4 actions, every action token is a fork; the analogue is decision steps.
- SimpleVLA-RL (2509.09674): temperature 1.6, clip-higher 0.28, no KL.
- RL4VLA (2505.19789): PPO > GRPO > DPO for VLA.
- πRL (2510.25889), RLinf-VLA (2510.06710), RIPT-VLA (2505.17016; RLOO for VLA).
- **Baseline context:** entropy fell 0.147 → ~0.08 nats/token (−45%) in 125 GRPO steps. Monitor it in every HiCo-PO run, and treat a T = 1.4 sensitivity run as a diagnostic, not a new rung.

---

## 7. Novelty assessment

### 7.1 Claim 1: analytic STOP branch

| Closest prior | Same | Different |
|---|---|---|
| Sutton et al. 2000 (all-actions form); MAC (1709.00503); EPG (1706.05374); Nauman & Cygan (2210.13011) | integrate the gradient over actions instead of sampling | they need a learned Q for all actions; HiCo-PO integrates only STOP, with **exact** Q_S, and samples the rest |
| Option-critic termination gradient (1609.05140) | gradient on the termination probability weighted by (terminate − continue) value | learned Q_Ω/V_Ω; HiCo-PO's terminate value is exact and continuation is handled by sampled REINFORCE terms |
| Deep Optimal Stopping (JMLR 2019; 1804.05394) | exact stop payoff × soft stop probability + sampled continuation payoff | exogenous state process, backward induction, no control actions; HiCo-PO's continuation is generated by the same policy's moves |
| Q-Prop (1611.02247), Stein CVs, action-dependent baselines (1803.07246); critique by Tucker et al. (1802.10031) | control variate with an analytic expectation | their CV is a learned Q (the source of the "mirage"); here it is exact for STOP |
| S-GRPO (2505.07686), AnytimeReasoner (2505.13438), MRT (2503.07572) | value of exiting at many prefixes of one trajectory | require extra answer generation and a verifier; shaped or exogenous-budget objectives; not navigation |
| Embodied-Navigator r_stop (2608.17512), GEAR (2609.18581), L2Stop (2009.13112) | STOP-specific signal from geometry or labels | hand-set shaping or supervised labels (biased w.r.t. the task return); no use of the value of *unexercised* stop opportunities |

**Novelty:** moderate, as an instantiation. To our search, it is the first VLN/VLM-navigation RL estimator that marginalises STOP exactly at every visited decision using simulator geometry, at zero extra rollout or decoding cost.

**Risk: medium.** A theory reviewer can say "this is the all-actions PG / optimal stopping applied to one action". The paper survives if it (a) cites those works up front, (b) shows the structural property (exact Q_S, no extra generation, unlike S-GRPO/AnytimeReasoner), and (c) shows an empirical effect where outcome GRPO fails: OS−SR and the passed-goal failures.

**Defensible phrasing:**
> "In embodied navigation, the return of STOP at any visited pose is a known function of the simulator state. We therefore marginalise the STOP branch analytically at every visited decision (the all-actions policy-gradient form restricted to the terminal action, cf. Sutton et al. 2000; Bacon et al. 2017; Becker et al. 2019) and sample only movement actions. The estimator is unbiased for any action-independent baseline, needs no critic for STOP, and costs no extra rollouts."

**Avoid:** "new policy-gradient theorem", "hindsight", "first to give stop credit" (Embodied-Navigator and GEAR give STOP-specific signals), and "solves overshooting" before the C1/B1 comparisons.

**Required evidence:** v2 vs v2a (RLOO); v2 vs B1 (STOP-bias) by ≥ 1 SR; v2 vs C1 (supervised stop BCE on own rollouts); passed-goal failure count and OS−SR; stop-logit SNR at d < 3 m; two seeds.

### 7.2 Claim 2: privileged geometric state baseline

| Closest prior | Same | Different |
|---|---|---|
| Greensmith et al. 2004 §5.2; Wu et al. 2018 §5 | latent or privileged-state baseline, unbiased in POMDPs | generic; not VLN; not geometry |
| Le Critique (2608.16739); Informed AAC (2509.26000) | privileged value used as a baseline with MC returns / informed signals | LLM or generic RL; condition on h as well. HiCo-PO uses geometry only, which is cheaper and possibly higher-variance (Lyu 2023) |
| Pinto 2017; OpenAI 2018; legged robotics; RAFT | asymmetric critics with simulator state | bootstrapped critics; small non-VLM policies |
| OctoNav A2C; Macro-PPO; SeekVLN | learned value in VLN RL | policy features or a VLM critic, not privileged geometry |
| GAE §6.1; rllab | fit the baseline on previous data / hand features | generic practice |

**Novelty:** low to moderate. The theory and practice are standard; only the VLN instantiation (a geometry-only MLP baseline, no image critic, enabling G = 2) is new to our search.

**Risk: high as a headline.** It will read as an implementation detail unless (i) v4 (G = 2) holds SR within 0.5 of v3 at half the rollouts, and (ii) the fixed-checkpoint variance probe shows lower raw variance than RLOO.

**Defensible phrasing:**
> "Because the simulator exposes geodesic geometry at training time, we use a small out-of-sample regression on geometric features as a state baseline (unbiased under Monte-Carlo returns, Greensmith et al. 2004; it is never bootstrapped, avoiding the bias identified by Baisero & Amato 2022). This removes the need for a group of rollouts to form a baseline and halves rollout cost."

Add a geometry+RLOO blend (TETHER-style, Le Critique) and a geometry+policy-scalars baseline as ablations.

### 7.3 Claim 3: on-policy full-episode RL through hybrid GDN memory

| Closest prior | Same | Different |
|---|---|---|
| LANav (2607.18794) | Gated DeltaNet / linear-attention navigation policy fine-tuned with PPO after DAgger; GDN > Transformer > GRU | ObjectNav, small non-VLM policy, no language instruction, BPTT unspecified |
| DD-PPO; PoliFormer; ReLIC; Memo | on-policy RL with recurrent or transformer memory in embodied navigation | LSTM or softmax KV; not hybrid GDN; not a VLM |
| MiniMax-M1; Kimi Linear; Ring-linear; HARTS | RL on hybrid linear-attention LMs; HARTS backprops through state handoffs | text agents; no visual streaming; no embodiment |
| ActiveVLN, SACA, SeekVLN, LightNav-0, BudVLN | closed-loop RL of mono-RGB VLM VLN policies | full-context or re-encoded history (softmax attention); no recurrent state |
| AGaLiTe; S5; RLBenchNet | gated linear attention / SSM states trained by RL | toy and benchmark POMDPs |

**Novelty:** moderate as an empirical first, by our search: a VLM VLN policy whose cross-episode memory is a persistent GDN state, trained by closed-loop RL with full-episode BPTT. The general idea (RL through recurrent memory in navigation) is old, and LANav makes "RL with GDN for navigation" not new.

**Risk: medium-high.** Reviewers will see the ActiveVLN recipe ported to a different backbone.

**Defensible phrasing:** use "we show that" rather than "first". For example:
> "Full-episode BPTT through the persistent GDN state is stable under on-policy RL at 4B scale (rollout/trainer mismatch ≈ 0.004 nats/token), and updating the memory pathway matters: BPTT > detached-state > readout-only."

This only holds with the C2 (readout-only) and (B) detached-state ablations, plus a memory-dependence probe, such as the Video-R1-style comparison of rollouts with intact vs cleared memory. Without them, keep claim 3 as setup, not contribution.

### 7.4 Overall positioning

- The strongest honest framing is **one claim with two supporting components**: "simulator geometry gives exact or cheap counterfactual information at training time; we use it to marginalise STOP exactly and to baseline movement credit, on a streaming recurrent-memory VLM policy."
- Claim 3 is the setting. It is not a contribution unless the BPTT ablation shows an effect.
- Do not claim SOTA. With mono-RGB, R2R+RxR-only data, LightNav-0 (68.5), Qwen-RobotNav-4B (66.9) and Robostral (77.4) are far above. Frame the results as matched-budget gains over GRPO on the same base.

---

## 8. Corrections and updates to RL_survey.md and RL_technical.md

1. **2506.09839** is titled "OctoNav: Towards Generalist Embodied Navigation"; OctoNav-R1 is the model. It reports no R2R-CE numbers.
2. **Mean Actor-Critic** (1709.00503): the arXiv listing gives Allen, Asadi, Roderick, Mohamed, Konidaris, Littman, with no venue. Cite as "Allen, Asadi et al., arXiv 2017", not "Asadi et al.".
3. **Expected Policy Gradients** (1706.05374) is AAAI 2018 (Ciosek & Whiteson). A longer JMLR version may exist (UNVERIFIED here).
4. **Baisero & Amato 2022** (RL_survey §7.4): the bias result concerns V(s) used as an estimate of V(h) or Q(h,a) in the gradient, and holds even with MC-trained critics. It is not specific to λ < 1 bootstrapping. A pure state *baseline* with MC returns is unbiased (Greensmith 2004). Plan v1.2 §8 is consistent with this; the survey text should be updated.
5. **"PPO with a critic beats GRPO"** (survey lesson 6): the evidence is mixed on navigation. Macro-PPO: PPO 68.1 > GRPO 66.8. LongNav-R1: PPO value head unstable. NavGRPO reportedly favours GRPO (search-level only, UNVERIFIED).
6. **Claim-3 novelty** (survey §7.9): add **LANav (2607.18794)** as the closest prior (GDN navigation policy + PPO) and **HARTS (2608.28158)** for BPTT through linear-attention state in agentic RL.
7. **Window8 wording:** Qwen3.5-4B has no sliding window (gated full attention); Window8 is the project's eviction policy.
8. **MT-GRPO (2505.11821):** the first author changed (Zeng → Wei) and so did the title across versions. Cite the version used.
9. **"Hindsight stop"** naming (survey §0, §7.5) collides with HER, HCA and HCAPO. Rename it.
10. **OPSD** (2601.18734) is titled "Self-Distilled Reasoner…", and its best divergence is forward KL.
11. **ActiveVLN:** the paper's ablation without dynamic sampling gives 48.1/40.6. The headline 50.1/43.7 uses dynamic sampling.
12. **Le Critique** and **Informed AAC** are confirmed (2608.16739; 2509.26000, "Accepted at ICML 2026"). Le Critique's TETHER blend is a cheap extra ablation for v3.

---

## 9. Must-cite and must-compare (CVPR/ICML submission)

### 9.1 Must-compare (same base, same 1K order, same evaluator; budget-matched)

1. **SFT base** and **continued SFT / online DAgger at matched GPU-hours** (C3), to answer "is RL better than more supervision?".
2. **ActiveVLN-style GRPO** (done: 56.88 SR / 50.45 SPL). Also report v0-DS vs no-DS.
3. **RLOO** (v2a; Ahmadian et al. 2024), to isolate the estimator family from the STOP term.
4. **STOP-bias calibration** (B1), the free must-beat control for the STOP component.
5. **On-policy supervised STOP loss** (C1: BCE toward 1[d < 3] on own rollouts, L2Stop-like), to answer "is it just stop supervision?".
6. **Embodied-Navigator-style stop/approach reward shaping**, reimplemented on mono RGB: the closest published STOP-specific RL signal.
7. **A learned-critic step-credit baseline:** PPO with a value head on policy hidden states (OctoNav / Macro-PPO / RL4VLA style) or GiGPO with quantised-pose anchors. Without one, claim 2 lacks a comparator.
8. **Ablations for claim 3:** full BPTT vs detached state vs readout-only (C2).

External table only, with protocol caveats: ActiveVLN, SACA, BudVLN, SeekVLN, LightNav-0, plus the SFT references StreamVLN and MemVLN-4B.

### 9.2 Must-cite

- **VLN-RL:** ActiveVLN, SACA, SeekVLN/C2PO, BudVLN, Embodied-Navigator, LightNav-0, Macro-action PPO, GroundingVLN/GEAR, VLN-R1, LongNav-R1.
- **Estimators and stopping:** Sutton et al. 2000; MAC; EPG; Bacon et al. 2017; Becker et al. 2019; Q-Prop; Tucker et al. 2018; S-GRPO; AnytimeReasoner; L2Stop.
- **Baselines and asymmetry:** Greensmith et al. 2004; Wu et al. 2018; Baisero & Amato 2022; Pinto et al. 2017; Informed AAC; Le Critique; GAE.
- **GRPO family:** GRPO, DAPO, Dr.GRPO, RLOO, GiGPO.
- **Memory:** Gated DeltaNet, LANav, DD-PPO, PoliFormer, MiniMax-M1, HARTS, StreamVLN.

---

## 10. Suggested pre-v2 diagnostics (cheap; no new training)

1. On the 141 passed-goal-then-failed val episodes of the GRPO export, log σ_t at every step with d_t < 3 m. If max σ is above ~0.1 in most of them, the issue is calibration or credit, which claim 1 targets. If σ ≈ 0 throughout, it is recognition, and claim 1 is unlikely to help much.
2. Fixed-checkpoint estimator probe (RL_technical §8.4): per-episode gradient w.r.t. the restricted root logits for GRPO, RLOO, v2 and v3. Report SNR **on the STOP logit at d < 3 m steps**, separately from total variance.
3. Held-out R² of the geometric baseline on the existing 4,868 GRPO rollouts. If R² is below ~0.2, expect little variance reduction from v3 and G = 2 is at risk.

---

## References

**VLN / navigation RL**
- ActiveVLN 2509.12618: https://arxiv.org/abs/2509.12618
- VLN-R1 2506.17221: https://arxiv.org/abs/2506.17221
- SACA 2603.09740: https://arxiv.org/abs/2603.09740
- BudVLN 2602.06356: https://arxiv.org/abs/2602.06356
- SeekVLN / C2PO 2609.37353: https://arxiv.org/abs/2609.37353
- Embodied-Navigator / TAMP-Nav 2608.17512: https://arxiv.org/abs/2608.17512
- LightNav-0 2608.30935: https://arxiv.org/abs/2608.30935
- GN0 2606.03682: https://arxiv.org/abs/2606.03682
- MapDream 2602.00222: https://arxiv.org/abs/2602.00222
- Nav-R1 2509.10884: https://arxiv.org/abs/2509.10884
- ETP-R1 2512.20940: https://arxiv.org/abs/2512.20940
- Macro-action PPO 2609.03906: https://arxiv.org/abs/2609.03906
- Robostral Navigate 2607.20785: https://arxiv.org/abs/2607.20785
- JOP-VLN 2607.13461: https://arxiv.org/abs/2607.13461
- GroundingVLN / GEAR 2609.18581: https://arxiv.org/abs/2609.18581
- Progress-Think 2511.17097: https://arxiv.org/abs/2511.17097
- OctoNav 2506.09839: https://arxiv.org/abs/2506.09839
- LongNav-R1 2602.12351: https://arxiv.org/abs/2602.12351
- NavGRPO 2603.15370: https://arxiv.org/abs/2603.15370
- VerNav 2609.00920: https://arxiv.org/abs/2609.00920
- SeeNav-Agent 2512.02631: https://arxiv.org/abs/2512.02631
- LANav 2607.18794: https://arxiv.org/abs/2607.18794
- StreamVLN 2507.05240: https://arxiv.org/abs/2507.05240
- L2Stop 2009.13112: https://arxiv.org/abs/2009.13112
- PEAP-LLM 2505.07500: https://arxiv.org/abs/2505.07500
- DD-PPO 1911.00357: https://arxiv.org/abs/1911.00357
- PoliFormer 2406.20083: https://arxiv.org/abs/2406.20083
- ReLIC 2410.02751: https://arxiv.org/abs/2410.02751
- Memo 2510.19732: https://arxiv.org/abs/2510.19732
- VLD 2512.07976: https://arxiv.org/abs/2512.07976
- ADVISOR 2007.12173: https://arxiv.org/abs/2007.12173

**Estimators, stopping, control variates**
- Sutton et al. 2000 (K): https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation
- Mean Actor-Critic 1709.00503: https://arxiv.org/abs/1709.00503
- Expected Policy Gradients 1706.05374: https://arxiv.org/abs/1706.05374
- On All-Action / Many-Actions PG 2210.13011: https://arxiv.org/abs/2210.13011
- Option-Critic 1609.05140: https://arxiv.org/abs/1609.05140
- Deep Optimal Stopping (JMLR 20(74), 2019): https://jmlr.org/papers/v20/18-232.html · https://ar5iv.labs.arxiv.org/html/1804.05394
- Q-Prop 1611.02247: https://arxiv.org/abs/1611.02247
- Mirage of action-dependent baselines 1802.10031: https://arxiv.org/abs/1802.10031
- Action-dependent factorized baselines 1803.07246: https://arxiv.org/abs/1803.07246
- S-GRPO 2505.07686: https://arxiv.org/abs/2505.07686
- AnytimeReasoner 2505.13438: https://arxiv.org/abs/2505.13438
- MRT 2503.07572: https://arxiv.org/abs/2503.07572
- ESTAR 2602.10004: https://arxiv.org/abs/2602.10004
- Learning When to Stop (latent) 2511.21581: https://arxiv.org/abs/2511.21581
- PUMA 2605.17672: https://arxiv.org/abs/2605.17672
- DASH 2607.00482: https://arxiv.org/abs/2607.00482
- CVT-RL 2606.05263: https://arxiv.org/abs/2606.05263
- HCAPO 2603.08754: https://arxiv.org/abs/2603.08754

**Baselines, asymmetric and privileged critics**
- Greensmith et al. 2004: https://www.jmlr.org/papers/v5/greensmith04a.html
- Weaver & Tao 2001: https://arxiv.org/abs/1301.2315
- Baisero & Amato 2022: https://arxiv.org/abs/2105.11674
- Asymmetric DQN (UAI 2022): https://proceedings.mlr.press/v180/baisero22a.html
- Lambrechts et al. 2025: https://arxiv.org/abs/2501.19116
- Informed Asymmetric AC 2509.26000: https://arxiv.org/abs/2509.26000
- Le Critique 2608.16739: https://arxiv.org/abs/2608.16739
- Pinto et al. 2017: https://arxiv.org/abs/1710.06542
- OpenAI in-hand manipulation: https://arxiv.org/abs/1808.00177
- DreamWaQ: https://arxiv.org/abs/2301.10602
- RAFT: https://arxiv.org/abs/2608.22976
- COMA: https://arxiv.org/abs/1705.08926
- Lyu et al. 2021: https://arxiv.org/abs/2102.04402
- Lyu et al. JAIR 2023: https://arxiv.org/abs/2408.14597
- GAE: https://arxiv.org/abs/1506.02438
- rllab benchmark: https://arxiv.org/abs/1604.06778
- SWEET-RL: https://arxiv.org/abs/2503.15478
- BPCO: https://arxiv.org/abs/2608.23566

**GRPO family, credit assignment, OPD/DPO, entropy**
- GRPO 2402.03300: https://arxiv.org/abs/2402.03300
- DAPO: https://arxiv.org/abs/2503.14476
- Dr.GRPO: https://arxiv.org/abs/2503.20783
- RLOO: https://arxiv.org/abs/2402.14740
- REINFORCE++: https://arxiv.org/abs/2501.03262
- GSPO: https://arxiv.org/abs/2507.18071
- VAPO: https://arxiv.org/abs/2504.05118
- VinePPO: https://arxiv.org/abs/2410.01679
- PRIME: https://arxiv.org/abs/2502.01456
- Why GRPO Needs Normalization: https://arxiv.org/abs/2601.23135
- RLOO vs GRPO for agents: https://arxiv.org/abs/2510.15862
- pass@k: https://arxiv.org/abs/2504.13837
- ProRL: https://arxiv.org/abs/2505.24864
- GiGPO: https://arxiv.org/abs/2505.10978
- ECPO: https://arxiv.org/abs/2606.05885
- MT-GRPO: https://arxiv.org/abs/2505.11821
- SPA-RL: https://arxiv.org/abs/2505.20732
- ArCHer: https://arxiv.org/abs/2402.19446
- RAGEN: https://arxiv.org/abs/2504.20073
- Tree-GRPO: https://arxiv.org/abs/2509.21240
- PAV: https://arxiv.org/abs/2410.08146
- EMPG: https://arxiv.org/abs/2509.09265
- AgentGym-RL: https://arxiv.org/abs/2509.08755
- Turn-PPO: https://arxiv.org/abs/2512.17008
- STEP: https://arxiv.org/abs/2511.13091
- GACA: https://arxiv.org/abs/2609.12424
- TRACE: https://arxiv.org/abs/2607.13988
- EFCA: https://arxiv.org/abs/2608.08255
- SLEA-RL: https://arxiv.org/abs/2603.18079
- OPD blog: https://thinkingmachines.ai/blog/on-policy-distillation/
- GKD: https://arxiv.org/abs/2306.13649
- MiniLLM: https://arxiv.org/abs/2306.08543
- OPSD: https://arxiv.org/abs/2601.18734
- Rethinking OPD: https://arxiv.org/abs/2604.13016
- VLA-OPD: https://arxiv.org/abs/2603.26666
- Guided-OPD: https://arxiv.org/abs/2606.15912
- FA-OPD: https://arxiv.org/abs/2605.27095
- ETO: https://arxiv.org/abs/2403.02502
- IPR: https://arxiv.org/abs/2406.11176
- Agent Q: https://arxiv.org/abs/2408.07199
- Step-DPO: https://arxiv.org/abs/2406.18629
- Entropy mechanism: https://arxiv.org/abs/2505.22617
- 80/20 forking tokens: https://arxiv.org/abs/2506.01939
- SimpleVLA-RL: https://arxiv.org/abs/2509.09674
- RL4VLA: https://arxiv.org/abs/2505.19789
- πRL: https://arxiv.org/abs/2510.25889
- RLinf-VLA: https://arxiv.org/abs/2510.06710
- RIPT-VLA: https://arxiv.org/abs/2505.17016
- VLA-RL: https://arxiv.org/abs/2505.18719

**Recurrent and linear-attention memory, systems**
- Gated DeltaNet: https://arxiv.org/abs/2412.06464
- Qwen3.5-4B card: https://huggingface.co/Qwen/Qwen3.5-4B
- MiniMax-M1: https://arxiv.org/abs/2506.13585
- Kimi Linear: https://arxiv.org/abs/2510.26692
- Ring-linear: https://arxiv.org/abs/2510.19338
- HARTS: https://arxiv.org/abs/2608.28158
- KV-streams: https://arxiv.org/abs/2609.35750
- MEM1: https://arxiv.org/abs/2506.15841
- MemAgent: https://arxiv.org/abs/2507.02259
- AGaLiTe: https://arxiv.org/abs/2310.15719
- S5 in-context RL: https://arxiv.org/abs/2303.03982
- POPGym: https://arxiv.org/abs/2303.01859
- RLBenchNet: https://arxiv.org/abs/2505.15040
- GTrXL: https://arxiv.org/abs/1910.06764
- Rollout/training mismatch (TIS): https://fengyao.notion.site/off-policy-rl · https://neurips.cc/virtual/2025/124259
- MemoryVLA: https://arxiv.org/abs/2508.19236
- RoboMamba: https://arxiv.org/abs/2406.04339
- μVLA: https://arxiv.org/abs/2606.12497
- AVA-VLA: https://arxiv.org/abs/2511.18960
