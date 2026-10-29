# S8N + RL: event-boundary policy optimisation — design and pre-registered plan (2026-10-27)

Branch `RL_s8n` (from `RL_memfork` c63c319). This file is the dated design log (pre-registration, amendments,
decisions as they were taken). **Results and the current status are in `s8n_rl_report.md`**, whose tables are
regenerated from the journals by `scripts/rl/s8n_report.py` (`results/s8n/`); where a number here differs from
the report in the last digit (bootstrap CIs here came from a 5,000-draw development script, the report uses
`compare_evals.py` with 10,000 draws), the report is authoritative.

Status (2026-10-29): GRPO, GRPO+F-B and the retention probes are finished and evaluated; event-boundary
forking failed its pre-registered test (H1); F-B ties GRPO; memory twins has a 10-step pilot only, its runs
were cancelled with all other jobs by a cluster administrator (§8, Interruption).

## 1. Thesis (one idea, not two)

S8N's memory rule and the RL algorithm should be the *same* mechanism seen from two sides.

S8N writes an observation into long-term attention memory when it moves the agent's own recurrent
state: per scoring GDN layer l, `p_t^l = ||S_after - S_before||_F / max(||S_before||_F, eps_l)` (FP32
side pass of the delta rule, birth-fixed), and each attention layer keeps the top-4 frames by `p^l`.
High `p_t` is an **event boundary** of the agent's own experience (a new room, a junction, a landmark):
the moment the compressed past stops predicting the present.

Claim: *events are the right unit for both memory and learning in a streaming agent.*

1. **Memory** (S8N, done): keep the frames at event boundaries.
2. **Exploration / credit** (new): branch RL rollouts at event boundaries and credit each event segment
   by its sibling contrast. The sampled decision right after the world changes is the one whose
   alternatives matter, and the trajectory between two boundaries is a natural temporal unit for credit.
3. **Why it is one system**: the fork trigger *is* the memory write signal; the fork snapshot *is* the
   bounded event memory (O(1) in t, ~230 MiB for Qwen3.5-4B: 8 attention layers x (prefix + 4 frames +
   4 recent groups) KV + 24 GDN/conv states + 8 scorer states), whereas a full-history agent's branch
   state grows with t (1.0 / 1.9 / 3.9 GiB of KV at t = 50 / 100 / 200). And the policy is trained on
   exactly the event memory it acted with (realised-memory replay, §3.1), which SFT on expert streams
   never exposes it to.

Framing: event segmentation theory (Zacks et al.: prediction-error boundaries segment experience and
gate episodic encoding); tree-structured RL for LLMs branches at *policy* uncertainty (entropy:
TreeRL, ARPO); an embodied agent should branch where the *world* surprises its memory.

## 2. Starting point (verified reports)

| Policy | SFT val_unseen SR | + GRPO | + GRPO+F-B | other |
|---|---:|---:|---:|---|
| Window8 | 51.55 | 56.88 / 57.37 | 58.40 / 58.08 | GRPO G=8 59.82 (matched interaction) |
| M5_E1 | 51.22 | 55.68 | 60.03 | MemFork 60.74 (n.s. vs F-B; retention collapsed to the M5 prior) |
| **S8N** | **52.42** (SPL 48.68) | — | — | this branch |

F-B = one fork per main trajectory at the first decision in [target, target+8] (target ~ U[1, N*))
with first-token entropy >= the median of the previous step, one branch, sibling contrasts divided by
one std of all main rewards of the step (`rl/hico.fork_coefficients`, `fork_scale: batch`).

## 3. Implementation on this branch

### 3.1 S8N rollouts (`rl/sweep_memory.py`, `rl/sampling.py`, `rl/rollout.py`, `rl/episode.py`)

- `SamplingSession` serves `episodic_window_gdn_sweep_v1` with `SweepRolloutState`: the unchanged
  `GdnSweepSessionState` plus a record of the frame sources every attention layer retains after each
  promotion (`layer_banks`: step -> {layer: sources}, stored only when it changes).
- The trainer's full-episode forward attends to exactly that memory (`injected_layer_plans` -> per-layer
  `LayerSelectionPlan`, the S8N scorer side pass is skipped). Offline re-scoring from BF16 training
  activations could flip near-tied frames and train on a memory the policy never saw; on the FP32 test
  model the injected plans equal the offline S8N plans bit for bit (`tests/rl/test_s8n_rollout.py`).
- Fork snapshots clone the full S8N state (`clone_sweep_state`): running top-F frames per layer,
  resident row IDs, recent groups with birth priorities, scorers' FP32 recurrent state and conv tails,
  the realised record. Restored branches continue bitwise as the original session would.
- Every response records the per-layer drift of its observation (`drift`); rollouts log `step_drift`.

### 3.2 Event-boundary forking (`fork.trigger`)

`event_score(o_t) = mean_l log p_t^l` (known after o_t is encoded, before a_t is drawn, so placement
stays causal and `fork_coefficients` stays unbiased exactly as for entropy forks).

| trigger | fork at the first decision in [target, target+window] with |
|---|---|
| `entropy` (F-B) | first-token entropy >= q_0.5 of the previous step's entropies |
| `drift` (**EBF**) | event score >= `drift_quantile` of the previous step's event scores |
| `drift_entropy` | both |

Estimator, scale, branches and budgets are those of F-B, so EBF differs from F-B *only* in where it forks.

**EBF-pos** (`ebf_pos_s8n_r2r1k.yaml`, `fork.drift_by_position: true`): the threshold is the previous step's
median *within the decision's position bucket* (`rl/rollout.position_bucket`, t = 0 | 1-2 | 3-6 | 7-14 | ...;
a bucket with < 16 decisions inherits the previous bucket's threshold). It forks where memory is more
surprised than usual at that point of an episode. Measured reason (base S8N, F-B config, step 0, 32 mains,
2,198 decisions, `svln_runs/dev/s8n_v1/analysis/drift_by_t_fb_step0.txt`): the raw event score falls with t
because ||S_before|| grows (mean -0.19 at t = 0, -0.71 at t = 3-5, -0.95 at t >= 50; Spearman with t/N = -0.36),
and the share of decisions above the pooled median is 1.00 / 0.93 / 0.75 / 0.57 / 0.49 / 0.29 for
t in [1,3) / [3,6) / [6,12) / [12,25) / [25,50) / >= 50. With a window of 9 decisions a pooled-median EBF
therefore fires almost always near the window start, i.e. it would mostly re-time F-B rather than find
events. The trailing-window local surprise of §5 is biased the other way (0.00 / 0.04 / 0.25 / 0.50 / 0.57 /
0.60 above its median), so it is not used. Step-0 thresholds of both variants come from
`scripts/rl/calibrate_drift.py` on the F-B run (the thresholds `fork_threshold` computes at its step 1).
Step 0 uses `initial_min_drift` = the median event score of the F-B run's step-0 main decisions (same
base policy, same 8 prompts), so EBF's first threshold is computed exactly as its later ones.

## 4. Hypotheses (pre-registered)

- **H1 (mechanism)**: in S8N GRPO+F-B rollouts (entropy-placed forks, drift logged), forks at higher
  event score have larger sibling outcome divergence: P(R_branch != R_main) and E|R_branch - R_main|
  increase with the drift quantile at the fork. Test: logistic / rank correlation over all forks of the
  run, bootstrap over episodes. This is measured before EBF is run and decides the trigger variant.
  Because the raw score is confounded with t (§3.2), the deciding statistic is the position-normalised
  score `event_pos` (score minus the same step's median in the same position bucket): its quintile table
  and its coefficient in the logistic model on (event_pos, entropy, log(1 + t))
  (`scripts/rl/analyze_event_forks.py`: `by_event_pos_quintile`, `logistic_position`). EBF-pos is run if
  event_pos predicts informativeness (z > 2, monotone quintiles); raw EBF only if the raw score does so
  beyond t; otherwise H1 fails (see below).
- **H2 (geometry)**: event-score peaks are enriched at reference-path turning points and where
  geodesic progress changes slope (from `step_d`, `step_bearing`, positions in the rollout logs).
- **H3 (main)**: S8N + EBF > S8N + GRPO (ActiveVLN) and >= S8N + F-B and S8N + GRPO G=8 on val_unseen
  SR (paired McNemar over 1,839 episodes; checkpoint chosen on val_seen-300 only).
- **H4 (efficiency)**: EBF reaches F-B's final val_seen-300 SR with fewer environment steps.

If H1 fails (drift does not predict fork informativeness), report it and do not claim the coupling;
fall back to `drift_entropy` or local-surprise (event score minus its recent median) only as a stated
exploratory variant.

## 5. Risks

- Early-episode bias: `||S_before||` is small after the prefix, so p_t is large for the first frames.
  Targets start at t >= 1 and the threshold is a quantile over all decisions of the previous step; if
  forks pile up at t <= 3, use the local-surprise variant (decided from H1/H2 data, not from val results).
- Interaction cost is identical to F-B by construction; G=8 is the matched-interaction control.
- MemFork on S8N is not a drop-in port: MemFork's q_phi uses M5 token committed-write features and one
  bank shared by all layers, while S8N keeps a per-layer bank. A faithful variant needs per-layer drop
  decisions over (4 + 1) frames per layer with drift features; deferred until §6 runs 1–3 exist.

## 6. Runs (1k R2R prompts, 125 steps, seed 20261010, 2 x RTX PRO 6000, sequential)

| # | config | purpose |
|---|---|---|
| 0 | S8N base, val_unseen + val_seen-300 | reproduce 964/1,839 on HiPerGator before any RL (dev allocation gate) |
| 1 | `hico_grpo_fork_bs_s8n_r2r1k.yaml` | GRPO+F-B (entropy forks); H1 data and EBF calibration, so it runs first |
| 2 | `activevln_grpo_s8n_r2r1k.yaml` | ActiveVLN GRPO baseline |
| 3 | `ebf_pos_s8n_r2r1k.yaml` or `ebf_s8n_r2r1k.yaml` | event-boundary forks; variant chosen by H1 on run 1 |
| 4 | `activevln_grpo_g8_s8n_r2r1k.yaml` | matched-interaction control |

Measured cost on 2 x RTX PRO 6000 (one full-size step each, base S8N): F-B 370 s/step (32 mains + 30
branches, peak 53 GB/GPU), G=8 747 s/step (64 trajectories, peak 57 GB/GPU); trainer-vs-rollout
log-prob mismatch 0.0042-0.0043 mean with realised-memory replay (re-scored plans disagree with the
realised memory in 5.5% of plan entries, replay gate).

Exports at steps 25/50/75/100/125 -> val_seen-300 for selection, selected + final -> val_unseen.

## 7. Interim results (2026-10-28, F-B steps 0-40; formal read on all 125 steps pending)

Base S8N on HiPerGator (gate run 0): val_unseen 947/1,839 (SR 51.50, SPL 47.93, 0 failures; reference 964,
inside the pre-set +-28 tolerance), val_seen-300 SR 51.67 / SPL 48.67. All RL deltas are taken against 51.50.
Analysis files: `svln_runs/dev/s8n_v1/analysis/h1_*`, `h2_*`.

- **H1 fails at the early read** (1,147 forks, 327 episodes). event_pos: z = 1.02, quintiles 0.60 / 0.72 /
  0.66 / 0.69 / 0.63 (not monotone). Raw event: z = 3.3 with t/N* only, z = 0.39 with t/N* + log(1 + t), i.e.
  the raw score is a proxy for "early in the episode". The coefficient with position controls is
  0.03-0.05 log-odds per SD, too small to matter even if a larger sample made it significant.
- The pre-registered "informative" target is dominated by the main outcome (reward has a final-distance term:
  96% of forks of successful mains differ, 20% of failed). Exploratory success-flip target (19% of forks):
  event z ~ 0.6; flips are driven by low t/N* (z = -6.7) and are rare near the goal (4% at d/d0 < 0.25,
  ~25% at 0.5-1.0). No single feature lifts the flip rate above ~1.3x the base rate, so fork *placement* has
  little headroom as a lever.
- **H2 proxy holds moderately**: after two forward steps (o_t not a rotation of o_{t-1}), the share of
  top-20% event_pos decisions rises from 0.19 on straight path to 0.32 where the shortest path bends 17-69
  degrees; Spearman(event_pos, bend) = 0.22, (event_pos, progress-slope change) = 0.17. (Positions are not
  logged, so bends come from the privileged bearing error, not the reference path.)
- S8N banks at episode end (episodes >= 30 steps): layers 3/23/27/31 keep 64-81% of frames from t <= 3,
  layers 7/11/15 keep mid-episode frames (median source at 25-37% of the episode).

Reading: S8N's drift marks geometric events (good for the memory claim) but those are not where sampled
alternatives decide the outcome, so the drift-fork coupling of §1 is not supported. Per §4, EBF is not run as
the main method. Run 4 (G=8) moves into slot 3 (submitted: train 48055208, evals 48055209-14).
Open choice (user): (a) bounded-memory tree RL (F-B on S8N + realised-memory replay; needs F-B > G=8), or
(b) RL-learned per-layer retention (MemFork-S8N), motivated by the start-frame bias above; pre-test whether
the policy's actions change when retained frames are swapped before building it.

**Formal H1 read, all 125 F-B steps** (3,598 forks, 997 episodes; `analysis/*_all.json`): pre-registered
statistic event_pos z = 1.88, quintiles 0.625 / 0.705 / 0.668 / 0.672 / 0.650 -> **H1 fails** (needs z > 2 and
monotone). Raw event with t/N* + log(1 + t) controls: z = 1.54. Exploratory success-flip target (16.2% of
forks): event_pos z = 2.42, entropy z = 3.81, flip rate by event_pos quintile 0.146 / 0.146 / 0.161 / 0.186 /
0.171, i.e. at most ~1.2x; this is the stated exploratory direction of §4 (`drift_entropy`), not a confirmed
coupling. H2 proxy unchanged (top-20% event rate 0.19 straight -> 0.29 at 17-34 degree bends).
F-B finished 2026-10-28 12:10 (125 steps, 271,867 environment steps, train SR 0.596 on steps 0-24 -> 0.681 on
steps 100-124, truncation 3.7% -> 11.1%).

## 8. Five-day plan (2026-10-28 -> 11-02, 2 x RTX PRO 6000, serial; fixed before any RL evaluation)

Candidates. (a) **bounded-memory tree RL** = GRPO+F-B on S8N (run 1, trained). (b) **memory twins**
(`memory_twins_s8n_r2r1k.yaml`, `rl/memory_tilt.py`, `stream/sweep_retention.py`): F-B plus an RL-learned
per-layer retention rule priority = (1 - a_l) log p^l + a_l mean_l' log p^l' + b_l log(1 + s) (a = b = 0 is
S8N). Sample 0 of every final group acts with w + sigma eps; its twin re-runs the episode with the same seed
(common random numbers: identical until memory changes an action), no forks, w - sigma eps; the antithetic
return gap gives an ES gradient for w (Adam, identical on all ranks). Twins are not trained on. This targets
MemFork's M5 failure (memory credit drowned by action noise, retention collapsed to the prior).
Measured fragility of S8N's rule (offline replay of logged drift, 159 base episodes): b = +0.1 changes 46% of
the retained frames, +0.3 is almost recency, -0.3 keeps the first frames; a = 1 changes 37%.

Queue: F-B evals -> retention probes on val_seen-300 with fixed tilts (b = +0.1 "balanced", +2 "recency",
-2 "start"; `scripts/rl/make_tilt_checkpoint.py`, inference only) -> twins GPU smoke (2 steps) -> GRPO -> its
evals -> [twins, if D1] -> G=8 -> seeds.

- **D1 (after the probes, before any twins run):** the retention rule matters if some probe arm changes
  val_seen-300 SR by >= 3 pp against base S8N (155/300) or flips success on >= 15% of episodes (paired). Then
  the twins run goes right after GRPO's evaluations (before G=8). Otherwise twins is dropped as a method
  (reported with the probe), and the slot goes to G=8 and second seeds.
- **D2 (method choice, after GRPO, G=8, F-B and twins val_unseen at step 125):** primary metric val_unseen SR
  at step 125, paired McNemar over 1,839 episodes vs ActiveVLN GRPO and vs G=8. A method qualifies only if it
  beats both (p < 0.05). Between qualifying methods a tie is |dSR| < 1 pp or p > 0.05; tie-break in order:
  SR at matched environment steps (G=8 as reference), total interaction cost, coherence with S8N (twins learn
  S8N's own memory rule, so it is preferred on a tie). Checkpoints are only ever chosen on val_seen-300.
- **Seeds:** the chosen method and ActiveVLN GRPO get a second seed in the remaining days; the paper claim is
  made only if the sign holds on both seeds.

**F-B val_seen-300 (paired against base S8N 155/300; `compare_evals.py`).** Steps 25/50/75/100/125:
SR 52.00 / 56.00 / 63.00 / 61.00 / 61.33 (dSR +0.3 / +4.3 / +11.3 / +9.3 / +9.7 pp; bootstrap 95% CI at step 125
[+4.0, +15.3]; McNemar p = 0.0013, 53 gained / 24 lost). Not yet attributable to forking: the ActiveVLN GRPO
baseline is still to run.

**Amendment to D1 (2026-10-28 14:25, before any probe has run).** The success-flip criterion is dropped: step
25 of F-B changes SR by only +0.3 pp yet flips success on 59/300 = 19.7% of episodes, so the evaluator flips
~20% of outcomes under any small policy change. A flip rate of 15% therefore indicates chaotic sensitivity,
not that retention matters on average. D1 is now: some probe arm changes val_seen-300 SR by >= 3 pp against
base. Reading: if only the extreme arms (recency / start) move SR and balanced (+0.1) does not, the drift rule
matters but S8N sits near a local optimum along the birth axis; twins is still run (it can learn per-layer
and consensus directions), and this is said in the paper as the reason the learned rule moves little.

**F-B val_unseen (step 125, 1,839 episodes, 0 failures).** SR 60.63 / SPL 54.36 / NE 4.66 / OSR 67.92 against
base S8N 51.50 / 47.93 on the same evaluator: dSR +9.14 pp (bootstrap 95% CI [+6.85, +11.47]), dSPL +6.43,
325 gained / 157 lost (McNemar p < 1e-4). The D2 comparison with ActiveVLN GRPO and G=8 is pending.

**D1 result (2026-10-28 17:38; inference-only tilts on base S8N, val_seen-300, paired against 155/300).**

| arm | SR | SPL | dSR (95% CI) | gained / lost | McNemar p | actions differ |
|---|---|---|---|---|---|---|
| balanced b = +0.1 | 54.00 | 50.27 | +2.33 (-2.0, +6.7) | 25 / 18 | 0.36 | 77% |
| recency b = +2 | 53.00 | 49.53 | +1.33 (-3.7, +6.7) | 33 / 29 | 0.70 | 89% |
| start b = -2 | 57.33 | 53.17 | **+5.67 (+0.7, +10.7)** | 39 / 22 | 0.040 | 83% |

D1 passes (start arm, >= 3 pp), so the twins run is queued right after GRPO's evaluations, before G=8.
Readings, with care: (i) one arm of three at p = 0.04 (Bonferroni 0.12), so the start effect is confirmed on
val_unseen (queued) before it is stated; (ii) no arm is worse than S8N's own drift rule, including pure
recency, so the policy is robust to which frames it keeps and S8N's drift rule is not optimal for the
policy it was trained with; keeping episode-start frames (the instruction's first landmarks) helps.
Consequence for D2: memory twins must beat F-B with the *fixed* start tilt (inference only on F-B's step
125, `probes/fb125_tilt_start`), otherwise "RL learns the memory rule" reduces to "keep the first frames".

Queue (submit_twins_chain.sh, snapshot v4): GRPO chain -> base S8N val_seen-300 on v4 (must reproduce the
base journal exactly) -> F-B step 125 + start tilt val_seen-300 -> memory twins train + evals -> base +
start tilt val_unseen -> G=8.

**Follow-ups (2026-10-28 19:40).**
- Code-path check: base S8N on snapshot v4 (no tilt file) reproduces the base val_seen-300 journal exactly
  (300/300 episodes, identical actions). The probe differences are caused by the tilt alone.
- F-B step 125 + fixed start tilt (b = -2, inference only), val_seen-300: SR 59.33 / SPL 54.61, against F-B
  alone 61.33 / 56.81: dSR -2.0 (95% CI [-6.7, +2.7]), 24 gained / 30 lost, p = 0.50. The start rule that
  helps base S8N (+5.7) does not transfer to the RL policy, which was trained with S8N's drift rule. The best
  retention rule depends on the policy, which is the case for learning it jointly with the policy (memory
  twins) rather than fixing it after training. The D2 bar for twins is therefore F-B itself (the higher of
  F-B and F-B + start).
- GRPO 48037316 died at the step-10 save (group /blue quota full); resumed from step 5 as 48134830. Step 5
  re-ran identically (same rollouts, rewards, env steps); its update differs in the 4th digit of the grad
  norm (2.5716 vs 2.5713; non-deterministic GPU reductions), so later steps diverge as any re-run would.
  Runs are reproducible in distribution, not bitwise; this is one more reason for the second seeds.

**D2 for (a) bounded-memory tree RL (2026-10-29 07:33).** ActiveVLN GRPO on S8N, step 125, val_unseen:
SR 60.41 / SPL 54.66 (base 51.50 / 47.93: dSR +8.92, CI [+6.58, +11.20]). F-B against GRPO, paired over 1,839
episodes: SR 60.63 vs 60.41, dSR +0.22 (95% CI [-2.01, +2.45]), dSPL -0.30, 219 gained / 215 lost, McNemar
p = 0.89. On val_seen-300 F-B led at steps 75 / 100 (+3.0 / +4.7, both n.s.) and tied at 125 (+0.3). F-B used
596k environment steps (272k main + 324k branch) against GRPO's 269k, and 12.4 h against 8.1 h of
training. **F-B does not qualify; forked sibling credit adds nothing over GRPO on S8N.** With H1 failed as
well, (a) is reported as a negative result.

**Amendment to D2 (2026-10-29 07:50, before any twins result).**
- Since F-B = GRPO, the twins method is stated on plain GRPO: `memory_twins_grpo_s8n_r2r1k.yaml` (GRPO +
  memory_tilt, snapshot v5 = v4 code). Primary comparison: twins-GRPO vs ActiveVLN GRPO, paired val_unseen at
  step 125. The pre-registered F-B + twins run (`memory_twins_s8n_r2r1k`, running) gives a second paired
  estimate of the twins effect, against F-B.
- Compute control: G = 5 GRPO (`activevln_grpo_g5_s8n_r2r1k.yaml`) replaces G = 8. Twins runs 4 trained
  samples + 1 untrained twin per group, so G = 5 matches its rollouts while training on all of them. G = 8 was
  the control for F-B's 2.2x environment steps; it stays queued last and is optional.
- Qualification: twins must beat GRPO and G = 5 (McNemar p < 0.05 each) and not lose to F-B. Then a second
  seed of twins-GRPO and of GRPO; the claim needs the same sign on both seeds.

**Interruption (2026-10-29 08:47:03).** A UF Research Computing admin account cancelled all jobs of the user
(running start-rule probe at 259/1,839, the queued twins-GRPO / G = 5 / G = 8 chains, and ~50 stale jobs) and
the login session ended. No reason in Slurm. Nothing on disk was lost. Resubmission waits for the reason;
`svln_runs/dev/s8n_v1/restart_after_cancel.sh` (STAGE=twins, then STAGE=g5) queues one stage at a time with
afterany dependencies only, and eval_r2r.sbatch now exits at once on a missing export.

## 9. Paper story (draft, 2026-10-29; to be revised with the twins result)

One contribution, two halves: **what a bounded streaming memory keeps should be learned with the policy, by
RL, and doing so needs credit assignment at the level of the memory rule, not of actions.**

1. *S8N (stage 1/2).* Streaming VLN with constant memory: R = 4 recent groups + per-layer top-4 frames ranked by
   the FP32 drift of the GDN state; 51.5 SR val_unseen (this evaluator).
2. *Finding: the retention rule is policy-dependent.* Inference-only changes of the rule move base S8N by up
   to +5.7 SR (keep episode-start frames; val_seen-300, p = 0.04; val_unseen confirmation pending), while the
   same rule does nothing for the RL policy (-2.0, n.s.), and no tested rule is worse than S8N's own. The rule
   matters, and the right rule depends on the policy, so it should be learned jointly with it.
3. *Method: memory twins.* Per-layer retention priorities (consensus a_l, birth b_l; 0 = S8N) learned
   during GRPO by antithetic ES on common-random-number twins: a twin re-runs its main's episode with the same
   action random stream and the opposite perturbation, so the two differ only through memory (pilot: 60-90% of
   pairs diverge, ~12% flip success). Cost: one untrained extra rollout per group, no extra backward pass.
4. *Negative results that motivate (3).* Action-level tree credit (F-B forking) ties GRPO (+0.2 SR, p = 0.89) at
   2.2x environment steps; event-placed forks (H1) are not more informative than random ones; MemFork-style
   per-decision memory credit (M5) collapsed to the prior. RL on S8N is limited by memory credit, not action
   credit.

Required evidence (in this order once jobs run again): twins-GRPO vs GRPO and vs G = 5 on val_unseen
(McNemar p < 0.05) -> start-rule probe on val_unseen -> second seeds of twins-GRPO and GRPO -> ablation
**twins without common random numbers** (twin with an independent seed: tests the central claim that CRN
is what makes memory credit learnable) -> the learned rule per layer (interpretation). Approximate GPU
hours: 12.5 + 2 + 12.5 + 23 + 12 = 62 h.

If twins-GRPO does not beat GRPO, the paper loses its method and (2)-(4) remain an analysis of RL with
bounded memory; that is reported as such and not dressed up.
