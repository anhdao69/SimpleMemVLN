# Rescue twins: counterfactual expert interventions from the agent's own streaming memory (design, 2026-10-30)

Pre-registered before any rescue training result. Results go to `s8n_local_campaign.md` §5.
Branch `RL_s8n_dev` (from `RL_s8n` 334bc35); estimator `grpo_rescue`.

## 1. Problem

After ActiveVLN GRPO, S8N's val_unseen failures (728 of 1,839, HiPerGator journal) are mostly agents that
leave the route and never get within 3 m of the goal (418) or wander until the cap (136); stopping errors
are a minority (122 passed the goal). Outcome-only GRPO gives a failed trajectory one negative scalar spread
over ~90 decisions; it never shows the agent what recovering from its own mistake looks like. Prior
credit-assignment refinements on S8N gave nothing (GRPO+F-B = GRPO, p = 0.89; event-placed forks and
surprise-placed interventions are not supported by H1 or by the deviation analysis of 2026-10-30).

## 2. Method

For each trained main trajectory (plain ActiveVLN GRPO, unchanged):

1. **Trigger (causal, privileged):** the first decision t >= 1 at which the geodesic distance before a_t is
   >= delta = 1 m above the closest approach so far. It uses only poses before a_t.
2. **Snapshot:** the complete S8N streaming state before a_t (GDN/conv states, attention K/V of prefix,
   per-layer frame banks and recent groups, scorer states, realised-memory record), O(1) in t.
3. **Intervention twin:** after the main ends, restore the snapshot, replay the simulator prefix (verified
   bit-exact), let Habitat's greedy geodesic follower act for <= 16 decisions until its 3rd forward move,
   streaming its actions into the policy's *own* memory as forced responses, then let the policy continue
   on-policy (independent generator) with the main's remaining budget.
4. **Credit** (s = batch reward std, floor 0.75, as GRPO+F-B):
   - forced expert decisions: max(0, R_twin - R_main) / s — imitate a correction only when it beat the
     policy's own continuation from the same state (a Q-filter estimated by a counterfactual rollout;
     cf. Nair et al. 2018's Q-filtered behaviour cloning, here without a learned critic);
   - the policy's own decisions after the takeover: (R_twin - R_main) / s, an admissible baseline since the
     main's continuation was sampled independently;
   - branch prefix (the main's decisions < t): 0. Branches with zero coefficients are not trained.

Why streaming memory matters: the intervention starts from the exact belief the agent had when it went wrong
(no re-encoding, ~230 MiB snapshot), and the expert's segment becomes part of the agent's own memory, which
the trainer replays exactly (realised-memory replay), so the policy learns recovery under the memory it will
actually have at test time.

Probes (no training, base S8N): a third of hard failures become successes (6/16, 37.5 %), takeover ~14
decisions; never made a triggered success worse (0/1, 0/2).

## 3. Runs (R2R 1k, 125 steps, seed 20261010, identical to ActiveVLN GRPO otherwise)

| # | config | role |
|---|---|---|
| R1 | `rescue_grpo_s8n_r2r1k` | the method |
| C1 | `activevln_grpo_g5_s8n_r2r1k` (local) | compute control (5 rollouts per group), shared with memory twins |
| A1 | rescue with `imitation: none` | is verified imitation needed, or only the post-intervention RL credit? |
| A2 | rescue with unverified imitation (constant coefficient) | is the outcome filter needed (vs DAgger-style imitation)? |

A1/A2 are run only if R1 qualifies (§4).

## 4. Hypotheses and decision rules (fixed now)

- **H-R1 (main):** R1 > ActiveVLN GRPO on val_unseen SR at step 125. Paired against a local GRPO-family
  journal (C1) with McNemar p < 0.05; against the HiPerGator GRPO (60.41) only descriptively (unpaired).
- **H-R2 (mechanism):** R1 reduces "never within 3 m" + cap failures relative to C1 (failure taxonomy of
  `compare_evals.py`), more than it changes "passed goal then stopped outside".
- **H-R3 (signal):** the training-time rescue rate of triggered failed mains rises over training (the policy
  continues better after corrections), and the trigger rate of mains falls.
- Checkpoint selection only on val_seen-300; val_unseen once at step 125.
- If R1 does not beat C1, rescue is reported as a negative result with its probes and training statistics.

## 5. Revision 2026-10-31: rewind (after R1 was stopped at step 22)

R1 (fork at the trigger t) made the policy spin: 79 % of the forced expert decisions were turn-arounds from
states where the agent is already lost, whose direction depends on the goal position the agent cannot see;
verified imitation of them raised the mains' turn share 0.36 -> 0.53 and truncation 0.03 -> 0.23 in 22 steps
(paired against run 1 on the same prompts; `s8n_local_campaign.md` 2026-10-31 05:15). R1 is reported as stopped,
with these statistics.

**R1' (rewind, `rewind_grpo_s8n_r2r1k`)**: the same trigger, but the branch forks from the main's closest
approach k* = first argmin_{k<t} d_k, from the S8N state snapshotted there (retaken at every new strict minimum;
one constant-size clone). The expert then re-decides the decisions where the trajectory left its best point,
from on-path states in the agent's own memory, and its STOP is a forced decision (the correction of "passed the
goal, stopped outside"). Credit unchanged (verified imitation of forced decisions, signed contrast on the
policy's continuation, mains = GRPO). Guarded by `ops/local/drift_guard.py`.

Hypotheses (fixed before R1' starts): H-R1 and H-R2 as in §4 with R1' in place of R1 (control C1 = run 3,
GRPO G = 5); additionally **H-R4**: the forced decisions of R1' have a turn share close to ordinary navigation
(< 0.5, vs 0.79 for R1) and R1' never trips the drift guard. Ablation (only if H-R1 holds):
`rewind_noimit_grpo_s8n_r2r1k` (forced decisions uncredited). If R1' is ahead of C1 but n.s., second seeds of
R1' and C1 (post hoc, reported as such).
