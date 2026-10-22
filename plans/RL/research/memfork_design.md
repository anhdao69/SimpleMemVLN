# MemFork — learning what to remember and how to act from forked memory states

Status: **implemented, CPU-tested; no GPU run yet** (2026-10-22). Nothing in this document is a result.
Code on branch `RL_memfork` (from `RL_m5` = `m5_gdn_s_sweep` @ fc017d9 + the RL package of `RL_research`).

## 1. Problem

M5 (`episodic_window_v1`, R = 4, B = 1,200) gives each full-attention layer the instruction, the 4 most
recent observation groups and a bank of 1,200 old visual K/V rows. Bank membership is decided once per
admission (before observation t, source t − R expires) by a fixed GDN signal, the committed write
mean_h β‖v − S̄ᵀk‖ of the preceding GDN layer, computed under `no_grad` (S0). M5 beats random retention by
+3.3 SR (M5 vs M5c), so *what is remembered matters*. But:

- SFT cannot learn retention: priorities are detached and top-B is discrete.
- The S-sweep only compares fixed GDN statistics; the L-series fits rankers to local leave-one-out
  attention labels, which the plan itself calls proxies for navigation consequences (§12.1, §13.1).
- §13.1: branches with different memories "do not inherit GRPO+F-B's fixed-state sibling-baseline
  argument … unless a correct joint retention/action policy objective is explicitly derived".

MemFork derives that objective and optimises it.

## 2. Agent state and joint policy

Agent state before decision t: z_t = (GDN recurrent + conv states, attention K/V of prefix/recent groups,
bank F_t = set of ≤ 4 retained source frames, committed-write/feature scorer states, position ledger).
z_t is a deterministic function of the history h_t = (o_0, a_0, …, o_t) **and** of the past memory
decisions m_{<t}.

At each admission with a full pool P_t = F_{t−1} ∪ {s = t − R}, |P_t| = 5, the memory policy drops one frame:

  q_φ(m_t = j | z_t) = softmax_j( −u_φ(x_j) ),   j ∈ P_t,

with x_j a frame descriptor fixed at the frame's birth: for each of the 8 attention layers l, x_jl is the
log of the frame mean of the 300 FP32 committed-write scores mean_h β‖e‖ that M5's own scorer computes
from the preceding GDN layer (2, 6, …, 30), i.e. S7's per-layer frame statistic. (Implemented version;
the richer F1_v2 per-head channels need the GDN-sweep side pass and are left for a later variant.) The
retained frame set is shared by all 8 attention layers (1,200 rows = 4 frames).

  u_φ(x) = κ · ( mean_l x_l + g_φ((x − shift)/scale) ),

with g_φ an 8→32→1 MLP whose last layer is zero-initialised and κ > 0 learnable. At initialisation the
policy is a tempered frame-level M5 rule; greedy decoding (evaluation) keeps the running top-4 frames by
u (ties keep the older frame, as M5). shift/scale and κ₀ = spread / (mean within-pool std of mean_l x_l)
come from 4 greedy step-0 training episodes per rank (`rl/memfork.calibrate`; spread 2.0 in the configs).
Actions follow the unchanged policy a_t ~ π_θ(· | z_t) (tempered text decoding). The joint
trajectory distribution is p(τ) = Π_t q_φ(m_t | z_t) π_θ(a_t | z_t) P(o_{t+1} | o_{≤t}, a_{≤t}).

Objective: J(θ, φ) = E_τ[R(τ)] with ActiveVLN's terminal reward (unchanged).

  ∇J = E[ Σ_t Â_t ( ∇_θ log π_θ(a_t | z_t) + ∇_φ log q_φ(m_t | z_t) ) ].

Features x_j are detached functions of θ (as M5's priorities), so the φ-gradient is the exact score
function of the sampled memory decisions and the θ-gradient is unchanged in form.

## 3. Forking the memory: fork-consistent joint credit (the bridge)

GRPO+F-B snapshots the streaming state before an uncertain *action* and samples M continuations from it.
MemFork forks the **memory**: the fork point is an uncertain *memory decision*.

- Placement (causal): pre-drawn target τ ~ U[R+F, N*) (the first memory decision is at step R + F = 8);
  the fork is the first t ∈ [τ, τ + 8] whose memory-decision entropy H(q_φ(·|pool_t)) is at least the
  previous RL step's median memory-decision entropy. H is computed from the pool before m_t is drawn
  (`EpisodicSessionState.pending_decision_entropy`), so the rule never sees m_t or a_t.
- Snapshot *before* m_t (`before_memory` hook in `rl/rollout._advance`): z_t^- = GDN/conv states,
  attention K/V, bank, recent groups and scorer states (`rl/sampling.clone_episodic`), O(state) device
  copies. Each branch re-draws m_t from its own generator, then a_t, m_{t+1}, a_{t+1}, … on-policy.
- Given z_t^- and the simulator state, the main suffix and the M branch suffixes are i.i.d. draws of
  the joint policy's conditional process, so the sibling baseline of each suffix is independent of that
  suffix's own decisions and the F-B estimator (1/(1+M) averaging, batch scale) is unbiased for ∇_θ J and
  ∇_φ J. Credit (`rl/hico.memory_coefficients(..., snapshot_before_memory=True)`): m_s takes the
  coefficient of a_s — root advantage before the fork, the sibling contrast (R_main − R_branch)/scale
  from the fork on, zero on a branch's replayed prefix.
- Siblings from the same z_t^- that keep different frames and reach different returns are a direct,
  on-policy, outcome-level estimate of the value of a retention choice — the "L4-return" label of the
  M5 plan, obtained inside RL without a labelling campaign or the local-LOO attention proxy.

Verified by exact enumeration of a toy memory POMDP in which memory decisions change the action
distribution (`tests/rl/test_memfork_estimator.py`), for both placements: memory forks with same-step
credit and action forks (snapshot after m_t, m_s credited like a_{s−1}) give E[estimator] = ∇J for θ and
φ (rtol 1e-5). Detectably biased: sibling credit for a memory decision shared by the siblings (action
forks with same-step credit), double-counting a re-drawn m_t (root credit on the main + sibling credit on
the branch), and dropping the 1/(1+M) averaging. Rollout ordering (snapshot strictly before m_t; branches
re-draw m_t) is tested on a fake session and mutation-checked (`tests/rl/test_fork_rollout.py`).

Action-entropy placement with a snapshot after m_t is ablation A3 (`memfork_actfork_m5_r2r1k`).

## 4. Why the hybrid GDN architecture makes this practical

- Forking needs an O(state) snapshot. GDN state is constant-size (24 layers × 32 × 128 × 128 FP32 plus
  conv tails), the bank is bounded (≤ 1,200 rows/layer) and the recent window is R = 4 groups, so a fork
  costs a few hundred MB of device copies, independent of episode length; no re-encoding of history.
- Memory features come from the scorer side pass M5 already runs; the controller has ~2k parameters.
- The trainer replays the exact realised bank (injected plan), so action log-probs are computed under
  the memory the policy actually acted with (no BF16 near-tie plan flips between rollout and trainer).

## 5. Experiments (all R2R, 1k ActiveVLN subset, 125 steps, G = 4, unchanged evaluator)

Main comparison (M5_E1 initialisation):
1. M5_E1 (SFT) — local evaluation.
2. ActiveVLN GRPO on M5_E1 (token-level S0 memory).
3. GRPO+F-B on M5_E1 (token-level S0 memory, action forks).
4. **MemFork** (learned frame retention + joint fork credit).

Ablations (each isolates one claim; queued in `scripts/rl/chain_m5.sh` on RL_m5):
- A1 `frame_greedy_fb_m5_r2r1k`: frame-level greedy prior, q_φ frozen, + F-B — frame memory alone.
- A2 `memfork_grpo_m5_r2r1k`: learned retention with GRPO credit only (no forks) — are forks needed?
- A3 `memfork_actfork_m5_r2r1k`: learned retention with F-B's action forks — does forking the memory matter?
- Matched interaction: `grpo_g8_m5_r2r1k` (GRPO, G = 8 on M5_E1).
Not queued (compute): random-retention prior; second seeds.

Development only on the val_seen-300 subset; val_unseen once per final checkpoint; paired tests over
episodes; two seeds for the final comparison if time allows.

Pilot (development, val_seen-300, same 300 episodes as the Window8 pilots): M5_E1 native token-level S0
memory vs M5_E1 with the frame-level greedy prior (`scripts/rl/make_frame_prior_checkpoint.py`) — does
the frame-level memory keep M5_E1's skill before RL?

## 6. Analysis plan

Memory behaviour: which frames are kept (age, turn/landmark frames, GDN surprise rank), retention entropy
over training, drop decisions credited positively vs negatively, overlap with S0/S7/S8 choices; success
vs. whether the instruction's referenced landmarks were retained (when annotations allow); credit mass on
memory decisions by group type (all-fail / mixed / all-success).
