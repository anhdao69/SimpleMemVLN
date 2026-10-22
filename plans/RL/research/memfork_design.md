# MemFork — learning what to remember and how to act from forked memory states

Status: **design + implementation in progress** (2026-10-22). Nothing in this document is a result.
Branch `RL_m5` (from `m5_gdn_s_sweep` @ fc017d9 + the RL package of `RL_research`).

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

with x_j a frame descriptor fixed at the frame's birth: F1_v2 GDN channels (log β, log(−g), log‖v‖,
log‖e‖, log‖v̂‖, cosine error, log β‖e‖), averaged over the frame's 300 visual tokens and the 32 value
heads, for each of the 8 scoring GDN layers (2, 6, …, 30): 56 numbers, computed by the same native side
pass as M5's scorer. The retained frame set is shared by all 8 attention layers (1,200 rows = 4 frames).

  u_φ(x) = κ · ( log s7(x) + g_φ(x) ),   s7(x) = frame mean of β‖e‖ averaged over layers,

with g_φ a 56→32→1 MLP whose last layer is zero-initialised and κ > 0 learnable. At initialisation the
policy is a tempered frame-level M5 (S7-family) rule; κ → ∞ recovers deterministic running top-4.
Actions follow the unchanged policy a_t ~ π_θ(· | z_t) (tempered text decoding). The joint
trajectory distribution is p(τ) = Π_t q_φ(m_t | z_t) π_θ(a_t | z_t) P(o_{t+1} | o_{≤t}, a_{≤t}).

Objective: J(θ, φ) = E_τ[R(τ)] with ActiveVLN's terminal reward (unchanged).

  ∇J = E[ Σ_t Â_t ( ∇_θ log π_θ(a_t | z_t) + ∇_φ log q_φ(m_t | z_t) ) ].

Features x_j are detached functions of θ (as M5's priorities), so the φ-gradient is the exact score
function of the sampled memory decisions and the θ-gradient is unchanged in form.

## 3. Fork-consistent credit (the bridge)

GRPO+F-B snapshots the streaming memory before a decision t and samples M continuations from it. With
memory decisions in the policy, the snapshot is z_t (including the bank, the pending candidate and the
scorer states; `rl/sampling.clone_episodic`), and each branch samples **both** its memory decisions
m_{t:} and actions a_{t:} afresh. Given (z_t, simulator state at t), the main suffix and the M branch
suffixes are i.i.d. draws of the same conditional process of the joint policy. The sibling baseline of
each suffix is therefore independent of that suffix's own decisions, and the F-B estimator, with the
same 1/(1+M) averaging and batch scale, is unbiased for ∇_θ J **and** ∇_φ J. Root decisions keep GRPO's
group advantage.

What would be biased (and is excluded, tested by enumeration):
- branching only the memory while replaying the main's actions (the counterfactual "same actions,
  different memory" contrast): suffix actions are then not drawn from π given the branch's memory;
- crediting a memory decision with a contrast whose baseline was computed from branches that share its
  realised value (e.g. forking after the admission but crediting the admission).

Consequence: two siblings from the same z_t that keep different frames and reach different returns give
an on-policy, outcome-level estimate of the value of remembering — the L4-return label of the plan,
obtained without a separate labelling campaign and without the local-LOO proxy.

Fork placement (causal, as F-B): the fork decision is the first t in [τ, τ + 8] whose action entropy or
memory-decision entropy exceeds the previous step's median (pre-drawn τ). Using memory entropy places
forks where the agent is uncertain about what to keep.

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

Ablations (each isolates one claim):
- A1 frame-level greedy prior, frozen (κ fixed large, g_φ = 0) + F-B: effect of frame-level memory alone.
- A2 learned memory, GRPO credit only (no forks): are forks needed for memory credit?
- A3 learned memory + forks placed by action entropy only: memory-aware fork placement.
- A4 MemFork with random retention prior (u ≡ 0, uniform drops) — memory content vs. capacity.
Matched interaction: report env/policy steps; GRPO G = 8 on M5_E1 if F-B-style runs exceed GRPO's.

Development only on the val_seen-300 subset; val_unseen once per final checkpoint; paired tests over
episodes; two seeds for the final comparison if time allows.

Pilot gate before the full run: closed-loop val_seen-300 of M5_E1 with the frame-level greedy prior vs
its native token-level S0 (does frame-level retention keep M5_E1's skill?).

## 6. Analysis plan

Memory behaviour: which frames are kept (age, turn/landmark frames, GDN surprise rank), retention entropy
over training, drop decisions credited positively vs negatively, overlap with S0/S7/S8 choices; success
vs. whether the instruction's referenced landmarks were retained (when annotations allow); credit mass on
memory decisions by group type (all-fail / mixed / all-success).
