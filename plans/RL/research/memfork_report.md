# RL on the M5 episodic-memory policy: GRPO, GRPO+F-B and MemFork

Status: **IN PROGRESS — no M5 RL result exists yet** (2026-10-22). Every table cell below is filled only
from a finished run and its single val_unseen evaluation; pending cells say so.

Method: `memfork_design.md`. Window8 predecessor (GRPO+F-B): `hico_po_report.md`.

## 1. Protocol (identical for every row)

- Initial policy: M5_E1 (HF `anhdao69/Streaming_VGGT` `M5_E1/epoch-1`, revision 9af25919…;
  `pytorch_model.bin` sha256 92089fa1…, verified locally). `episodic_window_v1`, R = 4, B = 1,200.
- RL data: the 1k prefix of ActiveVLN's fixed-order R2R 4K list (subset sha256 ba062de3…), 125 steps
  × 8 instructions, G = 4 (G = 8 for the matched-interaction control), T = 1.2, ActiveVLN reward,
  dynamic sampling, LR 1e-6, clip 0.2/0.28, seed 20261010 — unchanged from the Window8 runs.
- Evaluation: the unchanged SimpleMem R2R-VLNCE evaluator (Habitat 0.2.4, 500-action cap), val_unseen
  (1,839 episodes), **once per final checkpoint (step 125)**; MemFork checkpoints evaluated with greedy
  retention. Development decisions only on the fixed val_seen-300 subset. All M5 journals are produced
  from one checkout (identical evaluation sources); paired tests over episodes vs the local M5_E1 journal.
- Compute: 2 shared GPUs (≈95 GB each) on a shared host; env/policy steps reported per run.

## 2. Runs (queue: `scripts/rl/chain_m5.sh`, unit `rl-chain-m5`)

| Run | Config | Status |
|---|---|---|
| M5_E1 (no RL), local val_unseen | — | pending |
| val_seen-300 pilot: M5_E1 native vs frame-level prior | — | pending |
| ActiveVLN GRPO | `activevln_grpo_m5_r2r1k` | pending |
| GRPO+F-B | `hico_grpo_fork_bs_m5_r2r1k` | pending |
| **MemFork** (learned retention, memory forks) | `memfork_m5_r2r1k` | pending |
| A3 MemFork with action forks | `memfork_actfork_m5_r2r1k` | pending |
| A2 MemFork-GRPO (no forks) | `memfork_grpo_m5_r2r1k` | pending |
| A1 frame-greedy memory + F-B (retention not learned) | `frame_greedy_fb_m5_r2r1k` | pending |
| GRPO G = 8 (matched interaction) | `activevln_grpo_g8_m5_r2r1k` | pending |

## 3. Results (val_unseen, step 125)

Pending — no run has finished.

## 4. Memory behaviour

Pending (`scripts/rl/analyze_memfork.py` on the MemFork training records).

## 5. Verification done so far (CPU)

- Joint estimator unbiased for θ and φ by exact enumeration, both fork placements; biased variants
  detected (`tests/rl/test_memfork_estimator.py`).
- Tiny real Qwen3.5: streaming with stochastic retention == offline training forward with the injected
  realised banks; greedy retention == running top-F frames; fork snapshots independent; pending
  decision entropy == drawn entropy (`tests/rl/test_memfork.py`, `tests/rl/test_m5_snapshot.py`).
- Mutation checks: shallow snapshot, shallow bank clone, off-by-one bank injection, snapshot after m_t
  under memory forks — each makes its test fail.
