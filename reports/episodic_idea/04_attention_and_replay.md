# 04 — Episodic attention, layer-major forward and serving replay (Tasks 4–6)

Status: **CPU-tested** and **GPU-tested** (real pinned model, one H100). Numerical replay
only; no closed-loop navigation claim.

## Attention (tests/vln/test_episodic_attention.py — 12 tests, all pass)

FP64 dense-mask oracle (prefix causal; each query sees prefix + its plan's rows + the R
most recent groups, causal in its own group), uneven groups, GQA 4Q/2KV, arbitrary plans
from the real plan builder:

* Configurations R8/B0 (batches 1, 5), R9/B0 (16), R8/B6 (4), R4/B12 (3), R3/B5 (64),
  R2/B30 (7, partial final batch): outputs `1e-10`, all Q/K/V gradients `1e-9`.
* Repeated old rows receive the summed gradient of every later window (dense oracle);
  a loss restricted to the final group reaches an old episodic K row that is outside the
  recent window.
* Saved storage ≤ original Q/K/V (no all-episode duplicated windows).
* V0 (R8/B0) through the new function equals the legacy batched Window8 function
  **bitwise** in the forward and to `1e-12` in gradients.
* Plan routed to the wrong layer, plan without the episodic window, and a tampered digest
  (checked again in backward) fail.
* Real FA2 on H100 (BF16, 16Q/4KV, d=128, 33 groups): new path vs serial legacy, batches
  4/8/16: outputs within `0.02/0.015`, gradient relative norm error < 2%.

## Layer-major forward (tests/vln/test_episodic_forward.py — 8 tests, all pass)

Tiny real Qwen3.5 text model (2 GDN + 2 attention layers, FP32 CPU, native torch
fallbacks, 300 visual rows/group):

* **Offline training forward == incremental serving** (session state machine, promotion
  before each observation, single-token responses, EOS+separator close) for R8/B0, R3/B0,
  R3/B300 committed, R2/B450 committed, R3/B300 random: hidden states `2e-4/2e-5`;
  serving episodic sets equal the offline plan of the last group; resident source scores
  equal offline scores.
* Project forward with B0/R8 equals the **native** HF text model with legacy Window8
  attention **bitwise**.
* Gradient checkpointing on vs off: identical outputs, gradients `1e-6`; the final-group
  loss gives nonzero gradient to group-0 visual inputs (old episodic rows + GDN path);
  changing future inputs leaves all earlier outputs bitwise unchanged.
* Two episodes forwarded before either backward (interleaved): plans stay intact
  (digests) and gradients equal a non-interleaved reference.

## Real pinned model gates (H100, `quals/interactive_894903/check_episodic.json`, PASS)

**A. Feature disabled.** Real 24-step R2R episode (7,701 tokens): V0 (`episodic_window_v1`,
R8/B0) vs legacy `window8` — vocabulary logits and loss **bitwise equal**.

**C. Offline vs real `StreamSession` text loop** (forced canonical responses, real
frames, final label set to STOP only for the replay fixture). Inherited BF16 budget
`atol .75, rtol .03, RMS ≤ .15`, declared before measurement:

| Arm | Dataset | Steps | Max logit err | RMS | Argmax agree | Decoded actions | Max KV / guard |
|---|---|---:|---:|---:|---:|---:|---:|
| V0 | R2R | 64 | 0.453 | 0.041 | 100% | 100% | 2,650 / 5,120 |
| M2 | R2R | 64 | 0.474 | 0.040 | 99.5% | 100% | 2,950 / 5,420 |
| M2c | R2R | 64 | 0.453 | 0.039 | 99.5% | 100% | 2,950 / 5,420 |
| V0 | RxR | 64 | 0.438 | 0.035 | 99.0% | 100% | 2,715 / 5,120 |
| M2 | RxR | 64 | 0.438 | 0.036 | 100% | 100% | 3,015 / 5,420 |
| M2c | RxR | 64 | 0.438 | 0.037 | 99.0% | 100% | 3,015 / 5,420 |
| V0 | R2R | 128 | 0.594 | 0.038 | 99.5% | 100% | 2,628 / 5,120 |
| M2 | R2R | 128 | 0.594 | 0.040 | 99.7% | 100% | 2,928 / 5,420 |
| M2c | R2R | 128 | 0.594 | 0.038 | 99.5% | 100% | 2,928 / 5,420 |
| V0 | RxR | 128 | 0.750 | 0.038 | 100% | 100% | 2,723 / 5,120 |
| M2 | RxR | 128 | 0.750 | 0.039 | 99.7% | 100% | 3,023 / 5,420 |
| M2c | RxR | 128 | 0.750 | 0.038 | 100% | 100% | 3,023 / 5,420 |

All argmax disagreements are retained in the JSON. V0 RxR-128 reaches the 0.75 absolute
boundary (passes only through the declared `rtol` term); recorded, not loosened.
Selected-ID agreement: M2c 0 mismatches in every comparison; M2 see 03 (score skew).

**D.** Every step: repeated `observe(t)` returns the saved result with no promotion or
state change; resident rows == prefix + |E| + recent groups (no history archive);
KV ≤ derived guard; reset clears the bank.

**E. ≥500-step cyclic real-RGB dry stream** (forced cyclic actions, 40 real frames):

| Arm | Steps | Peak resident KV | Guard | Wall | s/step | Peak alloc |
|---|---:|---:|---:|---:|---:|---:|
| M2 | 520 | 2,946 | 5,420 | 85.9 s | 0.165 | 14.5 GiB |
| V0 | 520 | 2,646 | 5,120 | 66.8 s | 0.128 | 14.5 GiB |

The cache is bounded (prefix + 300 + 8 groups) and does not grow with episode length.
M2 serving overhead in this replay (score extraction, sort, repack) ≈ +29% per step;
this is a dry-stream timing, not the evaluator's closed-loop latency.
