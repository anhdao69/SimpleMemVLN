# 03 — Committed-write score and selector validation (Tasks 2–3)

Status: **CPU-tested** (FP64/FP32 references) and **GPU-tested** on the real pinned model.
Not a navigation result.

## Selector (tests/vln/test_episodic_selector.py — 13 tests, all pass)

* Running top-B equals an independent batch top-B over all prefix-eligible rows for
  **every** group of episodes with 1, 8, 9, 10, 23, 17 (R4/B1200), 14 (B1000) and 12
  (R3/B50) groups, with heavily tied quantized scores; counts are
  `min(B, max(0, t−R+1)·300)`, rows never reach the recent window, IDs strictly ascending.
* Tie-break priority desc, original ID asc (explicit fixture).
* Changing scores of groups ≥12 leaves every plan for groups ≤19 unchanged (R8): no future
  dependence. B0 needs and accepts no scores.
* Random keys: identical across chunking (37-token chunks vs whole), 600/600 unique,
  <1% equal across layers; per-layer identities differ with equal counts; digests
  reproducible; the random control refuses write scores.
* Fail-closed: NaN scores, non-FP32 scores, duplicate candidates, plan tampering
  (read-only arrays; digest re-verified before backward), malformed image geometry.

## Committed delta (tests/vln/test_committed_write_reference.py — 7 tests, all pass)

* FP64 WY block scan versus an independent token-by-token recurrence oracle with nonzero
  initial state, variable decay, 4 value heads over 2 key heads (GVA), **unequal key/value
  widths (5 vs 3)**, three different segmentations: scores and final state agree at
  `1e-10`.
* β=0 → zero scores; β=1 matches oracle; half β halves the first delta (no β²);
  near-zero key; very strong decay (`g=−80`, `exp` underflow) finite and exact.
* An earlier (prefix/text) write changes later scores but never earlier ones.
* Preparation equals the native mixer's tensors captured from a real `Qwen3_5DecoderLayer`
  (k after FLA-formula L2 norm, v, g, β; FP32 CPU, `1e-5`), and the helper's carried state
  equals the native HF torch chunk recurrence's final state (`2e-4`).
* Serving per-append scoring (prefix, observation, single response tokens buffered,
  closing segment) equals offline whole-sequence scoring with super-block splitting.

## Real pinned model on H100 (scripts/vln/check_episodic.py, gate B)

| Fixture | Tokens | Helper time (layer 2) | Final-state rel. error vs dispatched FLA `chunk_gated_delta_rule` | Visual score quantiles (0/10/50/90/100%) |
|---|---:|---:|---:|---|
| R2R 96 steps | 30,589 | 0.049 s | 0.204% | 0.030 / 0.049 / 0.062 / 0.079 / 0.185 |
| RxR 96 steps | 30,683 | 0.050 s | 0.235% | 0.034 / 0.047 / 0.059 / 0.075 / 0.187 |

The residual 0.2% is FLA's BF16 inputs/arithmetic versus the FP32 helper; the dispatched
module is `fla.ops.gated_delta_rule.chunk` (forward; backward stays on the guarded
TileLang Hopper path, untouched).

## Score sensitivity: train/serve selection skew (recorded finding)

Offline (training) and online (serving) committed scores of the same visual rows differ
by up to **22–27% of the group maximum** on real 64/128-step R2R/RxR replays, while all
response-token logits stay inside the inherited BF16 budget. The score is a small
residual (`v − Sᵀk`; median ≈0.06 vs O(1) values), so ordinary BF16 differences between
a full-sequence and an incremental forward are amplified. Training-time diagnostics on
real 21-step episodes show cutoff margins at B/B+1 of only ~1e-6–5e-5 (absolute).
Consequently exact per-step 300-row sets frequently differ between offline and online
(424/512 and 924/1024 per-layer-step comparisons not identical for M2; 0 for M2c, whose
priorities do not depend on activations). Per plan §5.4 the score is **not** relaxed or
replaced; this is a property of the M2 arm to report with its results, and a
rank-bucket/quantized score is a candidate Phase-2 stabilization experiment only.

Training-time bank diagnostics (M2 smoke, real 21-step R2R): 300 rows from all 13
eligible source groups at the last group; fraction from steps 0–2 = 0.24–0.29 (layer 3),
0.32–0.34 (layer 7), 0.40–0.43 (layer 11) — moderate early-frame concentration that grows
with depth. Full-corpus statistics are logged per episode in `exposures_rank*.jsonl`
(`episodic` field) during production.
