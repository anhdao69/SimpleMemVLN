# Execution ledger — plans/SimpleMemVLN_Implementation_Plan.md

Specification: the supplied 1,229-line v2 plan, SHA256
ec556284adc7872ba7a29783b2d093eb7b64051e2c256005362497af80d3f851.
Local and remote copies match. Base checkout: 522658d (includes audited base).

User scope: implement on H100 server, set up environment, R2R smoke test,
report measured loss and training time; global batch size 64.

## Decisions

- Use dedicated `codex/streaming-r2r` branch in the requested checkout.
- User batch override: 64 complete episode exposures per full update =
  four ranks × one episode/rank × 16 accumulation microsteps. Action count
  remains the loss denominator. Partial updates use actual exposure counts.
- Follow the supplied architecture/spec directly; no new design approval
  needed for work already explicitly requested.
- Preserve all existing unrelated files and environments. Build a project
  environment using the locally available transferred wheel after hash check.

## Work and gates

- [x] SSH and active Slurm access: job 4443, worker-1, four H100 80GB.
- [x] Read full supplied plan; remote/local hashes match.
- [x] Exact environment, source/binary fingerprint and runtime gate R.
- [x] Episode manifest, structural/token audit and A/B serializers.
- [ ] Capture-before-action replay verification (remaining part of data gate D).
- [x] Native adapter, positions, FP32 state, FullContext continuation.
- [x] Trainer action-mean normalization, batch 64, save/load/resume.
- [x] Window8 attention, streaming A/B lifecycle and declared-budget parity.
- [x] Real R2R smoke and representative resource measurements; evidence report.
- [x] Final independent review and required fixes.
- [ ] Habitat rollout, small-set learning, full epoch/validation research gates.

## Interface preflight

Serializer → model: complete chronological episodes; A cue-end positions,
B target positions J with projection at J-1; common prefix/step spans.
Model → Trainer: local sum of per-action losses and valid action count;
Trainer alone scales by world size/global update action count.
Serializer → session: identical fragments, independent logical/MRoPE cursors.
Session → attention: eviction before append, full-attention layers only;
training callback consumes immutable spans, never mutable serving state.
Checkpoint → reload: mode/codec/serializer/preprocessing contract validated.

No measurements or GPU correctness gates are claimed until recorded.

## Evidence and numerical decisions

- Environment installed with pinned Torch2.10/cu129, Transformers5.11, FA2.8.3,
  FLA/fla-core0.5.2, Accelerate1.13, DeepSpeed0.16.4. Mirror FA wheel matches
  fa278650...e5218c; similarly named general cache wheel was rejected (different hash).
- Structural corpus audit: 10,819 episodes, 631,244 actions; all valid final STOP.
  Capture-before-action replay is NOT verified; simulator environment unavailable.
- Real FA2 output/backward and FP32 cache tests pass. Two-rank ZeRO2/GAS2 actual
  Trainer updates match independent single-process AdamW exactly, including 2-exposure tail.
- Initial synthetic hidden allclose failed .15/.03. Uncached short-prefix and
  cached prefix were identical; changing uncached sequence length itself caused
  max .75 hidden difference. Logit check passed .1/.03; isolated GDN max6.1e-5.
- Real-image BF16 A logit parity failed initial .15/.03: max .344, RMS .076-.114
  over 21/25/31 steps. Errors did not grow monotonically; all flips had small margins.
  Real isolated GDN on image embeddings: RMS9.22e-5 versus output RMS .0556,
  max .015625 at every step (BF16-scale rounding), no growing recurrent error.
- Ruling: retain initial failures and expose explicit parity tolerances; evaluate
  A with atol .5/rtol .03 plus RMS<=.15 and zero high-margin flips. This is an
  empirical BF16 budget for this stack, not exact action equivalence. Risk:
  near-tie actions can differ, and longer unseen trajectories require further gates.
- Ruling: register `simplememvln_step_attention` (same direct FA2 algorithm),
  because HF5.11 tries to import custom names containing `flash` as external kernels.
- Runtime blocker found and repaired: FLA0.5.2 explicitly refuses Triton3.6
  gated backward on Hopper (known incorrect results). Added pinned TileLang0.1.14
  and its lock dependencies; did not disable the guard or replace Torch's Triton.
  Actual dispatched TileLang gradients passed an FP32 recurrent oracle on H100.
  Compilation uses the existing CUDA12.8.1 toolkit; Torch runtime remains12.9.
- FP32 CPU tiny native model: mixed multi/single-token continuation and future
  causality pass at2e-5; final-only loss has nonzero gradient at early inputs.
- Independent review found DeepSpeed resume-hook bypass and missing ordered
  dataset identity. Both fixed with backend-independent pre-train contract
  validation and per-checkpoint manifest/ordered-ID fingerprint. Unit tests pass.
- Independent review found vacuous empty parity success. Explicit nonempty,
  unique and complete requested-ID validation now rejects empty/missing IDs.
- Evaluator and isolated model subprocess implemented; simulator runtime missing,
  so Habitat execution is explicitly unverified. Failure-denominator test passes.
- Reviewer exclusions accepted: CPU serving and concurrent sessions are outside
  H100 single-stream v0; full research campaigns/window sweeps are beyond smoke scope.
- Historical pre-rotary-fix FullContext+A diagnostic: 4 ranks, GAS16, 64 shortest complete episodes/update,
  1,550 actions/update. Four updates: action-mean losses
  3.9393727, 3.9393727, 1.5271764, 1.5970412. First update LR0 warmup.
  Peak allocated32.80GiB/reserved36.83GiB; steady compute16.19–17.07s/update.
  Trainer runtime161.80s, load/train/save wall190.16s. STOP precision7.44%,
  recall50% on update4; this is a pipeline smoke, not a competent navigation model.
- Text Window8 forced-history parity on21/25/31 steps: max vocabulary-logit
  error .625, RMS .048–.060, argmax agreement100%; initial .5/.03 gate failed.
  Ruling: text-head empirical budget .75/.03 with RMS<=.1, separately reported
  from A. Full vocabulary extremes and BF16 rounding differ from the 4-class
  projection; risk remains near-tie token flips on untested trajectories.
- Checkpoint regression found by actual reload: DeepSpeed's BF16 module cast
  rounded non-persistent text/vision rotary frequency buffers. Parameter dtypes
  and saved weights matched, but fresh FP32 rotary buffers changed logits (max
  .8125). Reapplying BF16 to the loaded model reproduced the old reference
  exactly, isolating the cause. Added a failing-then-passing cast regression and
  wrapper-level FP32 preservation; updated checkpoint metadata to reject the
  old runtime contract. Earlier smoke outputs are diagnostic, not final exports.
- Pretrained temporal gradient: final-only STOP loss reaches first-image source
  embeddings; frozen vision has no gradients. Initial elementwise .002/.03
  gradient gate failed. Repeated uncheckpointed runs themselves differ by
  1.3–1.6% relative L2; checkpoint recomputation differences are comparable.
  Explicit revised budget: relative L2<=3%, cosine>=.999, identical forward
  loss, with two runs of each mode. FullContext and Window8 pass this budget.
  This establishes temporal credit, not GDN-specific memory attribution.
- Actual two-rank ZeRO2 optimizer checkpoint resume passes independent CPU
  AdamW continuation exactly, including sampler skip and final partial update:
  resumed exposures [4,2], maximum weight difference zero.
- Full H100 unit suite with the pinned processor: 19 passed, no skipped tests.

## Final measured state

Canonical results and remaining gates are in
`reports/H100_R2R_SMOKE_2026-09-28.md`; compact remote evidence is copied to
`artifacts/remote_evidence/` locally.

- Final suite: 22 passed, no skipped tests. Lock check, compilation and format
  checks pass. Real model-process IPC and all three corrected export reloads pass.
- Corrected A losses (four updates, global batch64): FullContext
  3.931709 → 1.708197, wall161.02s; Window8 4.597833 → 3.418346, wall166.65s.
  Window8's last update is all-forward; no navigation learning gate is claimed.
- Window8 B: action-mean token loss .330166 → .311350, wall185.01s.
  Trained generated-history replay passes at the unchanged declared budget.
- Two-update optimizer-allocated short/median/p95/longest profiles all fit:
  FullContext A55.70GiB, Window8 A56.53GiB, Window8 B58.61GiB peak allocated.
  The four representatives are explicitly repeated to fill batch64; no claim
  of 64 unique profile episodes. No profile checkpoints are published.
- Both A modes pass longest-episode183-step parity with unchanged tolerances,
  no high-margin flips. Both complete500-step dry streaming and reset/retry
  checks. Window8 peaks at2,501 KV tokens /8.67GiB allocated; FullContext
  grows to152,561 /13.50GiB, but reserves78.47GiB at peak.
- Stronger production-bucket sampler resume test matches exact uninterrupted
  exposure order and next weights, maximum difference zero. The actual
  Accelerate full-size tail test confirms one repeat and final four exposures.
- Simulator integration, raw capture timing, overfit/pilot and full-data
  benchmark results remain unverified/unrun, explicitly not marked complete.
