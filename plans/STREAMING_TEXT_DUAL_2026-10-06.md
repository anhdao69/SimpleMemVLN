# Window8 text dual-lane implementation and qualification

## Confirmed experiment

The user selected fresh original Qwen3.5-4B weights, two epochs on joint R2R
and English-guide RxR_15deg, and the existing four-H100 interactive allocation
4659 on worker-3 (768 GiB host RAM). This is a fresh run, with canonical text
action history; it is not adaptation of either FullContext candidate checkpoint.
Start from `streaming_logits_dual` origin commit `6b0f30a` and implement on
`streaming_text_dual`. Preserve existing sources, running jobs and checkpoints.

## Scientific and causal contract

Preserve the architecture in `dual_lane.md`: four independent step lanes at
layers 16/20/24/28, 21,054,000 extra parameters, zero initial output projections,
1 MiB FP32 runtime state per episode. Native GDN state persists. The instruction
prefix cannot write. Ordinary generated action tokens cannot write. The last
existing token of EOS-plus-separator is the one writer per completed group.
Writes can influence subsequent groups, never the preceding action decision.
Window8 retains the prefix and eight observation/feedback groups in native KV;
eviction does not reset lane state, native GDN, or logical positions.

Use `qwen_text`, serializer `vln_append_only_chat_v1`, EOS supervision, per-action
mean response-token CE, then global action-count normalization across ranks/GAS.
Keep frozen vision, full-episode BPTT, B1 unpadded episodes, gradient checkpointing,
no truncated trajectories, no new labels/auxiliary objectives. Schedule: global
eight episodes = four ranks × GAS2, 7,704 updates, 232 warmup, native LR 5e-6,
lane LR 1e-4, seed 429. Preserve epoch-1 and epoch-2 exports.

## Performance experiment

The existing Window8 attention runs a separate FlashAttention call for every
group and reconstructs one window at a time in backward. Add opt-in
`runtime.window_attention_batch_steps` (default 1, allowed 1–64). Pack bounded
batches of independent windows for FlashAttention varlen; bottom-right causality
is evaluated independently for each packed window. Save only original Q/K/V,
recompute one bounded batch in backward, and scatter-add all gradients to the
shared prefix/history. No all-episode replicated KV storage and no detach of
temporal gradients. Streaming inference keeps its existing cached append path.
Serialized runtime setting follows saved configuration for reproducibility.

Compare batches 1/4/16/32 at 64/183/627 groups, then use matched full-model
optimizer profiles to decide. A faster attention microbenchmark alone does not
establish faster training. Keep RAM activation offload; qualify the actual text
loss and optimizer slots on the longest R2R and RxR episodes. Prefer measured
GPU optimizer execution if it fits; CPU optimizer offload is a fallback.

## Acceptance sequence

1. Audit plans, implementation report, latest consolidated quality/latency report,
   serializers, session lifecycle, loss reduction, recovery, and GPU availability.
2. Compare packed attention values/gradients to an independent FP64 dense oracle
   and installed BF16 FA2 serial path, with variable lengths and eviction.
3. Run the complete pinned test suite. Test actual Qwen text zero-init output,
   lane gradients after opening projection, fixed-history streaming across
   steps 8/9/16, idempotent retry and reset. Keep existing numerical budgets.
4. Measure matched full-model profiles and all-rank longest-episode updates after
   optimizer state allocation. Require finite loss, no OOM, all ranks complete.
5. Train/save a short mixed R2R/RxR fixture and strictly reload its text/lane
   weights. Exercise distributed text loss normalization and recovery.
6. Commit qualified source/config/tests/report; use an immutable source snapshot
   and fresh output directory. Launch with a 700 GiB host working-set guard,
   periodic complete recovery checkpoints and epoch exports. Verify actual
   Trainer schedule, all-rank progress and GPU use after launch.

No claim of 100% universal correctness or improved navigation SR follows from
these gates. The October 6 consolidated report shows mixed candidate dual-lane
quality results and a stronger existing Window8 text baseline; this experiment
needs its own subsequent navigation evaluation.
