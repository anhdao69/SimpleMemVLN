# Candidate-logits no-action-history implementation plan

**Goal:** Smoke-test FullContext candidate logits without explicit previous-action input.

**Approved specification:** User-approved bounded ablation: `feedback_format: none`,
`append_action_tokens: false`, fixed `<|im_end|>\n` step closure, unchanged observation
history/GDN/readout/loss, distinct navigation contract. Existing history modes remain unchanged.

**Architecture:** The shared serializer returns class-independent boundary tokens.
Training and streaming already consume that same method. No model or cache rewrite is needed.

**Constraints:** Use the existing interactive allocation only; no sbatch, srun, GPU
allocation request, or modification of other running training. Worktree branch
`streaming_logits`, base `a86f2329c69f2e3df9f8e797961bbf56143e1a89`.

## Tasks and verification

1. Add failing contract/feedback/streaming tests in `tests/vln/test_no_action_history.py`.
   Implement `serializer_version(config)` in contracts and boundary-only feedback in
   `episode_serializer.py`; add `configs/vln_candidate_logits_no_history.yaml`.
   Verify the full CPU suite. Expected: old behavior unchanged, new tests pass.
2. Extend exact-tokenizer tests to prove changed action targets leave every model input
   unchanged; retain positive controls proving history-enabled inputs do change.
   Extend the native candidate integration gate to accept `--feedback none`.
3. Stage an isolated source snapshot on the server. Reconfirm existing allocation and
   idle GPUs. Run tokenizer checks and a four-rank, global-batch-eight, three-update
   FullContext no-history smoke from pinned pretrained Qwen. Run native streaming,
   reset/retry, recurrent-state, gradient and parity checks inside the same allocation.
4. Record source, commands, losses, times, memory, limitations and architecture in a
   detailed technical report in `plans/`. Review changes and rerun relevant tests.

## Review focus

- Targets must not enter serialized inputs; vary nonfinal labels, preserve final STOP.
- History checkpoints must not silently resume under the no-history contract.
- Boundary tokens belong to the current step; Window8 must evict complete groups.
- Logical positions count boundaries but never reset because of KV eviction.
- A tiny smoke cannot establish full-length memory safety or navigation quality.

## Execution evidence

- Baseline: 87 passed, 10 skipped locally.
- RED: four new tests failed on missing contract/class-independent feedback.
- Initial GREEN: 91 passed, 10 skipped locally.
- FullContext keeps history in KV but does not populate the Window8-only resident FIFO;
  tests follow this existing behavior rather than changing the streaming engine.
- Expanded local suite: 92 passed, 12 skipped; server suite: 104 passed, zero skipped.
- Native FullContext no-history integration passed; initial training completed three
  finite updates but NFS multiprocessing cleanup produced EBUSY tracebacks.
- Ruling: use unique node-local TMPDIR/kernel storage in the smoke runner, consistent
  with existing production launchers, rather than changing data workers or model behavior.
  Cost if wrong: node-local scratch can fill; this short profile-only test is bounded.
- Runner regression observed RED then GREEN; full local suite: 93 passed, 12 skipped.
- Corrected server suite: 105 passed, no skips; three-update training and native
  FullContext integration completed without traceback/OOM; max reserved 41.2266 GiB.
- Losses 2.077797, 1.117108, 2.622760; all-forward predictions on the third update
  are reported as a quality concern, not hidden behind a runtime-pass result.
- Independent review: no Critical/Important findings. Deferred minor: no-history CPU
  Window8 controller test checks bookkeeping with an empty mock cache, not real KV removal.
  Existing native cache/eviction tests remain; requested native smoke is FullContext.
