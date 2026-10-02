# Candidate-logit navigation: approved design

Base: `341ec2cfce934a872c4478cb06c4f70354113bc4` on `streaming`.
Approved 2026-10-02, including the three experimental-cleanliness amendments.

Add `candidate_logits` without changing existing text/classification contracts.
Select A/B/C/D LM-head rows directly; default `lm_rows_trainable` uses backbone
LR 5e-6. `lm_rows_frozen` freezes the shared input/output parameter;
`copied_linear` alone uses classifier_lr and copies pretrained weights/bias.
Pinned tokenizer IDs are 32/33/34/35. Require single-token distinct nonspecial
labels and exact concatenation equivalence after the rendered `Action:\n` cue.
New serializer: `vln_candidate_logits_v1`.

The decision read precedes feedback. Training uses gold history; inference uses
its own argmax. Feedback formats: `candidate_token` and `canonical_action_text`.
Both append one deterministic block with fixed assistant terminator/newline;
neither generates tokens. Entire observation/feedback belongs to one FIFO group.
Positions advance absolutely; Window8 only evicts full-attention KV, never GDN.
Preserve reset, retry idempotency and fail-closed partial-session semantics.

Four-way CE uses none, sqrt_inverse_frequency, or effective_number weights.
For counts n and p=n/sum(n), normalize with sum(p*w)=1, NOT mean(w)=1.
Retain global valid-action denominator over ranks/GAS, log weighted/unweighted
loss separately, and derive weights only from the selected training manifest.
Missing-class training counts fail closed when balancing is requested.

Per-class precision/recall/support, confusion, predicted/truth distributions,
overall accuracy and macro recall make forward collapse visible. Closed-loop
Habitat has no automatic ground-truth action oracle: report supervised STOP
recall separately from model/forced STOP and navigation success.

Benchmark one model at a time, same idle hardware/runtime/frame/instruction,
16 warmups + 64 measured decisions and optional 500 decisions. Instrumentation
is opt-in and separate from headline timing. Report preprocessing, vision,
language append, projection, feedback append and text decoding; no claimed 3x
speedup. Full training/navigation results require actual runs, not smoke CE.

Compatibility: candidate labels, cue, feedback format, head mode, class counts
and weighting policy are saved and checked. No reinterpretation of old weights.
Preserve existing shared parameter storage on torch checkpoint save/reload.

## Implementation sequence

1. Contract/serializer/dataset length audit with real-tokenizer boundary tests.
2. Selected LM readout, frozen/copied modes, CE and gradient/optimizer tests.
3. Streaming feedback, diagnostics and history/eviction/reset/retry tests.
4. Manifest-derived balancing, metrics and distributed normalization tests.
5. Reload/parity/old-text and full-model smoke integration checks.
6. Benchmark/evaluation/config controls, reproducible commands and final report.

Implementation is inline in an isolated worktree; final independent review.
No existing allocation or training source snapshot may be modified.
