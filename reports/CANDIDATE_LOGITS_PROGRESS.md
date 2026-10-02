# Candidate-logits implementation ledger

Plan/spec: reports/CANDIDATE_LOGITS_DESIGN.md. Base 341ec2c.
User explicitly approved implementation with three amendments; no further
design-approval pause. Isolated sibling worktree avoids altering old checkout.
Shared interfaces: serializer candidate_ids/feedback_ids -> readout/session;
model action_logits -> training/session; weighted losses -> global denominator;
saved resolved counts -> reload. Preserve old-mode metadata byte-for-byte.

- Setup: branch streaming_logits created; remote scratch separate from training.
- Task 1: contract/tokenizer complete, e4a53d6. Exact pinned tokenizer A/B/C/D
  IDs 32/33/34/35; `Action:\n` preserves the append boundary.
- Task 2: pretrained readout/loss complete, 600314c. Tied rows use backbone LR;
  distribution-weight normalization tested against the global action denominator.
- Task 3: streaming feedback complete, c07db67. Both feedback formats, retries,
  complete Window8 groups and persistent GDN covered by tests.
- Task 4: training diagnostics complete, 3c880ec. Counts resolved from the exact
  selected manifest and confusion/loss sums reduced across ranks.
- Tasks 5–6: integration, profiling, production overlays and report in progress.
- Verification: remote GPU suite 66 passed, zero skipped before latest profiling
  additions; latest local suite 62 passed, 10 tokenizer/GPU skips.
- Ruling: policy overlays merge after production recipe and before memory config
  without overriding model_max_length — preserves joint-dataset admission limits;
  cost if wrong: changed admitted episode population. Covered by config test.
- Ruling: component timings are opt-in, synchronized diagnostics, not headline
  latency; nested history/LM timings must not be added together.
