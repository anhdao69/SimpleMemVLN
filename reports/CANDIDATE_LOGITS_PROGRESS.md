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
- Task 5: real H100 integration passed all four memory/feedback combinations;
  exact checkpoint reload passed after a 2-GPU, GAS4, global8 smoke (3 updates).
  Local suite 63 passed/10 skipped; remote full GPU suite 73 passed/0 skipped.
- Task 6: implemented in 066938a; matched same-H100 benchmark completed,
  82.65 ms candidate versus 122.85 ms text median on the verified repeat.
- Final review: fresh read-only reviewer found no Critical/Important issues;
  independently verified 63 passed/10 skips locally.
- Final: minor (deferred): integration high-margin criterion compares against
  observed error and is mathematically redundant. Independent absolute/RMS
  bounds still enforce numerical parity; exact argmax agreement is reported,
  not asserted. Strengthen the fixed-fixture decision gate in a later change.
- Final: Ruling: full-training convergence and Habitat quality remain unclaimed
  — smoke is insufficient and the server Janus interpreter lacks Habitat — cost
  if wrong: experimental quality could be misread as validated; report marks all
  candidate navigation scores pending.
- Final: Ruling: remote smoke/reload/benchmark evidence is verified by the main
  implementer, outside the reviewer's local scope — cost if wrong: runtime
  regressions missed by CPU tests; retain raw remote results and run final suite.
- Final: fixed benchmark allocator provenance (legacy CUDA env alias was omitted)
  — test_benchmark_records_legacy_allocator_environment RED→GREEN;
  full suite 64 passed/10 skipped locally and 74 passed/0 skipped on H100.
- Ruling: report the new paired H100 timings separately from historical Blackwell
  results — original RGB artifact is unavailable on this server; both new models
  use the same documented replacement frame — cost if wrong: an invalid
  cross-environment speed claim. No such claim is made.
