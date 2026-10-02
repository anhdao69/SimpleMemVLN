# Candidate-logits implementation ledger

Plan/spec: reports/CANDIDATE_LOGITS_DESIGN.md. Base 341ec2c.
User explicitly approved implementation with three amendments; no further
design-approval pause. Isolated sibling worktree avoids altering old checkout.
Shared interfaces: serializer candidate_ids/feedback_ids -> readout/session;
model action_logits -> training/session; weighted losses -> global denominator;
saved resolved counts -> reload. Preserve old-mode metadata byte-for-byte.

- Setup: branch streaming_logits created; remote scratch separate from training.
- Tasks 1–6: pending.
