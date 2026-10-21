# Preparation and replay reproduction

This bundle contains nine command recipes: deterministic panel regeneration,
the original failed training-rendering gate, completed expert rendering, teacher
caching, and the five successful G-2 replay gates. Rendering remains failed; the
user-authorized continuation is recorded separately in `SCOPE_UPDATE.md`.

Panel regeneration ran into a new directory and reproduced both frozen panel
files byte-for-byte. Its command is an actual verification invocation, not a
claim to recover the original selection shell command. Expert and teacher argv
come from completed process records; explicit environment overrides come from
the preserved launcher source. Inherited shell environment was not recorded.

G-2 top-level recipes are reconstructed from saved worker invocations and the
verified gate results. Every original worker argv and PYTHONPATH is retained in
its invocation snapshot. The current orchestrator adds Python safe-path `-P`
to prevent `scripts/diag/statistics.py` shadowing the standard library. The
successful G1 evidence is from `G1_retry_safe_path`, not the failed first attempt.
No replay was rerun while packaging these records.

Historical destinations are preserved for audit. For a new reproduction, map
all output destinations and their downstream references to new empty locations;
do not overwrite frozen outputs. The rendering command uses defaults: explicitly
supply new `--output` and `--report` destinations before rerunning it. Reproduce
the failed gate as a failed result; do not tune it to pass.

D-e and final report commands still need to be added after completion.
