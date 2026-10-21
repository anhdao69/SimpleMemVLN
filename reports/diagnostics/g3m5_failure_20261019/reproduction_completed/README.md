# Completed-stage reproduction commands

`commands.json` preserves the original executable arguments for five two-episode
instrumentation gates, five full-panel captures, and six completed analysis jobs.
Every entry links to its original queue record and hash-verified completion evidence.
The G3/G1 capture arguments also match the recorded parallel launches.

The executable reproduction form prepends `env` with explicit settings recovered
from the launcher. It does not claim the unrecorded inherited shell environment
was captured. Working directories and interpreter paths are explicit; the full
software versions remain in the report environment provenance.

These are historical commands, with historical output paths. Use new empty output
roots and consistently remap their downstream references when reproducing. Keep
the original absolute scene paths for D-e episode UIDs. Do not run these commands
against the frozen measurements. Existing output guards are not a reason to delete
original evidence.

This is a partial command inventory, not a final reproduction/completion claim.
Preparation, G-2, D-e and final report commands still need their own records.
The analysis command snapshot contains running D-e entries for provenance only;
those entries are excluded from the completed executable command list.
