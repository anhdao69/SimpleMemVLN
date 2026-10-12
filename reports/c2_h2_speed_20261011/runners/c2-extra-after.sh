#!/usr/bin/env bash
set -euo pipefail
R=/mnt/data/vmo-ai-task/anhdh35
while ! grep -q WARM_BASELINE_DONE "$R/c2-warm-after.log"; do
 if grep -q -E 'Traceback|Exited with exit code' "$R/c2-warm-after.log"; then exit 1; fi
 sleep 10
done
python "$R/c2-extra-arms.py"
