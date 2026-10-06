#!/usr/bin/env bash
# Keep the requested ordering: finish R2R, validate RxR smoke, then run all RxR.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
R2R_OUTPUT_DIR="${R2R_OUTPUT_DIR:-$ROOT/artifacts/evaluation/r2r-rxr15-r2r/epoch-1}"
R2R_PID_FILE="${R2R_PID_FILE:-$ROOT/artifacts/evaluation/r2r-rxr15-r2r/epoch-1.pid}"
RXR_RUNNER="${RXR_RUNNER:-$ROOT/scripts/eval/run_rxr15_epoch1.sh}"
RXR_SMOKE_OUTPUT_DIR="${RXR_SMOKE_OUTPUT_DIR:-$ROOT/artifacts/evaluation/r2r-rxr15-rxr-smoke/epoch-1}"
RXR_OUTPUT_DIR="${RXR_OUTPUT_DIR:-$ROOT/artifacts/evaluation/r2r-rxr15-rxr/epoch-1}"
EXPECTED_R2R_EPISODES="${EXPECTED_R2R_EPISODES:-1839}"
EXPECTED_RXR_EPISODES="${EXPECTED_RXR_EPISODES:-3669}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"

check_summary() {
  "$ROOT/.venv/bin/python" - "$1" "$2" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1]);expected=int(sys.argv[2])
if not path.exists():sys.exit(1)
d=json.loads(path.read_text())
if d.get('scheduled_episodes')!=expected or d.get('episodes',0)>expected:
    sys.exit(2)
if not d.get('complete') or d['episodes']!=expected:sys.exit(1)
if d.get('failures',0):
    journal=path.parent/'episodes.jsonl'
    if not journal.exists():sys.exit(2)
    failures=0
    for line in journal.open():
        row=json.loads(line)
        failures+=bool(row.get('failed'))
        if row.get('fatal'):sys.exit(2)
    if failures!=d['failures']:sys.exit(2)
sys.exit(0)
PY
}

while true; do
  status=0
  check_summary "$R2R_OUTPUT_DIR/summary.json" "$EXPECTED_R2R_EPISODES" || status=$?
  if [[ "$status" == 0 ]]; then break; fi
  if [[ "$status" != 1 ]]; then echo 'R2R summary failed validation' >&2; exit 2; fi
  if [[ ! -s "$R2R_PID_FILE" ]] || ! kill -0 "$(cat "$R2R_PID_FILE")" 2>/dev/null; then
    echo 'R2R stopped before finishing; RxR was not started' >&2
    exit 2
  fi
  sleep "$WAIT_SECONDS"
done
echo 'R2R complete; running RxR smoke' >&2
OUTPUT_DIR="$RXR_SMOKE_OUTPUT_DIR" "$RXR_RUNNER" --max-episodes 2
check_summary "$RXR_SMOKE_OUTPUT_DIR/summary.json" 2 || {
  echo 'RxR smoke did not complete cleanly' >&2
  exit 2
}
echo 'RxR smoke complete; running the full 3669-episode split' >&2
OUTPUT_DIR="$RXR_OUTPUT_DIR" "$RXR_RUNNER"
check_summary "$RXR_OUTPUT_DIR/summary.json" "$EXPECTED_RXR_EPISODES" || {
  echo 'RxR full split did not complete cleanly' >&2
  exit 2
}
echo 'RxR full split complete' >&2
