#!/usr/bin/env bash
# Run supervisors one after another (one training+evaluation at a time on the shared GPUs).
# Usage: queue_runs.sh WAIT_UNIT "NAME CONFIG FINAL" ["NAME CONFIG FINAL" ...]
set -uo pipefail
cd /data/anh/code/SimpleMem
wait_unit=$1; shift
log=/data/anh/rl_baseline/logs/queue.log
while systemctl --user is-active -q "$wait_unit"; do sleep 60; done
echo "$(date -Is) $wait_unit finished" >> "$log"
for spec in "$@"; do
  set -- $spec
  unit="rl-supervisor-$1"
  echo "$(date -Is) start $spec" >> "$log"
  systemd-run --user --unit="$unit" --collect -p WorkingDirectory=/data/anh/code/SimpleMem \
    /bin/bash scripts/rl/supervise.sh "$1" "$2" "$3" >> "$log" 2>&1
  sleep 30
  while systemctl --user is-active -q "$unit"; do sleep 60; done
  echo "$(date -Is) finished $spec" >> "$log"
done
