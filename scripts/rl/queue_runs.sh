#!/usr/bin/env bash
# Run supervisors one after another (one training+evaluation at a time on the shared GPUs).
# Usage: queue_runs.sh WAIT_UNIT "NAME CONFIG FINAL" ["NAME CONFIG FINAL" ...]
set -uo pipefail
cd /data/anh/code/SimpleMem
wait_unit=$1; shift
log=/data/anh/rl_baseline/logs/queue.log
# Only an explicit inactive/failed state counts as stopped: under host memory pressure systemctl can
# fail or time out, and treating that as "stopped" once started the next run while this one ran.
stopped() { local s; s=$(systemctl --user is-active "$1" 2>/dev/null); [ "$s" = inactive ] || [ "$s" = failed ]; }
until stopped "$wait_unit"; do sleep 60; done
echo "$(date -Is) $wait_unit finished" >> "$log"
for spec in "$@"; do
  set -- $spec
  unit="rl-supervisor-$1"
  echo "$(date -Is) start $spec" >> "$log"
  systemd-run --user --unit="$unit" --collect -p WorkingDirectory=/data/anh/code/SimpleMem \
    /bin/bash scripts/rl/supervise.sh "$1" "$2" "$3" >> "$log" 2>&1
  sleep 30
  until stopped "$unit"; do sleep 60; done
  echo "$(date -Is) finished $spec" >> "$log"
done
