#!/usr/bin/env bash
# Run an RL command in its own transient systemd user unit with a host-RAM cap, so an
# out-of-memory event can only stop this job (never the tmux pane, the Claude session or
# the running baseline evaluation). Usage: launch_unit.sh UNIT_NAME LOG_FILE CMD...
set -euo pipefail
unit=${1:?unit}; logfile=${2:?log}; shift 2
systemd-run --user --unit="$unit" --collect -p MemoryHigh=36G -p MemoryMax=50G -p MemorySwapMax=0 \
  -p WorkingDirectory=/data/anh/code/SimpleMem \
  /bin/bash -c 'source /data/anh/simplemem-eval/env.sh; exec "$@" >>"'"$logfile"'" 2>&1' _ "$@"
