#!/usr/bin/env bash
# Run an RL command in its own transient systemd user unit with a host-RAM cap, so an
# out-of-memory event can only stop this job (never the tmux pane, the Claude session or
# the running baseline evaluation). Usage: launch_unit.sh UNIT_NAME LOG_FILE CMD...
# User-manager services get kernel OOM score adjustment +200 (manager +100, services +100), while
# other users' jobs run at 0, so a host-RAM OOM picked our jobs first. +100 is the lowest a user unit
# may request (the manager's floor); RL_OOM_SCORE_ADJUST overrides it once root lowers that floor.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)   # this checkout (also a git worktree such as SimpleMem-m5)
unit=${1:?unit}; logfile=${2:?log}; shift 2
systemd-run --user --unit="$unit" --collect -p MemoryMax=32G -p MemorySwapMax=0 \
  -p OOMScoreAdjust="${RL_OOM_SCORE_ADJUST:-100}" \
  -p WorkingDirectory="$ROOT" \
  /bin/bash -c 'source /data/anh/simplemem-eval/env.sh; export SMV_ROOT="'"$ROOT"'";
    export SIMULATOR_SOURCE="$SMV_ROOT/docs/evaluation_setup" PYTHONPATH="$SMV_ROOT/src:$SMV_ROOT/docs/evaluation_setup";
    exec "$@" >>"'"$logfile"'" 2>&1' _ "$@"
