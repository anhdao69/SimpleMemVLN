#!/usr/bin/env bash
# Unattended supervisor for an ActiveVLN GRPO run (runs as its own user unit).
# Usage: supervise.sh RUN_NAME CONFIG FINAL_STEP
#  1. Keeps training unit rl-$RUN_NAME alive: if it stops before COMPLETE it is relaunched with
#     --resume from the last per-rank checkpoint (at most 12 restarts per 6 hours). Every (re)launch
#     first waits until each GPU has GPU_FREE_MIB free, since the GPUs are shared with other users'
#     jobs and a launch into a nearly full GPU only dies with CUDA OOM at the first update.
#  2. After COMPLETE, evaluates exports/step_FINAL on R2R val_unseen with the unchanged evaluator,
#     in 4 shards (2 per GPU; scripts/rl/eval_shards.py), merges the shards and writes the paired
#     comparison against the completed baseline journal.
set -uo pipefail
NAME=${1:?run name}; CONFIG=${2:?config}; FINAL=${3:?final step}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN=/data/anh/rl_baseline/runs/$NAME
LOGS=/data/anh/rl_baseline/logs
EVAL=/data/anh/rl_baseline/evals/$NAME-step$(printf %04d "$FINAL")
BASELINE=${BASELINE:-/data/anh/simplemem-eval/results/r2r-full/epoch-2}   # reference journal for the episode list and paired comparison
UNIT=rl-$NAME
LOG=$LOGS/supervisor-$NAME.log
cd "$ROOT"
say() { echo "$(date -Is) $*" >> "$LOG"; }
complete() { python3 -c "import json,sys; sys.exit(0 if json.load(open('$1/summary.json')).get('complete') else 1)" 2>/dev/null; }
# Only an explicit inactive/failed state counts as stopped: under host memory pressure systemctl can
# fail or time out, and treating that as "stopped" once started the next run while this one ran.
stopped() { local s; s=$(systemctl --user is-active "$1" 2>/dev/null); [ "$s" = inactive ] || [ "$s" = failed ]; }
restarts=()
GPU_FREE_MIB=${GPU_FREE_MIB:-46000}  # rank 0 holds ~40 GiB of ZeRO-2 state + grads plus a ~1 GiB helper
# Host RAM is shared too: a (re)launch into a host without room for the run only gets OOM-killed again,
# and the kernel OOM killer can then take other units with it. memwatch logs of the G=8 run: unit memory
# 11-15 GiB in steady state, trainer anonymous memory at most 14 GiB; the 31 GiB unit peak is reclaimable
# page cache of the memory-mapped checkpoint.
HOST_AVAIL_MIB=${HOST_AVAIL_MIB:-20000}

wait_gpus() {
  local logged=0 free avail
  while true; do
    # Memory our GPU keeper holds counts as free: the keeper gives it to our trainer when it starts.
    free=$(python3 "$ROOT/scripts/rl/gpu_keeper.py" free 2>/dev/null ||
           nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
    avail=$(awk '/^MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo)
    [ "${free:-0}" -ge "$GPU_FREE_MIB" ] && [ "${avail:-0}" -ge "$HOST_AVAIL_MIB" ] && break
    [ $logged = 1 ] || { say "waiting for ${GPU_FREE_MIB} MiB free on every GPU (min free ${free} MiB) and ${HOST_AVAIL_MIB} MiB host RAM available (${avail} MiB)"; logged=1; }
    sleep 120
  done
  [ $logged = 0 ] || say "GPUs and host RAM free enough (min GPU free ${free} MiB, host available ${avail} MiB)"
}

launch_train() {
  local extra=() waited=0
  # A failed transient unit stays loaded while it deactivates; wait until it is collected.
  while [ "$(systemctl --user show "$UNIT" -p LoadState --value 2>/dev/null)" = loaded ] && [ $waited -lt 600 ]; do
    systemctl --user reset-failed "$UNIT" 2>/dev/null; sleep 10; waited=$((waited + 10))
  done
  wait_gpus
  [ -s "$RUN/rl_state.json" ] && extra=(--resume)
  say "launch training ${extra[*]}"
  scripts/rl/launch_unit.sh "$UNIT" "$LOGS/$NAME.log" /usr/bin/env PYTORCH_ALLOC_CONF= \
    PYTORCH_CUDA_ALLOC_CONF= /data/anh/code/SimpleMem/.venv/bin/torchrun --nproc_per_node=2 --master_port 29661 \
    -m qwen_vl.rl.train_grpo --config "$CONFIG" --output-dir "$RUN" "${extra[@]}" >> "$LOG" 2>&1
  systemd-run --user --unit="rl-memwatch-$NAME" --collect -p MemoryMax=200M /bin/bash "$ROOT/scripts/rl/memwatch.sh" "$UNIT" "$LOGS/memwatch_$NAME.log" >/dev/null 2>&1 || true
}

say "supervisor start $NAME $CONFIG final=$FINAL"
while true; do
  if [ -f "$RUN/COMPLETE" ]; then
    [ -f "$EVAL/ids0.json" ] || { mkdir -p "$EVAL"; python3 scripts/rl/eval_shards.py split "$BASELINE" 4 "$EVAL/ids" >> "$LOG" 2>&1; }
    done_shards=0
    for pair in "0 1" "1 0" "2 1" "3 0"; do
      set -- $pair
      shard="$EVAL/shard$1"
      if complete "$shard"; then
        done_shards=$((done_shards + 1))
      elif stopped "rl-eval-$NAME-$1"; then
        python3 scripts/rl/eval_shards.py retry "$shard" >> "$LOG" 2>&1
        say "launch eval shard $1 on GPU $2"
        mkdir -p "$shard"
        scripts/rl/launch_unit.sh "rl-eval-$NAME-$1" "$LOGS/eval-$NAME-shard$1.log" \
          bash scripts/rl/eval_export.sh "$RUN/exports/step_$(printf %04d "$FINAL")" "$2" "$shard" \
          --episode-list "$EVAL/ids$1.json" >> "$LOG" 2>&1
        # Each evaluator's model load peaks at ~10 GB host RAM (CPU state dict); four at once with other
        # users' jobs OOM-killed a shard on the 62 GB host (2026-10-22). Stagger the loads.
        sleep "${EVAL_STAGGER_SECONDS:-150}"
      fi
    done
    if [ $done_shards = 4 ]; then
      python3 scripts/rl/eval_shards.py merge "$BASELINE" "$EVAL/val_unseen" "$EVAL"/shard{0,1,2,3} >> "$LOG" 2>&1 &&
      python3 scripts/rl/compare_evals.py --base "$BASELINE" --new "$EVAL/val_unseen" --name-base "${BASELINE_NAME:-sft_epoch2}" \
        --name-new "$NAME-step$FINAL" --out "$EVAL/comparison.json" >> "$LOG" 2>&1 &&
      say "evaluation merged and compared: $EVAL" && exit 0
      say "merge/compare failed"; exit 1
    fi
  elif stopped "$UNIT"; then
    now=$(date +%s); kept=()
    for t in "${restarts[@]}"; do [ $((now - t)) -lt 21600 ] && kept+=("$t"); done
    restarts=("${kept[@]}")
    if [ ${#restarts[@]} -ge 12 ]; then say "too many restarts; giving up"; exit 1; fi
    [ ${#restarts[@]} -gt 0 ] || [ -s "$RUN/rl_state.json" ] || [ -d "$RUN" ] && say "training unit inactive before COMPLETE"
    [ -f "$LOGS/$NAME.log" ] && tail -n 3 "$LOGS/$NAME.log" >> "$LOG"
    restarts+=("$now"); launch_train
  fi
  sleep 60
done
