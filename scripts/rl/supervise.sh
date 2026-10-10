#!/usr/bin/env bash
# Unattended supervisor for an ActiveVLN GRPO run (runs as its own user unit).
# Usage: supervise.sh RUN_NAME CONFIG FINAL_STEP
#  1. Keeps training unit rl-$RUN_NAME alive: if it stops before COMPLETE it is relaunched with
#     --resume from the last per-rank checkpoint (at most 6 restarts per 6 hours).
#  2. After COMPLETE, evaluates exports/step_FINAL on R2R val_unseen with the unchanged evaluator,
#     in 4 shards (2 per GPU; scripts/rl/eval_shards.py), merges the shards and writes the paired
#     comparison against the completed baseline journal.
set -uo pipefail
NAME=${1:?run name}; CONFIG=${2:?config}; FINAL=${3:?final step}
ROOT=/data/anh/code/SimpleMem
RUN=/data/anh/rl_baseline/runs/$NAME
LOGS=/data/anh/rl_baseline/logs
EVAL=/data/anh/rl_baseline/evals/$NAME-step$(printf %04d "$FINAL")
BASELINE=/data/anh/simplemem-eval/results/r2r-full/epoch-2
UNIT=rl-$NAME
LOG=$LOGS/supervisor-$NAME.log
cd "$ROOT"
say() { echo "$(date -Is) $*" >> "$LOG"; }
complete() { python3 -c "import json,sys; sys.exit(0 if json.load(open('$1/summary.json')).get('complete') else 1)" 2>/dev/null; }
restarts=()

launch_train() {
  local extra=() waited=0
  # A failed transient unit stays loaded while it deactivates; wait until it is collected.
  while [ "$(systemctl --user show "$UNIT" -p LoadState --value 2>/dev/null)" = loaded ] && [ $waited -lt 600 ]; do
    systemctl --user reset-failed "$UNIT" 2>/dev/null; sleep 10; waited=$((waited + 10))
  done
  [ -s "$RUN/rl_state.json" ] && extra=(--resume)
  say "launch training ${extra[*]}"
  scripts/rl/launch_unit.sh "$UNIT" "$LOGS/$NAME.log" /usr/bin/env PYTORCH_ALLOC_CONF= \
    PYTORCH_CUDA_ALLOC_CONF= .venv/bin/torchrun --nproc_per_node=2 --master_port 29661 \
    -m qwen_vl.rl.train_grpo --config "$CONFIG" --output-dir "$RUN" "${extra[@]}" >> "$LOG" 2>&1
  systemd-run --user --unit="rl-memwatch-$NAME" --collect -p MemoryMax=200M /bin/bash /data/anh/code/SimpleMem/scripts/rl/memwatch.sh "$UNIT" "$LOGS/memwatch_$NAME.log" >/dev/null 2>&1 || true
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
      elif ! systemctl --user is-active -q "rl-eval-$NAME-$1"; then
        say "launch eval shard $1 on GPU $2"
        mkdir -p "$shard"
        scripts/rl/launch_unit.sh "rl-eval-$NAME-$1" "$LOGS/eval-$NAME-shard$1.log" \
          bash scripts/rl/eval_export.sh "$RUN/exports/step_$(printf %04d "$FINAL")" "$2" "$shard" \
          --episode-list "$EVAL/ids$1.json" >> "$LOG" 2>&1
      fi
    done
    if [ $done_shards = 4 ]; then
      python3 scripts/rl/eval_shards.py merge "$BASELINE" "$EVAL/val_unseen" "$EVAL"/shard{0,1,2,3} >> "$LOG" 2>&1 &&
      python3 scripts/rl/compare_evals.py --base "$BASELINE" --new "$EVAL/val_unseen" --name-base sft_epoch2 \
        --name-new "$NAME-step$FINAL" --out "$EVAL/comparison.json" >> "$LOG" 2>&1 &&
      say "evaluation merged and compared: $EVAL" && exit 0
      say "merge/compare failed"; exit 1
    fi
  elif ! systemctl --user is-active -q "$UNIT"; then
    now=$(date +%s); kept=()
    for t in "${restarts[@]}"; do [ $((now - t)) -lt 21600 ] && kept+=("$t"); done
    restarts=("${kept[@]}")
    if [ ${#restarts[@]} -ge 6 ]; then say "too many restarts; giving up"; exit 1; fi
    [ ${#restarts[@]} -gt 0 ] || [ -s "$RUN/rl_state.json" ] || [ -d "$RUN" ] && say "training unit inactive before COMPLETE"
    [ -f "$LOGS/$NAME.log" ] && tail -n 3 "$LOGS/$NAME.log" >> "$LOG"
    restarts+=("$now"); launch_train
  fi
  sleep 60
done
