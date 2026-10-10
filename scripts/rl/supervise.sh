#!/usr/bin/env bash
# Unattended supervisor for the ActiveVLN GRPO baseline (runs as its own user unit).
#  1. Keeps training unit rl-grpo-r2r4k alive: if it stops before COMPLETE, it is relaunched
#     with --resume from the last per-rank checkpoint (at most 6 restarts per 6 hours).
#  2. After COMPLETE, evaluates the final export (step 500) and ActiveVLN's reference step
#     (350) on R2R val_unseen with the unchanged evaluator (scripts/rl/eval_export.sh).
set -uo pipefail
ROOT=/data/anh/code/SimpleMem
RUN=/data/anh/rl_baseline/runs/grpo_r2r4k
LOGS=/data/anh/rl_baseline/logs
EVALS=/data/anh/rl_baseline/evals
LOG=$LOGS/supervisor.log
cd "$ROOT"
say() { echo "$(date -Is) $*" >> "$LOG"; }
restarts=()

launch_train() {
  local extra=()
  [ -s "$RUN/rl_state.json" ] && extra=(--resume)
  say "launch training ${extra[*]}"
  scripts/rl/launch_unit.sh rl-grpo-r2r4k "$LOGS/grpo_r2r4k.log" /usr/bin/env PYTORCH_ALLOC_CONF= \
    PYTORCH_CUDA_ALLOC_CONF= .venv/bin/torchrun --nproc_per_node=2 --master_port 29661 \
    -m qwen_vl.rl.train_grpo --config configs/rl/activevln_grpo_r2r4k.yaml --output-dir "$RUN" \
    "${extra[@]}" >> "$LOG" 2>&1
  (setsid nohup scripts/rl/memwatch.sh rl-grpo-r2r4k "$LOGS/memwatch_grpo_r2r4k.log" >/dev/null 2>&1 &)
}

launch_eval() {  # step gpu
  local step=$1 gpu=$2 unit="rl-eval-step$1"
  local out="$EVALS/grpo_r2r4k-step$(printf %04d "$step")/val_unseen"
  mkdir -p "$out"
  say "launch eval step $step on GPU $gpu -> $out"
  scripts/rl/launch_unit.sh "$unit" "$LOGS/eval_step${step}.log" \
    bash scripts/rl/eval_export.sh "$RUN/exports/step_$(printf %04d "$step")" "$gpu" "$out" >> "$LOG" 2>&1
}

say "supervisor start"
while true; do
  if [ -f "$RUN/COMPLETE" ]; then
    complete=0
    for pair in "500 1" "350 0"; do
      set -- $pair
      f="$EVALS/grpo_r2r4k-step$(printf %04d "$1")/val_unseen/summary.json"
      if python3 -c "import json,sys; sys.exit(0 if json.load(open('$f')).get('complete') else 1)" 2>/dev/null; then
        complete=$((complete + 1))
      elif ! systemctl --user is-active -q "rl-eval-step$1"; then
        launch_eval "$1" "$2"
      fi
    done
    if [ $complete = 2 ]; then say "all evaluations complete"; exit 0; fi
  elif ! systemctl --user is-active -q rl-grpo-r2r4k; then
    now=$(date +%s); kept=()
    for t in "${restarts[@]}"; do [ $((now - t)) -lt 21600 ] && kept+=("$t"); done
    restarts=("${kept[@]}")
    if [ ${#restarts[@]} -ge 6 ]; then say "too many restarts; giving up"; exit 1; fi
    say "training unit inactive before COMPLETE; last log lines:"; tail -n 5 "$LOGS/grpo_r2r4k.log" >> "$LOG"
    restarts+=("$now"); launch_train
  fi
  sleep 120
done
