#!/usr/bin/env bash
# Wait for WAIT_UNIT, run a 2-step real-model smoke of SMOKE_CONFIG, and queue runs only if it passes.
# Usage: smoke_then_queue.sh WAIT_UNIT SMOKE_CONFIG "NAME CONFIG FINAL" ["NAME CONFIG FINAL" ...]
# The first spec is the run under test: on a failed smoke it is skipped and the rest still run.
set -uo pipefail
cd /data/anh/code/SimpleMem
wait_unit=$1; smoke_config=$2; shift 2
log=/data/anh/rl_baseline/logs/queue.log
while systemctl --user is-active -q "$wait_unit"; do sleep 60; done
name=smoke_$(basename "$smoke_config" .yaml)_$(date +%Y%m%d%H%M)
out=/data/anh/rl_baseline/runs/$name
echo "$(date -Is) $wait_unit finished; smoke $smoke_config -> $out" >> "$log"
scripts/rl/launch_unit.sh "rl-$name" "/data/anh/rl_baseline/logs/$name.log" /usr/bin/env \
  PYTORCH_ALLOC_CONF= PYTORCH_CUDA_ALLOC_CONF= .venv/bin/torchrun --nproc_per_node=2 --master_port 29671 \
  -m qwen_vl.rl.train_grpo --config "$smoke_config" --output-dir "$out" --max-steps 2 \
  --prompts-per-step 4 --group-size 2 --budget-limit 60 --export-steps "" --save-every 1000 >> "$log" 2>&1
sleep 30
while systemctl --user is-active -q "rl-$name"; do sleep 30; done
if python3 scripts/rl/check_fork_smoke.py "$out" >> "$log" 2>&1; then
  echo "$(date -Is) smoke passed" >> "$log"
  exec scripts/rl/queue_runs.sh rl-none-placeholder "$@"
fi
echo "$(date -Is) smoke FAILED; skipping: $1" >> "$log"
shift
[ $# -gt 0 ] && exec scripts/rl/queue_runs.sh rl-none-placeholder "$@"
