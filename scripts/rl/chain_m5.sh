#!/usr/bin/env bash
# M5 RL queue (user request 2026-10-22), run by the persistent user unit rl-chain-m5.service:
#   wait for the Window8 chain (G = 8 control + its evaluation) -> 2-step smokes of GRPO and GRPO+F-B on
#   M5_E1 -> local evaluation of M5_E1 itself (reference journal for paired tests) -> GRPO on M5_E1 ->
#   GRPO+F-B on M5_E1, each evaluated on val_unseen once at step 125.
# Steps are idempotent (supervise.sh resumes training, skips finished shards, exits 0 when evaluated),
# so a restart after a host OOM continues where it stopped. A failed smoke writes SMOKE_FAILED and ends
# the chain with exit 0 (no systemd restart loop); delete that marker to retry after a fix.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); cd "$ROOT"
R=/data/anh/rl_baseline; log=$R/logs/queue_m5.log
say() { echo "$(date -Is) $*" >> "$log"; }
say "chain_m5 start ($ROOT @ $(git rev-parse --short HEAD))"
while systemctl --user is-active -q rl-chain; do sleep 120; done

smoke() {   # NAME CONFIG [--no-forks]
  local name=$1 config=$2 flag=${3:-} out=$R/runs/$1
  if [ -f "$out/SMOKE_PASSED" ]; then return 0; fi
  if [ -f "$out/SMOKE_FAILED" ]; then say "smoke $name failed earlier; chain stopped"; exit 0; fi
  if [ ! -f "$out/STOPPED" ]; then
    rm -rf "$out"
    say "smoke $name"
    scripts/rl/launch_unit.sh "rl-$name" "$R/logs/$name.log" /usr/bin/env PYTORCH_ALLOC_CONF= \
      PYTORCH_CUDA_ALLOC_CONF= /data/anh/code/SimpleMem/.venv/bin/torchrun --nproc_per_node=2 \
      --master_port 29681 -m qwen_vl.rl.train_grpo --config "$config" --output-dir "$out" --max-steps 2 \
      --prompts-per-step 4 --group-size 2 --budget-limit 60 --export-steps "" --save-every 1000 >> "$log" 2>&1
    sleep 30
    while systemctl --user is-active -q "rl-$name"; do sleep 30; done
  fi
  if python3 scripts/rl/check_fork_smoke.py "$out" $flag >> "$log" 2>&1; then
    touch "$out/SMOKE_PASSED"; say "smoke $name passed"
  else
    touch "$out/SMOKE_FAILED"; say "smoke $name FAILED; chain stopped"; exit 0
  fi
}
smoke smoke_grpo_m5 configs/rl/activevln_grpo_m5_r2r1k.yaml --no-forks
smoke smoke_fork_bs_m5 configs/rl/hico_grpo_fork_bs_m5_r2r1k.yaml

# M5_E1 itself, evaluated by the same supervisor/evaluator as an RL export at "step 0".
BASE=$R/runs/m5_e1_base
mkdir -p "$BASE/exports"
[ -e "$BASE/exports/step_0000" ] || ln -s /data/anh/checkpoints/simplemem/streaming_vggt/M5_E1/epoch-1 "$BASE/exports/step_0000"
[ -f "$BASE/COMPLETE" ] || echo '{"note": "M5_E1 published checkpoint, no RL training"}' > "$BASE/COMPLETE"
say "evaluate M5_E1"
bash scripts/rl/supervise.sh m5_e1_base none 0 || { say "M5_E1 evaluation failed"; exit 1; }

export BASELINE=$R/evals/m5_e1_base-step0000/val_unseen BASELINE_NAME=m5_e1
for spec in "grpo_m5_r2r1k configs/rl/activevln_grpo_m5_r2r1k.yaml 125" \
            "grpo_fork_bs_m5_r2r1k configs/rl/hico_grpo_fork_bs_m5_r2r1k.yaml 125"; do
  set -- $spec
  say "supervise $spec"
  bash scripts/rl/supervise.sh "$1" "$2" "$3" || { say "$1 failed"; exit 1; }
done
say "chain_m5 done"
