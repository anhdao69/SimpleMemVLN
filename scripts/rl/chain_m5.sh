#!/usr/bin/env bash
# M5 RL queue (user request 2026-10-22), run by the persistent user unit rl-chain-m5.service:
#   wait for the Window8 chain (G = 8 control + its evaluation)
#   -> 2-step smokes: GRPO and GRPO+F-B on M5_E1 (this checkout, RL_m5), MemFork (RL_memfork worktree)
#   -> local val_unseen evaluation of M5_E1 (reference journal for paired tests), concurrently the
#      val_seen-300 development pilot: M5_E1 native token memory vs the MemFork frame-level prior
#   -> GRPO on M5_E1 -> GRPO+F-B on M5_E1 -> MemFork -> A2 (MemFork-GRPO, no forks)
#      -> A1 (frame-greedy memory + F-B) -> GRPO G = 8 on M5_E1 (matched interaction)
#   each evaluated on val_unseen once at its final step 125.
# Steps are idempotent (supervise.sh resumes training, skips finished shards, exits 0 when evaluated),
# so a restart after a host OOM continues where it stopped. A failed baseline smoke writes SMOKE_FAILED
# and ends the chain with exit 0 (no systemd restart loop); a failed MemFork smoke only skips the
# MemFork runs. Delete the marker to retry after a fix.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); cd "$ROOT"
MF=/data/anh/code/SimpleMem-memfork
R=/data/anh/rl_baseline; log=$R/logs/queue_m5.log
say() { echo "$(date -Is) $*" >> "$log"; }
say "chain_m5 start ($ROOT @ $(git rev-parse --short HEAD); memfork @ $(git -C $MF rev-parse --short HEAD))"
while systemctl --user is-active -q rl-chain; do sleep 120; done

smoke() {   # CHECKOUT NAME CONFIG [checker flags...]; returns 1 on failure
  local root=$1 name=$2 config=$3 out=$R/runs/$2; shift 3
  if [ -f "$out/SMOKE_PASSED" ]; then return 0; fi
  if [ -f "$out/SMOKE_FAILED" ]; then say "smoke $name failed earlier"; return 1; fi
  if [ ! -f "$out/STOPPED" ]; then
    rm -rf "$out"
    say "smoke $name ($root)"
    local exports=""                    # MemFork smokes also export step 2 (retention file check)
    case " $* " in *" --memory "*) exports=2 ;; esac
    (cd "$root" && scripts/rl/launch_unit.sh "rl-$name" "$R/logs/$name.log" /usr/bin/env PYTORCH_ALLOC_CONF= \
      PYTORCH_CUDA_ALLOC_CONF= /data/anh/code/SimpleMem/.venv/bin/torchrun --nproc_per_node=2 \
      --master_port 29681 -m qwen_vl.rl.train_grpo --config "$config" --output-dir "$out" --max-steps 2 \
      --prompts-per-step 4 --group-size 2 --budget-limit 60 --export-steps "$exports" \
      --save-every 1000) >> "$log" 2>&1
    sleep 30
    while systemctl --user is-active -q "rl-$name"; do sleep 30; done
  fi
  if python3 "$root/scripts/rl/check_fork_smoke.py" "$out" "$@" >> "$log" 2>&1; then
    touch "$out/SMOKE_PASSED"; say "smoke $name passed"
  else
    touch "$out/SMOKE_FAILED"; say "smoke $name FAILED"; return 1
  fi
}
smoke "$ROOT" smoke_grpo_m5 configs/rl/activevln_grpo_m5_r2r1k.yaml --no-forks || { say "chain stopped"; exit 0; }
smoke "$ROOT" smoke_fork_bs_m5 configs/rl/hico_grpo_fork_bs_m5_r2r1k.yaml || { say "chain stopped"; exit 0; }
memfork_ok=1
smoke "$MF" smoke_memfork_m5 configs/rl/memfork_m5_r2r1k.yaml --memory || memfork_ok=0

pilot() {   # NAME CHECKPOINT PHYSICAL_GPU: val_seen-300 development evaluation (never val_unseen)
  local out=$R/evals/pilot_valseen300/$1
  if [ -f "$out/summary.json" ] && grep -q '"complete": true' "$out/summary.json"; then return 0; fi
  systemctl --user is-active -q "rl-pilot-$1" && return 0
  rm -rf "$out"; mkdir -p "$out"; say "pilot $1 on GPU $3"
  (cd "$MF" && scripts/rl/launch_unit.sh "rl-pilot-$1" "$R/logs/pilot-$1.log" bash scripts/rl/eval_export.sh \
    "$2" "$3" "$out" --split val_seen --episode-list "$R/evals/pilot_valseen300/ids_valseen300.json") >> "$log" 2>&1
}
pilot m5_e1 /data/anh/checkpoints/simplemem/streaming_vggt/M5_E1/epoch-1 0
pilot m5_e1_frame_prior $R/runs/m5_e1_frame_prior/ckpt 1

# M5_E1 itself, evaluated by the same supervisor/evaluator as an RL export at "step 0".
BASE=$R/runs/m5_e1_base
mkdir -p "$BASE/exports"
[ -e "$BASE/exports/step_0000" ] || ln -s /data/anh/checkpoints/simplemem/streaming_vggt/M5_E1/epoch-1 "$BASE/exports/step_0000"
[ -f "$BASE/COMPLETE" ] || echo '{"note": "M5_E1 published checkpoint, no RL training"}' > "$BASE/COMPLETE"
say "evaluate M5_E1"
bash scripts/rl/supervise.sh m5_e1_base none 0 || { say "M5_E1 evaluation failed"; exit 1; }
while systemctl --user is-active -q rl-pilot-m5_e1 || systemctl --user is-active -q rl-pilot-m5_e1_frame_prior; do
  sleep 60; done
say "pilots finished"

export BASELINE=$R/evals/m5_e1_base-step0000/val_unseen BASELINE_NAME=m5_e1
run() {   # CHECKOUT NAME CONFIG STEPS
  say "supervise $2 $3 $4 ($1 @ $(git -C "$1" rev-parse --short HEAD))"
  bash "$1/scripts/rl/supervise.sh" "$2" "$3" "$4" || { say "$2 failed"; exit 1; }
}
run "$ROOT" grpo_m5_r2r1k configs/rl/activevln_grpo_m5_r2r1k.yaml 125
run "$ROOT" grpo_fork_bs_m5_r2r1k configs/rl/hico_grpo_fork_bs_m5_r2r1k.yaml 125
if [ $memfork_ok = 1 ]; then
  run "$MF" memfork_m5_r2r1k configs/rl/memfork_m5_r2r1k.yaml 125
  run "$MF" memfork_grpo_m5_r2r1k configs/rl/memfork_grpo_m5_r2r1k.yaml 125
  run "$MF" frame_greedy_fb_m5_r2r1k configs/rl/frame_greedy_fb_m5_r2r1k.yaml 125
else
  say "MemFork smoke failed: MemFork runs skipped (fix, delete $R/runs/smoke_memfork_m5, restart rl-chain-m5)"
fi
run "$ROOT" grpo_g8_m5_r2r1k configs/rl/activevln_grpo_g8_m5_r2r1k.yaml 125
say "chain_m5 done"
