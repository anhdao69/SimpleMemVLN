#!/bin/bash
# LOCAL 2026-11-13 22:3x. Pause CoRe seed 20261414 (rewind_sfr_mix1k_seed1414) while the three 52-episode RxR
# re-evaluations of rxr_fix_final_evals.sh run, then resume it. Reason: at 22:27:41 its rank 1 was SIGKILLed 43 s after
# the first of those evaluations started loading three model servers on GPU 1 (host memory); the supervisor resumed it
# from step_0005 at 22:29. The user allowed pausing the training for the RxR evaluations. The trainer is stopped while it
# is still loading step_0005 (no update lost). No STOPPED marker: hard2k_chain.sh / seed1414_chain.sh keep waiting.
# Resume = the supervisor relaunched with the environment launch_run.sh gave it (train_rl.sh resumes from rl_state.json).
set -uo pipefail
R=/nvme/public/svln/runs/rewind_sfr_mix1k_seed1414 FROZEN=/nvme/public/svln/code/rewind_sfr_mix1k_seed1414_20261113_2120
CONFIG=configs/rl/local/rewind_grpo_sfr_mix1k_seed1414.yaml
LOG=/nvme/public/svln/evals/rxr99_eval.log
say() { echo "$(date -Is) [local pause] $*" >> "$LOG"; }
tag() { python3 -I -c "import json; print(json.load(open('$R/rl_state.json'))['tag'])" 2>/dev/null; }
say "pausing rewind_sfr_mix1k_seed1414 (rl_state $(tag); resumable: $(ls "$R/resumable" | tr '\n' ' '))"
echo "=== $(date -Is) PAUSE (RxR evaluations; host memory); resumes from $(tag) ===" >> "$R.log"
pkill -u "$(id -u)" -f "$FROZEN/ops/local/[s]upervise_run.sh"; sleep 2
pkill -u "$(id -u)" -f "$FROZEN/ops/local/[t]rain_rl.sh|[t]orch.distributed.run.*$CONFIG|[q]wen_vl.rl.train_grpo --config $CONFIG"
sleep 30
pkill -9 -u "$(id -u)" -f "$FROZEN/ops/local/[t]rain_rl.sh|[t]orch.distributed.run.*$CONFIG|[q]wen_vl.rl.train_grpo --config $CONFIG"
pkill -u "$(id -u)" -f "SimpleMemVLN_rl_s8n/.venv/bin/python.*[m]ultiprocessing.spawn import spawn_main"
pkill -u "$(id -u)" -f "[q]wen_vl.rl.env_server --habitat-config $FROZEN/"
sleep 15
timeout 120 bash "$FROZEN/ops/local/mps.sh" stop >> "$LOG" 2>&1 || say "MPS stop failed or timed out"
left=$(pgrep -u "$(id -u)" -af "$FROZEN/ops/local/(supervise_run|train_rl)|qwen_vl.rl.train_grpo|spawn_main|qwen_vl.rl.env_server|nvidia-cuda-mps" | grep -v pgrep)
[[ -n $left ]] && say "left after the pause: $left"
say "paused at $(tag); GPUs: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
until grep -q "waiting for the CoRe seed-1414 chain" "$LOG" || ! pgrep -u "$(id -u)" -f "^bash /nvme/public/svln/analysis/paper/[r]xr_fix_final_evals.sh" > /dev/null; do sleep 20; done
while pgrep -u "$(id -u)" -f "/nvme/public/svln/code/rxrce_agent_eval_20261113/.*[h]abitat_r2r" > /dev/null; do sleep 10; done
say "52-episode evaluations done; resuming rewind_sfr_mix1k_seed1414 from $(tag)"
MPS_GPUS=0,1 bash "$FROZEN/ops/local/mps.sh" start >> "$LOG" 2>&1 || say "MPS start failed"
sleep 5
echo "=== $(date -Is) RESUME after the RxR evaluations (from $(tag)) ===" >> "$R.log"
(cd /nvme/public/SimpleMemVLN_dev && env SMV_ROOT=$FROZEN FROZEN=$FROZEN CONFIG=$CONFIG RUN=$R NAME=rewind_sfr_mix1k_seed1414 \
  RL_PROFILE_STEP= STEPS= FINAL_STEP=none TRAIN_GPUS=0,1 MAX_RESTARTS=20 \
  setsid nohup bash "$FROZEN/ops/local/supervise_run.sh" > /dev/null 2>&1 < /dev/null &)
say "supervisor relaunched"
n0=$(grep -c '"step"' "$R/metrics.jsonl" 2>/dev/null || echo 0)
for i in $(seq 1 60); do sleep 60; (( $(grep -c '"step"' "$R/metrics.jsonl" 2>/dev/null || echo 0) > n0 )) && break; done
say "after resume: $(tail -1 "$R/metrics.jsonl" | cut -c1-70); supervisor $(pgrep -u "$(id -u)" -f "$FROZEN/ops/local/[s]upervise_run.sh" > /dev/null && echo running || echo MISSING); drift guard $(pgrep -u "$(id -u)" -f "[d]rift_guard.py $R" > /dev/null && echo running || echo MISSING)"
