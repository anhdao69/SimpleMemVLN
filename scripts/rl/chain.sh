#!/usr/bin/env bash
# Remaining research runs in order, then their analyses, in one process that systemd restarts.
# Host-RAM OOM kills (other users' jobs) have killed the transient supervisor/queue/post units several
# times, so this runs from a persistent, enabled user unit (Restart=on-failure, WantedBy=default.target):
#   ~/.config/systemd/user/rl-chain.service  ->  ExecStart=/bin/bash scripts/rl/chain.sh
# Every step is idempotent: supervise.sh resumes training from the last checkpoint, skips finished
# eval shards and exits 0 once the evaluation is merged; post_eval.sh/post_pooled.sh only rewrite
# deterministic analyses. A restart therefore just continues where the killed run stopped.
set -uo pipefail
cd /data/anh/code/SimpleMem
log=/data/anh/rl_baseline/logs/queue.log
echo "$(date -Is) chain start" >> "$log"
# v2a (hico_v2a_r2r1k) dropped 2026-10-22: the next runs start from the M5 checkpoint instead.
for spec in "grpo_g8_r2r1k configs/rl/activevln_grpo_g8_r2r1k.yaml 125"; do
  set -- $spec
  echo "$(date -Is) chain: supervise $spec" >> "$log"
  bash scripts/rl/supervise.sh "$1" "$2" "$3" || { echo "$(date -Is) chain: $1 failed" >> "$log"; exit 1; }
done
E=/data/anh/rl_baseline/evals
bash scripts/rl/post_eval.sh "grpo_g8_step125 $E/grpo_g8_r2r1k-step0125"
bash scripts/rl/post_pooled.sh
echo "$(date -Is) chain done" >> "$log"
