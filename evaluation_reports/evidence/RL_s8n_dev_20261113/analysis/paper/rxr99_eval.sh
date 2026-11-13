#!/bin/bash
# LOCAL 2026-11-13. User: "98 of the 3,669 RxR-CE val-unseen of the best 71.9 R2R is failed, right? Can you help to fix
# those eps and eval and update the lastest RxR result". The 99 RxR val_unseen episodes (all in scene oLBMNvg9in8) that
# every model failed with "Required simulator metrics unavailable" are unreachable on the R2R-agent navmesh (1.5 m /
# 0.1 m); Habitat rebuilds the navmesh for the RxR-CE agent (0.88 m / 0.18 m, VLN-CE's RxR config), on which the episodes
# were generated (GT actions reach the goal on all 99; analysis/paper/rxr99_fix.md). Re-evaluate exactly those 99
# episodes with that navmesh, everything else unchanged (frozen tree svln/code/rxrce_agent_eval_20261113: src/qwen_vl
# identical to the original RxR evaluations; camera 1.25 m; first frames identical). GPU 1 only, no MPS (the CoRe
# seed-1414 training shares GPU 1).
# Usage: setsid nohup bash /nvme/public/svln/analysis/paper/rxr99_eval.sh NAME CHECKPOINT > /dev/null 2>&1 < /dev/null &
set -uo pipefail
NAME=$1 CKPT=$2
C=/nvme/public/svln/code/rxrce_agent_eval_20261113
OUT=/nvme/public/svln/evals/${NAME}_rxr99_rxrce_agent_val_unseen
[[ -e $OUT ]] && { echo "$OUT exists"; exit 1; }
echo "$(date -Is) start $NAME ($CKPT) -> $OUT" >> /nvme/public/svln/evals/rxr99_eval.log
USE_MPS=0 SMV_ROOT=$C CUDA_VISIBLE_DEVICES=1 WORKERS_PER_GPU=${WORKERS:-3} CHECKPOINT=$CKPT OUT=$OUT \
  EPISODES=/nvme/public/svln/analysis/paper/rxr99_episode_ids.json bash $C/ops/local/eval_rxr_rxrce_agent.sh > $OUT.log 2>&1
rc=$?
echo "$(date -Is) end $NAME rc=$rc: $(python3 -I -c "
import json; d=json.load(open('$OUT/merged/summary.json')); print({k: d[k] for k in ('episodes', 'failures', 'sr', 'spl', 'oracle_success', 'failure_reasons')})" 2>&1)" >> /nvme/public/svln/evals/rxr99_eval.log
