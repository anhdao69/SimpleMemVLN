#!/bin/bash
# S8N RL campaign on 2 x RTX PRO 6000 (group cap: 2 GPUs, so jobs run one at a time in this order).
# Usage: submit_campaign.sh CODE_SNAPSHOT [AFTER_JOBID]
#   CODE_SNAPSHOT: output of freeze_code.sh; AFTER_JOBID: e.g. the simulator build (afterok).
# Order: smoke -> S8N base eval (val_unseen, val_seen-300) -> F-B -> its evals -> GRPO -> its evals;
#        then (separately, after the F-B H1 analysis) EBF -> G=8.
# Evals per run: every export on val_seen-300 (selection), step 125 on val_unseen (pre-registered primary).
set -euo pipefail
CODE=${1:?code snapshot}
AFTER=${2:-}
OPS=/blue/yu954774.ucf/an221229.ucf/code/svln/ops
RUNS=/blue/yu954774.ucf/an221229.ucf/svln_runs
SUBSET=$CODE/data/rl/r2r_val_seen_subset300.json
export SMV_ROOT=$CODE
dep() { [[ -n $2 ]] && echo "--dependency=$1:$2" || true; }
source "$OPS/env_hpg.sh"
sub() { sbatch --parsable --export=ALL $(exclude_bad_nodes) "$@"; }

# SMOKE_LOG=<log>: output of a gpu_smoke.sbatch run that already passed on this snapshot; no new smoke is run.
if [[ -n ${SMOKE_LOG:-} ]]; then
  grep -q "^end=.*smoke=" "$SMOKE_LOG" || { echo "smoke $SMOKE_LOG did not pass" >&2; exit 1; }
  ok=$(dep afterok "$AFTER")
else
  smoke=$(sub $(dep afterok "$AFTER") "$OPS/gpu_smoke.sbatch")
  echo "smoke $smoke"
  ok=--dependency=afterok:$smoke
fi
# BASE_EVALS=skip: the S8N base evaluation was already run (e.g. in a dev_session.sbatch allocation).
# PREV=<jobid>: start the chain after that job ends (e.g. the last eval of an earlier campaign).
# RUN_NAMES: configs to train, in order. F-B runs first because EBF is calibrated and its trigger chosen
# from F-B rollouts (H1); EBF and the G=8 control are submitted afterwards with PREV and RUN_NAMES.
prev=${PREV:-}
if [[ ${BASE_EVALS:-run} != skip ]]; then
base=/blue/yu954774.ucf/an221229.ucf/checkpoints/svln/M5_S/S8N
e1=$(CHECKPOINT=$base OUT=$RUNS/evals/s8n_base/val_unseen SPLIT=val_unseen sub $ok \
     --job-name=eval-s8n-vu "$OPS/eval_r2r.sbatch")
e2=$(CHECKPOINT=$base OUT=$RUNS/evals/s8n_base/val_seen300 SPLIT=val_seen EPISODES=$SUBSET \
     sub --dependency=afterany:$e1 --time=06:00:00 --job-name=eval-s8n-vs "$OPS/eval_r2r.sbatch")
echo "base evals $e1 $e2"
prev=$e2
fi
for name in ${RUN_NAMES:-hico_grpo_fork_bs_s8n_r2r1k activevln_grpo_s8n_r2r1k}; do
  run=$RUNS/rl/$name
  train=$(CONFIG=configs/rl/$name.yaml RUN=$run sub $( [[ -n $prev ]] && echo "--dependency=afterany:$prev" || echo $ok ) \
          --job-name=rl-${name%_r2r1k} "$OPS/train_rl.sbatch")
  last=$train
  for step in 0025 0050 0075 0100 0125; do
    last=$(CHECKPOINT=$run/exports/step_$step OUT=$RUNS/evals/$name/step_$step/val_seen300 SPLIT=val_seen \
           EPISODES=$SUBSET sub --dependency=afterok:$train,afterany:$last --time=06:00:00 \
           --job-name=evs-${name%_r2r1k}-$step "$OPS/eval_r2r.sbatch")
  done
  last=$(CHECKPOINT=$run/exports/step_0125 OUT=$RUNS/evals/$name/step_0125/val_unseen SPLIT=val_unseen \
         sub --dependency=afterok:$train,afterany:$last --job-name=evu-${name%_r2r1k}-0125 "$OPS/eval_r2r.sbatch")
  echo "$name train $train evals ...$last"
  prev=$last
done
