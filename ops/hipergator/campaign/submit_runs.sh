#!/bin/bash
# Train + evaluate configs in order on snapshot v5, starting after job AFTER (serial: group cap 2 GPUs).
# Usage: submit_runs.sh AFTER NAME:EVALS [NAME:EVALS ...]
#   EVALS=all   val_seen-300 at every export + step 125 val_unseen (methods)
#   EVALS=final val_seen-300 and val_unseen at step 125 only (controls, seeds)
# Each training job is followed (afterok) by a CPU job moving its last resumable checkpoint to /orange.
set -euo pipefail
AFTER=${1:?}; shift
OPS=/blue/yu954774.ucf/an221229.ucf/code/svln/ops
RUNS=/blue/yu954774.ucf/an221229.ucf/svln_runs
export SMV_ROOT=$RUNS/code/s8n_rl_v5_20261029
source "$OPS/env_hpg.sh" >/dev/null
SUBSET=$SMV_ROOT/data/rl/r2r_val_seen_subset300.json
sub() { sbatch --parsable --export=ALL $(exclude_bad_nodes) "$@"; }
prev=$AFTER
for spec in "$@"; do
  name=${spec%%:*}; evals=${spec#*:}; run=$RUNS/rl/$name
  train=$(CONFIG=configs/rl/$name.yaml RUN=$run sub --dependency=afterany:$prev --job-name=rl-${name%_r2r1k} "$OPS/train_rl.sbatch")
  arch=$(sbatch --parsable --dependency=afterok:$train "$OPS/archive_to_orange.sbatch" "$run/resumable/LATEST")
  steps="0025 0050 0075 0100 0125"; [[ $evals == final ]] && steps=0125
  last=$train
  for step in $steps; do
    last=$(CHECKPOINT=$run/exports/step_$step OUT=$RUNS/evals/$name/step_$step/val_seen300 SPLIT=val_seen \
           EPISODES=$SUBSET sub --dependency=afterok:$train,afterany:$last --time=06:00:00 \
           --job-name=evs-${name%_r2r1k}-$step "$OPS/eval_r2r.sbatch")
  done
  last=$(CHECKPOINT=$run/exports/step_0125 OUT=$RUNS/evals/$name/step_0125/val_unseen SPLIT=val_unseen \
         sub --dependency=afterok:$train,afterany:$last --job-name=evu-${name%_r2r1k}-0125 "$OPS/eval_r2r.sbatch")
  echo "$name train $train archive $arch evals ...$last"
  prev=$last
done
echo "LAST=$prev"
