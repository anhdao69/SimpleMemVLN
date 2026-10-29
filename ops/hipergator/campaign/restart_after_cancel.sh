#!/bin/bash
# Restart of the S8N RL campaign after the 2026-10-29 08:47 admin cancellation, with a smaller queue footprint:
# at most one training run and its evaluations are queued at a time, and every dependency is afterany, so a
# failed job can never leave jobs pending forever (an evaluation of a missing export fails within seconds).
# Order: memory twins on GRPO (the method) -> its step 125 val_unseen + val_seen-300 -> start-rule probe on
# val_unseen. Submit the G = 5 control with STAGE=g5 after this stage has finished, and the no-CRN ablation
# with STAGE=nocrn SNAPSHOT=<v6> (needs the common_random_numbers flag: run ops/cpu_tests.sbatch on RL_s8n,
# then freeze_code.sh s8n_rl_v6_<date>; v5 lacks the flag, its default behaviour equals CRN twins).
# Usage: STAGE=twins|g5|nocrn [SNAPSHOT=dir] restart_after_cancel.sh
set -euo pipefail
OPS=/blue/yu954774.ucf/an221229.ucf/code/svln/ops
RUNS=/blue/yu954774.ucf/an221229.ucf/svln_runs
export SMV_ROOT=${SNAPSHOT:-$RUNS/code/s8n_rl_v5_20261029}
source "$OPS/env_hpg.sh" >/dev/null
SUBSET=$SMV_ROOT/data/rl/r2r_val_seen_subset300.json
sub() { sbatch --parsable --export=ALL $(exclude_bad_nodes) "$@"; }
case ${STAGE:?} in
  twins) name=memory_twins_grpo_s8n_r2r1k ;;
  g5)    name=activevln_grpo_g5_s8n_r2r1k ;;
  nocrn) name=memory_twins_grpo_nocrn_s8n_r2r1k
         [[ -f $SMV_ROOT/configs/rl/$name.yaml ]] || { echo "set SNAPSHOT to a snapshot with $name" >&2; exit 1; } ;;
  *)     echo "unknown STAGE $STAGE" >&2; exit 1 ;;
esac
run=$RUNS/rl/$name
[[ -e $run ]] && { echo "$run exists; inspect it before restarting" >&2; exit 1; }
train=$(CONFIG=configs/rl/$name.yaml RUN=$run sub --job-name=rl-${name%_r2r1k} "$OPS/train_rl.sbatch")
arch=$(sbatch --parsable --dependency=afterany:$train "$OPS/archive_to_orange.sbatch" "$run/resumable/LATEST")
evu=$(CHECKPOINT=$run/exports/step_0125 OUT=$RUNS/evals/$name/step_0125/val_unseen SPLIT=val_unseen \
      sub --dependency=afterany:$train --job-name=evu-${name%_r2r1k}-0125 "$OPS/eval_r2r.sbatch")
evs=$(CHECKPOINT=$run/exports/step_0125 OUT=$RUNS/evals/$name/step_0125/val_seen300 SPLIT=val_seen \
      EPISODES=$SUBSET sub --dependency=afterany:$evu --time=06:00:00 --job-name=evs-${name%_r2r1k}-0125 "$OPS/eval_r2r.sbatch")
echo "$name train $train archive $arch val_unseen $evu val_seen300 $evs"
if [[ $STAGE == twins ]]; then
  out=$RUNS/evals/probes/s8n_tilt_start/val_unseen
  [[ -e $out ]] && mv "$out" "$out.cancelled_20261029"      # partial (259/1839) run killed by the admin
  probe=$(CHECKPOINT=$RUNS/probes/s8n_tilt_start OUT=$out SPLIT=val_unseen \
          sub --dependency=afterany:$evs --job-name=probe-start-vu "$OPS/eval_r2r.sbatch")
  echo "start-rule probe val_unseen $probe"
fi
