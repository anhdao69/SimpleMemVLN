#!/bin/bash
# D1 passed (s8n_event_rl_design.md §8): after the ActiveVLN GRPO chain, on snapshot v4:
#  1. base S8N val_seen-300 re-run on v4 (code-path check: must reproduce 155/300 episode for episode)
#  2. F-B step 125 + start tilt (b = -2) val_seen-300 (does the retention effect add to RL?)
#  3. memory twins train + val_seen-300 evals of every export + step 125 val_unseen
#  4. base S8N + start tilt val_unseen (confirms the inference-only probe on the primary split)
# then the G=8 control (dependency moved here with scontrol update).
set -euo pipefail
AFTER=${1:?job to start after}
OPS=/blue/yu954774.ucf/an221229.ucf/code/svln/ops
RUNS=/blue/yu954774.ucf/an221229.ucf/svln_runs
export SMV_ROOT=$RUNS/code/s8n_rl_v4_20261028
source "$OPS/env_hpg.sh"
SUBSET=$SMV_ROOT/data/rl/r2r_val_seen_subset300.json
sub() { sbatch --parsable --export=ALL $(exclude_bad_nodes) "$@"; }
c1=$(CHECKPOINT=/blue/yu954774.ucf/an221229.ucf/checkpoints/svln/M5_S/S8N OUT=$RUNS/evals/s8n_base/val_seen300_v4 \
     SPLIT=val_seen EPISODES=$SUBSET sub --dependency=afterany:$AFTER --time=06:00:00 --job-name=check-base-v4 "$OPS/eval_r2r.sbatch")
c2=$(CHECKPOINT=$RUNS/probes/fb125_tilt_start OUT=$RUNS/evals/probes/fb125_tilt_start/val_seen300 \
     SPLIT=val_seen EPISODES=$SUBSET sub --dependency=afterany:$c1 --time=06:00:00 --job-name=probe-fb125-start "$OPS/eval_r2r.sbatch")
name=memory_twins_s8n_r2r1k; run=$RUNS/rl/$name
train=$(CONFIG=configs/rl/$name.yaml RUN=$run sub --dependency=afterany:$c2 --job-name=rl-${name%_r2r1k} "$OPS/train_rl.sbatch")
last=$train
for step in 0025 0050 0075 0100 0125; do
  last=$(CHECKPOINT=$run/exports/step_$step OUT=$RUNS/evals/$name/step_$step/val_seen300 SPLIT=val_seen \
         EPISODES=$SUBSET sub --dependency=afterok:$train,afterany:$last --time=06:00:00 \
         --job-name=evs-${name%_r2r1k}-$step "$OPS/eval_r2r.sbatch")
done
last=$(CHECKPOINT=$run/exports/step_0125 OUT=$RUNS/evals/$name/step_0125/val_unseen SPLIT=val_unseen \
       sub --dependency=afterok:$train,afterany:$last --job-name=evu-${name%_r2r1k}-0125 "$OPS/eval_r2r.sbatch")
c4=$(CHECKPOINT=$RUNS/probes/s8n_tilt_start OUT=$RUNS/evals/probes/s8n_tilt_start/val_unseen SPLIT=val_unseen \
     sub --dependency=afterany:$last --job-name=probe-start-vu "$OPS/eval_r2r.sbatch")
echo "checks $c1 $c2 twins train $train evals ...$last start-vu $c4"
echo "LAST=$c4"
