#!/bin/bash
#SBATCH --job-name=svln-dev
#SBATCH --account=yu954774.ucf
#SBATCH --qos=yu954774.ucf
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/blue/yu954774.ucf/an221229.ucf/svln_logs/dev_%j.out
# Development allocation on 2 x RTX PRO 6000 that never idles: runs DEV/queue/*.sh one at a time (sorted;
# add tasks while it runs) and releases the GPUs IDLE_EXIT minutes after the queue empties, or at once
# when DEV/STOP exists (HiPerGator cancels GPU jobs idle for 1 h). Each task's output goes to
# DEV/done/<task>.log with its exit code in DEV/done/<task>.rc. Ad-hoc commands: srun --jobid=<id> --overlap ...
# Usage: DEV=/blue/.../svln_runs/dev/<name> [IDLE_EXIT=20] sbatch $(exclude_bad_nodes) dev_session.sbatch
set -euo pipefail
source /blue/yu954774.ucf/an221229.ucf/code/svln/ops/env_hpg.sh
: "${DEV:?}"
IDLE_EXIT=${IDLE_EXIT:-20}
mkdir -p "$DEV/queue" "$DEV/done"
echo "host=$(hostname) gpus=$CUDA_VISIBLE_DEVICES job=$SLURM_JOB_ID start=$(date -Is) dev=$DEV"
provenance
gpu_health
idle_since=$(date +%s)
while [[ ! -e $DEV/STOP ]]; do
  task=$(find "$DEV/queue" -maxdepth 1 -name '*.sh' | sort | head -1)
  if [[ -z $task ]]; then
    (( $(date +%s) - idle_since > IDLE_EXIT * 60 )) && { echo "queue empty for ${IDLE_EXIT} min"; break; }
    sleep 15; continue
  fi
  name=$(basename "$task" .sh)
  mv "$task" "$DEV/done/$name.sh"
  echo "$(date -Is) start $name"
  rc=0; (cd "$SMV_ROOT" && bash "$DEV/done/$name.sh") > "$DEV/done/$name.log" 2>&1 || rc=$?
  echo "$rc" > "$DEV/done/$name.rc"
  echo "$(date -Is) end $name rc=$rc"
  idle_since=$(date +%s)
done
echo "end=$(date -Is)"
