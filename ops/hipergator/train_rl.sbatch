#!/bin/bash
#SBATCH --job-name=svln-rl
#SBATCH --account=yu954774.ucf
#SBATCH --qos=yu954774.ucf
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=4-00:00:00
#SBATCH --output=/blue/yu954774.ucf/an221229.ucf/svln_logs/rl_%j.out
# One ActiveVLN-protocol RL run (2 ranks, ZeRO-2, IPC rollout helpers, one Habitat server per rank).
# Resumes automatically when RUN already holds rl_state.json.
# Usage: CONFIG=configs/rl/<name>.yaml RUN=/blue/.../runs/<name> [EXTRA="--max-steps 1 ..."] sbatch train_rl.sbatch
set -euo pipefail
source /blue/yu954774.ucf/an221229.ucf/code/svln/ops/env_hpg.sh
: "${CONFIG:?}" "${RUN:?}"
cd "$SMV_ROOT"
mkdir -p "$(dirname "$RUN")"
resume=()
[[ -f $RUN/rl_state.json ]] && resume=(--resume)
echo "host=$(hostname) gpus=$CUDA_VISIBLE_DEVICES start=$(date -Is) config=$CONFIG run=$RUN ${resume[*]:-}"
provenance
gpu_health
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
export OMP_NUM_THREADS=4 PYTORCH_ALLOC_CONF= PYTORCH_CUDA_ALLOC_CONF=
port=$((29500 + SLURM_JOB_ID % 2000))
.venv/bin/torchrun --nproc_per_node=2 --master_port "$port" -m qwen_vl.rl.train_grpo \
  --config "$CONFIG" --output-dir "$RUN" "${resume[@]}" ${EXTRA:-}
echo "end=$(date -Is)"
