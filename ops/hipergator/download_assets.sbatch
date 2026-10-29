#!/bin/bash
#SBATCH --job-name=svln-download
#SBATCH --account=yu954774.ucf
#SBATCH --qos=yu954774.ucf-b
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/blue/yu954774.ucf/an221229.ucf/svln_logs/download_%j.out
set -euo pipefail
export HF_TOKEN="$(cat /blue/yu954774.ucf/an221229.ucf/.cache/svln_hf_token)"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_ENABLE_HF_TRANSFER=0
PY=/blue/yu954774.ucf/an221229.ucf/envs/hftools/bin/python
OPS=/blue/yu954774.ucf/an221229.ucf/code/svln/ops
DATA=/orange/yu954774.ucf/an221229.ucf/svln_data
CKPT=/blue/yu954774.ucf/an221229.ucf/checkpoints/svln
REC=$DATA/records
cd "$SLURM_SUBMIT_DIR"
echo "host=$(hostname) start=$(date -Is)"
"$PY" -I "$OPS/download_assets.py" --data-root "$DATA" --ckpt-root "$CKPT" \
    --record "$REC/${RECORD:-assets}.json" --parts "${PARTS:-r2r,rxr15,s8n,scenes}"
# Missing train scans: project script, pinned fallback mirror accepted only if byte-identical.
TRAIN_JSON="$DATA/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz"
[[ "${SKIP_TRAIN_SCENES:-0}" == 1 ]] || "$PY" -I /blue/yu954774.ucf/an221229.ucf/code/svln/RL_memfork/scripts/rl/download_train_scenes.py \
    --train-json "$TRAIN_JSON" --scenes-dir "$DATA/scene_datasets" --record "$REC/train-scene-mirror-verified.json"
echo "end=$(date -Is)"
