#!/usr/bin/env bash
# Launch one DAgger collection shard per GPU (R2R-VLNCE train) and wait for all of them.
# Every variable below must point at the same assets as the S8N evaluation and oracle manifest.
#
#   HABITAT_PYTHON=/storage/anhdh35/miniconda3/envs/spatialstack_dagger/bin/python \
#   MODEL_PYTHON=/storage/anhdh35/SimpleMemVLN/.venv/bin/python \
#   CHECKPOINT=/data14t/anhdh35/m5_s8n_epoch1/checkpoint/S8N \
#   MODEL_PATH=/path/to/Qwen3.5-4B \
#   DATA_PATH='/path/to/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz' \
#   SCENES_DIR=/path/to/scene_datasets \
#   ORACLE_MANIFEST=/path/to/joint_r2r_rxr15.jsonl \
#   OUT=/data14t/anhdh35/dagger/s8n_r2r_it1 GPUS="0 1 2 3" bash scripts/dagger/run_r2r_shards.sh
set -euo pipefail
: "${HABITAT_PYTHON:?}" "${MODEL_PYTHON:?}" "${CHECKPOINT:?}" "${MODEL_PATH:?}" "${DATA_PATH:?}"
: "${SCENES_DIR:?}" "${ORACLE_MANIFEST:?}" "${OUT:?}"
GPUS=${GPUS:-0}
TAG=${TAG:-s8n_r2r_it1}
MAX_EPISODES=${MAX_EPISODES:-0}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
read -r -a gpu_list <<< "$GPUS"
shards=${#gpu_list[@]}
mkdir -p "$OUT/logs"
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
pids=()
for index in "${!gpu_list[@]}"; do
  gpu=${gpu_list[$index]}
  "$HABITAT_PYTHON" -u -m qwen_vl.dagger.collect_r2r \
    --checkpoint "$CHECKPOINT" --model-python "$MODEL_PYTHON" --model-path "$MODEL_PATH" \
    --model-device "$gpu" --gpu "$gpu" --data-path "$DATA_PATH" --scenes-dir "$SCENES_DIR" \
    --oracle-manifest "$ORACLE_MANIFEST" --split train --tag "$TAG" --max-episodes "$MAX_EPISODES" \
    --shard-index "$index" --num-shards "$shards" --out "$OUT" \
    > "$OUT/logs/shard$(printf %03d "$index").log" 2>&1 &
  pids+=($!)
  sleep 60   # stagger model loads (host RAM peaks while weights load)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit $status
