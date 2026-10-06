#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
EVAL_PYTHON="${EVAL_PYTHON:-/storage/anhdh35/miniconda3/envs/spatialstack_dagger/bin/python}"
MODEL_PYTHON="${MODEL_PYTHON:-$ROOT/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-$ROOT/artifacts/checkpoints/r2r-rxr15-fullcontext-b/epoch-1}"
MODEL_PATH="${MODEL_PATH:-$ROOT/artifacts/checkpoints/qwen-base}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/artifacts/evaluation/r2r-rxr15-fullcontext-b/epoch-1}"
RXR_DATA="${RXR_DATA:-}"
RXR_GT="${RXR_GT:-}"
if [[ -z "$RXR_DATA" ]]; then
  RXR_DATA='/storage/anhdh35/JanusVLN/data/datasets/rxr_15deg/{split}/{split}_{role}.json.gz'
fi
if [[ -z "$RXR_GT" ]]; then
  RXR_GT='/storage/anhdh35/JanusVLN/data/datasets/rxr_15deg/{split}/{split}_{role}_gt.json.gz'
fi
MP3D_SCENES="${MP3D_SCENES:-/storage/anhdh35/JanusVLN/data/scene_datasets_full}"
SIMULATOR_SOURCE="${SIMULATOR_SOURCE:-/storage/anhdh35/SpatialStack_Dagger/src}"
EVAL_GPU="${EVAL_GPU:-0}"
SPLIT="${SPLIT:-val_unseen}"
test -s "$CHECKPOINT/navigation.json"
test -s "$CHECKPOINT/pytorch_model.bin"
test -s "$MODEL_PATH/model.safetensors.index.json"
unset VIRTUAL_ENV
export PYTHONPATH="$ROOT/src"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_DISABLE_PROGRESS_BARS=1 HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet
mkdir -p "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES="$EVAL_GPU" exec "$EVAL_PYTHON" -u -m qwen_vl.eval.habitat_r2r \
  --dataset-kind rxr15 --simulator-source "$SIMULATOR_SOURCE" \
  --checkpoint "$CHECKPOINT" --model-python "$MODEL_PYTHON" --model-path "$MODEL_PATH" \
  --habitat-config "$ROOT/configs/eval/vln_rxr15.yaml" \
  --data-path "$RXR_DATA" --ground-truth-path "$RXR_GT" \
  --scenes-dir "$MP3D_SCENES" --gpu "$EVAL_GPU" --split "$SPLIT" \
  --out "$OUTPUT_DIR" "$@"
