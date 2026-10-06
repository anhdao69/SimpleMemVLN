#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
EVAL_PYTHON="${EVAL_PYTHON:-/storage/anhdh35/miniconda3/envs/spatialstack_dagger/bin/python}"
MODEL_PYTHON="${MODEL_PYTHON:-$ROOT/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-$ROOT/artifacts/checkpoints/window8-b/epoch-1}"
MODEL_PATH="${MODEL_PATH:-$ROOT/artifacts/checkpoints/qwen-base}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/artifacts/evaluation/window8-b/epoch-1}"
R2R_DATA="${R2R_DATA:-}"
if [[ -z "$R2R_DATA" ]]; then
  R2R_DATA='/storage/anhdh35/SpatialForcing-VLN/data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz'
fi
MP3D_SCENES="${MP3D_SCENES:-/storage/anhdh35/SpatialForcing-VLN/data/scene_datasets}"
EVAL_GPU="${EVAL_GPU:-0}"
SPLIT="${SPLIT:-val_unseen}"
test -s "$CHECKPOINT/navigation.json"
if [[ ! -s "$CHECKPOINT/pytorch_model.bin" && ! -s "$CHECKPOINT/model.safetensors" ]]; then
  echo 'Window8 epoch-1 weights missing' >&2
  exit 2
fi
unset VIRTUAL_ENV
export PYTHONPATH="$ROOT/src"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_DISABLE_PROGRESS_BARS=1 HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet
mkdir -p "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES="$EVAL_GPU" exec "$EVAL_PYTHON" -u -m qwen_vl.eval.habitat_r2r \
  --checkpoint "$CHECKPOINT" --model-python "$MODEL_PYTHON" --model-path "$MODEL_PATH" \
  --habitat-config "$ROOT/configs/eval/vln_r2r.yaml" \
  --data-path "$R2R_DATA" --scenes-dir "$MP3D_SCENES" \
  --gpu "$EVAL_GPU" --split "$SPLIT" --out "$OUTPUT_DIR" "$@"
