#!/usr/bin/env bash
# One independent persistent model and Habitat process per epoch, concurrently.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
MODEL_PYTHON="${MODEL_PYTHON:-$ROOT/.venv/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-/storage/anhdh35/miniconda3/envs/spatialstack_dagger/bin/python}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT/artifacts/checkpoints/fullcontext-b}"
MODEL_PATH="${MODEL_PATH:-$ROOT/artifacts/checkpoints/qwen-base}"
R2R_DATA="${R2R_DATA:-}"
if [[ -z "$R2R_DATA" ]]; then
    R2R_DATA='/storage/anhdh35/SpatialForcing-VLN/data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz'
fi
MP3D_SCENES="${MP3D_SCENES:-/storage/anhdh35/SpatialForcing-VLN/data/scene_datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/artifacts/evaluation/fullcontext-b}"
SPLIT="${SPLIT:-val_unseen}"
# GPU IDs are physical; repeating an ID shares that GPU across epochs.
read -r -a GPU_IDS <<< "${EVAL_GPUS:-0 0 0}"
if [[ ${#GPU_IDS[@]} != 3 ]]; then
    echo 'EVAL_GPUS must contain exactly three GPU IDs, one for each epoch' >&2
    exit 2
fi
unset VIRTUAL_ENV
export PYTHONPATH="$ROOT/src"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS="${OMP_NUM_THREADS}"
# Dynamic FullContext KV grows each decision; expandable segments avoid
# retaining differently sized allocations that compete across the three workers.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_DISABLE_PROGRESS_BARS=1 HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet
mkdir -p "$OUTPUT_DIR"
# Require complete local downloads before creating any benchmark workers.
for epoch in 1 2 3; do
    test -s "$CHECKPOINT_ROOT/epoch-$epoch/pytorch_model.bin"
    test -s "$CHECKPOINT_ROOT/epoch-$epoch/navigation.json"
done
test -s "$MODEL_PATH/model.safetensors.index.json"
pids=()
cleanup() {
    for pid in "${pids[@]}"; do kill -- "-$pid" 2>/dev/null || true; done
}
trap 'cleanup; exit 130' INT TERM
for epoch in 1 2 3; do
    gpu="${GPU_IDS[$((epoch - 1))]}"
    # Isolated process groups let interruption clean up both simulator and model.
    CUDA_VISIBLE_DEVICES="$gpu" setsid "$EVAL_PYTHON" -u -m qwen_vl.eval.habitat_r2r \
        --checkpoint "$CHECKPOINT_ROOT/epoch-$epoch" \
        --model-python "$MODEL_PYTHON" --model-path "$MODEL_PATH" \
        --habitat-config "$ROOT/configs/eval/vln_r2r.yaml" \
        --data-path "$R2R_DATA" --scenes-dir "$MP3D_SCENES" \
        --gpu "$gpu" --split "$SPLIT" --out "$OUTPUT_DIR/epoch-$epoch" \
        "$@" > "$OUTPUT_DIR/epoch-$epoch.log" 2>&1 &
    pids+=("$!")
    printf 'epoch-%s pid=%s gpu=%s log=%s\n' "$epoch" "$!" "$gpu" "$OUTPUT_DIR/epoch-$epoch.log"
done
printf '%s\n' "${pids[@]}" > "$OUTPUT_DIR/worker_pids.txt"
status=0
for index in 0 1 2; do
    if wait "${pids[$index]}"; then
        printf 'epoch-%s completed\n' "$((index + 1))"
    else
        code=$?
        printf 'epoch-%s failed (exit %s); see log\n' "$((index + 1))" "$code" >&2
        status=1
    fi
done
exit "$status"
