#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$PROJECT_ROOT"

: "${ANNOTATION_PATH:?Set ANNOTATION_PATH to the Uniform-8 annotation JSON file}"
: "${DATA_ROOT:?Set DATA_ROOT to the directory containing the referenced images}"
export ANNOTATION_PATH DATA_ROOT

if [[ ! -f "$ANNOTATION_PATH" ]]; then
    echo "Annotation file does not exist: $ANNOTATION_PATH" >&2
    exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "Dataset directory does not exist: $DATA_ROOT" >&2
    exit 1
fi

ENV_DIR="${ENV_DIR:-${VIRTUAL_ENV:-$PROJECT_ROOT/.venv}}"
if [[ ! -f "$ENV_DIR/bin/activate" || ! -x "$ENV_DIR/bin/python" ]]; then
    echo "Missing Python environment: $ENV_DIR (run 'uv sync --locked')" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${CUDA_HOME:-}" ]]; then
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "CUDA_HOME is unset and nvcc was not found; load a CUDA toolkit" >&2
        exit 1
    fi
    CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
    export CUDA_HOME
fi
export PATH="$CUDA_HOME/bin:$PATH"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi was not found; an NVIDIA GPU environment is required" >&2
    exit 1
fi
visible_gpu_count="$(nvidia-smi --list-gpus | awk 'END { print NR + 0 }')"
if (( visible_gpu_count < 1 )); then
    echo "No NVIDIA GPUs are visible" >&2
    exit 1
fi

DATASET_CONFIG="${DATASET_CONFIG:-$PROJECT_ROOT/configs/r2r_uniform8.json}"
if [[ ! -f "$DATASET_CONFIG" ]]; then
    echo "Dataset config does not exist: $DATASET_CONFIG" >&2
    exit 1
fi

"$ENV_DIR/bin/python" - "$DATASET_CONFIG" <<'PY'
import json
import os
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)

for key, predicate in (
    ("annotation_path", os.path.isfile),
    ("data_path", os.path.isdir),
):
    path = os.path.expandvars(os.path.expanduser(config[key]))
    if not predicate(path):
        raise SystemExit(f"Dataset {key} does not exist: {path}")

print(f">>>>> dataset={config['dataset_name']}")
print(f">>>>> annotation={os.path.expandvars(config['annotation_path'])}")
print(f">>>>> data_root={os.path.expandvars(config['data_path'])}")
PY

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$PROJECT_ROOT/deepspeed.json}"
CACHE_DIR="${CACHE_DIR:-${HF_HUB_CACHE:-$PROJECT_ROOT/cache}}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output/uniform8}"

NPROC_PER_NODE="${NPROC_PER_NODE:-$visible_gpu_count}"
if ! [[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
    echo "NPROC_PER_NODE must be a positive integer" >&2
    exit 2
fi
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29531}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
MM_PROJECTOR_LR="${MM_PROJECTOR_LR:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-1}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
allocated_cpus="${CPU_COUNT:-$((NPROC_PER_NODE * 5))}"
default_workers=$((allocated_cpus / NPROC_PER_NODE - 1))
(( default_workers < 0 )) && default_workers=0
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-$default_workers}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-False}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
SAVE_FINAL_MODEL="${SAVE_FINAL_MODEL:-True}"
OPTIM="${OPTIM:-adamw_torch_fused}"
SPARSE_ACTION_LOGITS="${SPARSE_ACTION_LOGITS:-True}"
SKIP_MEMORY_METRICS="${SKIP_MEMORY_METRICS:-True}"

LOCAL_TMPDIR="${LOCAL_TMPDIR:-${TMPDIR:-/tmp}/simplememvln-$$}"
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR" "$LOCAL_TMPDIR/triton-cache"
export TMPDIR="$LOCAL_TMPDIR"
export TRITON_CACHE_DIR="$LOCAL_TMPDIR/triton-cache"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

train_args=(
    --model_name_or_path "$MODEL_PATH"
    --attn_implementation "$ATTN_IMPLEMENTATION"
    --tune_mm_llm True
    --tune_mm_mlp True
    --tune_mm_vision False
    --dataset_config "$DATASET_CONFIG"
    --max_history_frames 8
    --max_samples "$MAX_SAMPLES"
    --shuffle True
    --sparse_action_logits "$SPARSE_ACTION_LOGITS"
    --output_dir "$OUTPUT_DIR"
    --cache_dir "$CACHE_DIR"
    --bf16 True
    --tf32 True
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --learning_rate "$LEARNING_RATE"
    --mm_projector_lr "$MM_PROJECTOR_LR"
    --optim "$OPTIM"
    --model_max_length 12800
    --max_pixels $((576 * 28 * 28))
    --min_pixels $((16 * 28 * 28))
    --num_train_epochs "$NUM_TRAIN_EPOCHS"
    --lr_scheduler_type cosine
    --warmup_steps "$WARMUP_STEPS"
    --weight_decay 0.01
    --logging_steps "$LOGGING_STEPS"
    --save_strategy "$SAVE_STRATEGY"
    --save_steps "$SAVE_STEPS"
    --save_total_limit "$SAVE_TOTAL_LIMIT"
    --save_final_model "$SAVE_FINAL_MODEL"
    --deepspeed "$DEEPSPEED_CONFIG"
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING"
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
    --group_by_modality_length True
    --ddp_find_unused_parameters False
    --seed 42
    --report_to none
    --skip_memory_metrics "$SKIP_MEMORY_METRICS"
)

if (( DATALOADER_NUM_WORKERS > 0 )); then
    train_args+=(--dataloader_persistent_workers True --dataloader_prefetch_factor 4)
fi
if [[ -n "$MAX_STEPS" ]]; then
    if ! [[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]]; then
        echo "MAX_STEPS must be a positive integer" >&2
        exit 2
    fi
    train_args+=(--max_steps "$MAX_STEPS")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    train_args+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

echo ">>>>> Qwen3.5-4B Uniform-8 SFT on $NPROC_PER_NODE GPUs"
echo ">>>>> attention=$ATTN_IMPLEMENTATION deepspeed=$(basename "$DEEPSPEED_CONFIG") optimizer=$OPTIM"
echo ">>>>> micro_batch=$PER_DEVICE_TRAIN_BATCH_SIZE gradient_accumulation=$GRADIENT_ACCUMULATION_STEPS"
start_seconds=$SECONDS
"$ENV_DIR/bin/python" -m torch.distributed.run \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    -m qwen_vl.train.train_qwen \
    "${train_args[@]}" \
    2>&1 | tee -a "$OUTPUT_DIR/train.log"
elapsed_seconds=$((SECONDS - start_seconds))
echo ">>>>> end-to-end wall time: ${elapsed_seconds}s" | tee -a "$OUTPUT_DIR/train.log"
