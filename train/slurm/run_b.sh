#!/usr/bin/env bash
set -euo pipefail
root="${1:?source root}"
campaign="${2:?campaign path}"
variant="${3:?variant}"
cd "$root"
export PROJECT_ROOT="$root"
export PYTHONPATH="$root/src:$root"
export ENV_DIR=/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/.venv
export CUDA_HOME=/mnt/data/vmo-ai-task/anhdh35/cuda-12.8.1
export PATH="$CUDA_HOME/bin:$ENV_DIR/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
export NPROC_PER_NODE=4
export TMPDIR="/mnt/data/vmo-ai-task/anhdh35/tmp/smv-${SLURM_JOB_ID:?}"
mkdir -p "$TMPDIR"
model=/mnt/data/vmo-ai-task/anhdh35/cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
manifest=/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/artifacts/r2r_train.jsonl
export VLN_MODEL_PATH="$model"
"$ENV_DIR/bin/python" -m scripts.vln.campaign_preflight --root "$root" --campaign "$campaign" --variant "$variant" --manifest "$manifest"
common=(--vln_config configs/vln_r2r_v0_base.yaml --output_config configs/vln_r2r_b_3epoch_bs8.yaml --model_name_or_path "$model" --manifest "$manifest")
if [[ "$variant" == preflight ]]; then
    "$ENV_DIR/bin/python" -m pytest tests/vln -q > "$campaign/pytest.log" 2>&1
    "$ENV_DIR/bin/torchrun" --standalone --nproc_per_node=4 -m scripts.vln.check_epoch_saving --out "$campaign/epoch_save_test" > "$campaign/epoch_save_test.log" 2>&1
    "$ENV_DIR/bin/torchrun" --standalone --nproc_per_node=4 -m scripts.vln.profile_episode "${common[@]}" --out "$campaign/profile_fullcontext_b" > "$campaign/profile_fullcontext_b.log" 2>&1
    "$ENV_DIR/bin/python" -m scripts.vln.campaign_preflight --root "$root" --campaign "$campaign" --variant gate --manifest "$manifest"
    "$ENV_DIR/bin/python" -m scripts.vln.submit_campaign --root "$root" --campaign "$campaign" --verified-preflight
    exit 0
fi
[[ "$variant" == fullcontext || "$variant" == window8 ]] || exit 2
run="$campaign/$variant"
[[ ! -e "$run" ]] || { echo "Output already exists; explicit resume required" >&2; exit 1; }
mkdir -p "$run"
if [[ "$variant" == window8 ]]; then common+=(--memory_config configs/vln_memory_window8.yaml); fi
# Slurm's allocated device indices select only our four GPUs.
nvidia-smi --id="${CUDA_VISIBLE_DEVICES:?}" --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,memory.total --format=csv -l 5 > "$run/gpu_utilization.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
sleep 1
kill -0 "$monitor_pid"
bash train.sh "${common[@]}" --output_dir "$run" --run-name "r2r-${variant}-b-bs8-e3" > "$run/train.log" 2>&1
"$ENV_DIR/bin/python" -m scripts.vln.summarize_run "$run" > "$run/summary.log"
"$ENV_DIR/bin/python" -m scripts.vln.campaign_preflight --root "$root" --campaign "$campaign" --variant "complete-$variant" --manifest "$manifest"
