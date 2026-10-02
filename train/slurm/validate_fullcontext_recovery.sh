#!/usr/bin/env bash
set -euo pipefail
campaign=/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/outputs/fullcontext_recovery_20261001
cd "$campaign/source"
export PYTHONPATH="$PWD/src:$PWD"
export ENV_DIR=/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/.venv
export CUDA_HOME=/mnt/data/vmo-ai-task/anhdh35/cuda-12.8.1
export PATH="$CUDA_HOME/bin:$ENV_DIR/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
TMPDIR=$(mktemp -d /tmp/smv-recovery-test-XXXXXX)
export TMPDIR
export VLN_MODEL_PATH=/mnt/data/vmo-ai-task/anhdh35/cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
python -m pytest -q > "$campaign/pytest-gpu.log" 2>&1
for phase in baseline interrupted resume; do
    torchrun --standalone --nproc_per_node=4 -m scripts.vln.check_recovery_saving \
        --root "$campaign/recovery_test" --phase "$phase" > "$campaign/recovery-$phase.log" 2>&1
done
for case in failed worst; do
    torchrun --standalone --nproc_per_node=4 -m scripts.vln.profile_input_staging \
        --case "$case" --out "$campaign/profile-$case" \
        --vln_config configs/vln_r2r_v0_base.yaml \
        --output_config configs/vln_joint_b_2epoch_bs8.yaml \
        --model_name_or_path "$VLN_MODEL_PATH" \
        --manifest /mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl \
        > "$campaign/profile-$case.log" 2>&1
    python -m scripts.vln.profile_input_staging --verify-only \
        --case "$case" --out "$campaign/profile-$case" \
        --manifest /mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl \
        >> "$campaign/profile-$case.log" 2>&1
done
