#!/usr/bin/env bash
set -euo pipefail
G1_ROOT=/mnt/data/vmo-ai-task/anhdh35
cd "${G1_CODE_DIR:-$G1_ROOT/SimpleMemVLN-c2_vggt_omega_g1}"
export CUDA_HOME="${G1_CUDA_HOME:-$G1_ROOT/cuda-12.8.1}"
export PATH="$CUDA_HOME/bin:$G1_ROOT/SimpleMemVLN/.venv/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD:$G1_ROOT/third_party/vggt-omega"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}" OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
export VLN_MODEL_PATH="$G1_ROOT/cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
export TMPDIR="/tmp/g1-omega-${SLURM_JOB_ID:-manual}" TRITON_CACHE_DIR="/tmp/g1-omega-triton-${SLURM_JOB_ID:-manual}"
# CPU AdamW uses -march=native. Keep its binary cache local to each worker.
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/g1-torch210-cu129-${SLURMD_NODENAME:-$(hostname)}}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
exec "$@"
