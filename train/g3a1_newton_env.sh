#!/usr/bin/env bash
# Newton (evc10x H100) environment for G3A1. No shared package/driver changes:
# isolated CUDA 12.9.1 build components (matching torch cu129) for DeepSpeed CPUAdam,
# GCC 12 for TileLang's C++20 JIT, pinned Omega assets and Qwen snapshot.
set -euo pipefail
export G3A1_ASSETS="${G3A1_ASSETS:-/groups/yshang/an221229/vggt_omega_assets}"
export G3A1_VENV="${G3A1_VENV:-/lustre/fs1/home/an221229/code/SimpleMemVLN-c2_episodic/.venv}"
cd "${G3A1_CODE_DIR:?G3A1_CODE_DIR must point at the (immutable) source tree}"
export CUDA_HOME="$G3A1_ASSETS/cuda-12.9.1"
export CXX="$G3A1_ASSETS/bin/g++" CC="$G3A1_ASSETS/bin/gcc"  # GCC 12 + -mno-avx512fp16 (EL8 assembler)
export LD_LIBRARY_PATH="/apps/gcc/gcc-12.2.0/lib64:${LD_LIBRARY_PATH:-}"
export PATH="$CUDA_HOME/bin:$G3A1_VENV/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD:$G3A1_ASSETS/third_party/vggt-omega"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS:-1}}" OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/groups/yshang/an221229/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
export VLN_MODEL_PATH=/groups/yshang/an221229/cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
export TMPDIR="/tmp/g3a1-${SLURM_JOB_ID:-manual}" TRITON_CACHE_DIR="/tmp/g3a1-triton-${SLURM_JOB_ID:-manual}"
# CPUAdam is built with -march=native: keep its binary cache node-local.
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$G3A1_ASSETS/torch_ext_$(hostname)}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
exec "$@"
