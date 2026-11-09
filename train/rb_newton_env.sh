#!/usr/bin/env bash
# Newton (evc10x H100) environment for the S8N R/B training ablation: the qualified G3A1 newton environment
# (train/g3a1_newton_env.sh on c2_vggt_omega_g3_a1) without the VGGT teacher. Same pinned venv as S8N's stack
# (torch 2.10.0+cu129, Transformers 5.11.0, FLA 0.5.2, FlashAttention 2.8.3, DeepSpeed 0.16.4), isolated CUDA
# 12.9.1 for the DeepSpeed CPU-Adam JIT, GCC 12 (-mno-avx512fp16 wrapper for the EL8 assembler), offline Hub.
set -euo pipefail
ASSETS="${RB_ASSETS:-/groups/yshang/an221229/vggt_omega_assets}"
VENV="${RB_VENV:-/lustre/fs1/home/an221229/code/SimpleMemVLN-c2_episodic/.venv}"
cd "${RB_SOURCE:?RB_SOURCE must point at the sealed source tree}"
export CUDA_HOME="$ASSETS/cuda-12.9.1"
export CXX="$ASSETS/bin/g++" CC="$ASSETS/bin/gcc"
export LD_LIBRARY_PATH="/apps/gcc/gcc-12.2.0/lib64:${LD_LIBRARY_PATH:-}"
export PATH="$CUDA_HOME/bin:$VENV/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS:-4}}" OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/groups/yshang/an221229/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
export VLN_MODEL_PATH=/groups/yshang/an221229/cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
export TMPDIR="/tmp/rbabl-${SLURM_JOB_ID:-manual}"
# One persistent Triton cache for every allocation and node. FLA autotunes per sequence-block count (key NB), so a
# per-job cache re-benchmarks and recompiles all through each segment (S8N_R4B600 on a cold cache: 34 s/update
# over its first ten updates vs 12.8 s for S8N on worker-2's warm cache). Triton writes entries atomically.
export TRITON_CACHE_DIR="${RB_TRITON_CACHE_DIR:-/groups/yshang/an221229/cache/triton/rbabl_h100}"
# CPU Adam is built with -march=native: keep its binary cache per node.
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$ASSETS/torch_ext_$(hostname)}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
exec "$@"
