#!/usr/bin/env bash
set -euo pipefail
G=/mnt/data/vmo-ai-task/anhdh35/g3-fixed01-seed430-worker2-20261019
exec srun --jobid="${G3_ALLOCATION_ID:-4925}" --overlap --ntasks=1 --gpus=4 --cpus-per-task=60 --nodelist=worker-2 --job-name=g3a1_then_g3_fixed env G1_CODE_DIR="$G/source" G1_CUDA_HOME=/mnt/data/vmo-ai-task/anhdh35/cuda-12.9.1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MAX_JOBS=2 PYTHONDONTWRITEBYTECODE=1 bash "$G/source/train/g1_omega_env.sh" python -m scripts.vln.queue_g3a1_then_g3_fixed --config "$G/queue/queue.json"

