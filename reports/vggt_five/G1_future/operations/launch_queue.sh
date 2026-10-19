#!/usr/bin/env bash
set -euo pipefail
exec srun --jobid=4925 --overlap --ntasks=1 --gpus=4 --cpus-per-task=60 --nodelist=worker-2 env G1_CODE_DIR=/mnt/data/vmo-ai-task/anhdh35/g1-future-worker2-20261019/production/source-1fcc2af G1_CUDA_HOME=/mnt/data/vmo-ai-task/anhdh35/cuda-12.9.1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 bash /mnt/data/vmo-ai-task/anhdh35/g1-future-worker2-20261019/production/source-1fcc2af/train/g1_omega_env.sh python -m scripts.vln.queue_g3a1_then_g1_future --config /mnt/data/vmo-ai-task/anhdh35/g3a1-then-g1-future-20261019/queue.json

