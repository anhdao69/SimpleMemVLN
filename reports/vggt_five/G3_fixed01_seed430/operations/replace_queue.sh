#!/usr/bin/env bash
set -euo pipefail
G=/mnt/data/vmo-ai-task/anhdh35/g3-fixed01-seed430-worker2-20261019
R=/mnt/data/vmo-ai-task/anhdh35/g3a1-resume-worker2-20261019
Q=/mnt/data/vmo-ai-task/anhdh35/g3a1-then-g1-future-20261019
# Wait for the existing controller to finish its graceful checkpoint stop.
exec 9>"$Q/queue.lock"
flock -x 9
[[ -f "$R/production/train/REQUEST_CHECKPOINT_AND_STOP" ]]
mv "$R/production/train/REQUEST_CHECKPOINT_AND_STOP" "$R/production/train/REQUEST_CHECKPOINT_AND_STOP.$(date -u +%Y%m%dT%H%M%S).handled"
exec 9>&-
exec srun --jobid=4925 --overlap --ntasks=1 --gpus=4 --cpus-per-task=60 --nodelist=worker-2 --job-name=g3a1_then_g3_fixed env G1_CODE_DIR="$G/source" G1_CUDA_HOME=/mnt/data/vmo-ai-task/anhdh35/cuda-12.9.1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MAX_JOBS=2 PYTHONDONTWRITEBYTECODE=1 bash "$G/source/train/g1_omega_env.sh" bash "$G/source/train/qualify_then_queue_g3_fixed.sh"

