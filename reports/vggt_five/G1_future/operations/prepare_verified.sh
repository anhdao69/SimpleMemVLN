#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
cd '/mnt/data/vmo-ai-task/anhdh35/g3a1-resume-worker2-20261019/download'
'/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/.venv/bin/python' tools/prepare_resume.py --bundle . --src '/mnt/data/vmo-ai-task/anhdh35/g3a1-resume-worker2-20261019/source/SimpleMemVLN-g3a1-877815e' --run-dir '/mnt/data/vmo-ai-task/anhdh35/g3a1-resume-worker2-20261019/production' --image-root '/mnt/data/vmo-ai-task/anhdh35/g3a1-resume-worker2-20261019/image-alias/trajectory_data//groups/yshang/an221229/data/JanusVLN_data' --teacher-source '/mnt/data/vmo-ai-task/anhdh35/third_party/vggt-omega' --teacher-weights '/mnt/data/vmo-ai-task/anhdh35/models/vggt-omega/vggt_omega_1b_512.pt' --model '/mnt/data/vmo-ai-task/anhdh35/cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a' --dataloader-num-workers 8 --check-images --copy-mode hardlink
printf '%s\n' 'PREPARATION_COMPLETE' > '/mnt/data/vmo-ai-task/anhdh35/g3a1-resume-worker2-20261019/prepare.completed'
