#!/usr/bin/env bash
set -euo pipefail
B=/mnt/data/vmo-ai-task/anhdh35
G='/mnt/data/vmo-ai-task/anhdh35/g1-future-worker2-20261019'
R='/mnt/data/vmo-ai-task/anhdh35/g3a1-resume-worker2-20261019'
torchrun --standalone --nproc_per_node=4 -m scripts.vln.check_vggt_reduction > "$G/logs/distributed-reduction.log" 2>&1
python -m scripts.vln.profile_vggt_g1 --root "$G" --name mixed-extreme --fixture "$B/g1-omega-20261013/fixtures/mixed_extreme.jsonl" --updates 3 --threads 4 --workers 8 --attention-batch 32 --vision-batch 8 --teacher-batch 8 --cpu-adam --future-offset 1 > "$G/logs/mixed-extreme.log" 2>&1
VLN_TEST_FUTURE_OMEGA=1 python -m pytest tests/vln -q -rs -p no:cacheprovider --junitxml="$G/final-suite.xml" > "$G/logs/final-suite.log" 2>&1
G1_CODE_DIR="$R/source/SimpleMemVLN-g3a1-877815e" PYTHONDONTWRITEBYTECODE=1 bash "$R/source/SimpleMemVLN-g3a1-877815e/train/g1_omega_env.sh" python -m pytest tests/vln -q -rs -p no:cacheprovider --junitxml="$R/gpu-suite.xml" > "$R/gpu-suite.log" 2>&1
printf '%s\n' 'ALL_FINAL_CHECKS_PASS' > "$G/final-checks.completed"
