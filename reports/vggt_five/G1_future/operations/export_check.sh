#!/usr/bin/env bash
set -euo pipefail
G='/mnt/data/vmo-ai-task/anhdh35/g1-future-worker2-20261019'
[[ -f "$G/final-checks.completed" ]]
python -m pytest tests/vln/test_g1_export_greedy_audit.py -q -p no:cacheprovider --junitxml="$G/export-unit-suite.xml" > "$G/logs/export-unit.log" 2>&1
python -m scripts.vln.check_vggt_export --source "$G/experiments/recovery-reference/train/final" --output "$G/serving-export-smoke" --manifest /mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl --model-path "$VLN_MODEL_PATH" --allow-untrained-greedy-error > "$G/logs/export-check-smoke.log" 2>&1
date -u > "$G/export-check.completed"

