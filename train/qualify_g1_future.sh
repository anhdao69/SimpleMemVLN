#!/usr/bin/env bash
# Disposable qualification only; stops at the first failed gate.
set -euo pipefail
R="${1:?campaign root}"
B=/mnt/data/vmo-ai-task/anhdh35
F="$B/g1-omega-20261013/fixtures"
[[ "${SLURMD_NODENAME:-$(hostname)}" = worker-2 ]]
run() {
  local name="$1" fixture="$2" updates="$3" batch="$4"
  shift 4
  python -m scripts.vln.profile_vggt_g1 --root "$R" --name "$name" \
    --fixture "$fixture" --updates "$updates" --threads 4 --workers 8 \
    --attention-batch 32 --vision-batch 8 --teacher-batch "$batch" \
    --cpu-adam --future-offset 1 "$@" > "$R/logs/$name.log" 2>&1
}
VLN_TEST_FUTURE_OMEGA=1 python -m pytest tests/vln/test_vggt_future.py tests/vln/test_vggt_future_cuda.py tests/vln/test_vggt_g1_forward.py -q -s > "$R/logs/future-focus.log" 2>&1
run mixed-b8 "$F/mixed_tail.jsonl" 3 8
run mixed-b4 "$F/mixed_tail.jsonl" 3 4
run terminal "$R/fixtures/terminal.jsonl" 2 8
run r2r-representative "$F/r2rce_representative.jsonl" 8 8
run rxr-representative "$F/rxrce_representative.jsonl" 8 8
run r2r-longest "$F/r2rce_longest.jsonl" 3 8
run rxr-longest "$F/rxrce_longest.jsonl" 3 8
run recovery-reference "$F/mixed_tail.jsonl" 3 8 --save
run recovery-resumed "$F/mixed_tail.jsonl" 3 8 --save --first-update 3 \
  --resume "$R/experiments/recovery-reference/train/checkpoint-2"
python -m scripts.vln.check_vggt_recovery \
  --reference "$R/experiments/recovery-reference/train" \
  --resumed "$R/experiments/recovery-resumed/train" --out "$R/recovery_check.json" > "$R/logs/recovery-check.log" 2>&1
python -m scripts.vln.check_vggt_export \
  --source "$R/experiments/recovery-reference/train/final" --output "$R/serving-export" \
  --manifest "$B/SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl" \
  --model-path "$VLN_MODEL_PATH" > "$R/logs/export-check.log" 2>&1
date -u > "$R/qualification-completed.txt"
