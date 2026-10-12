#!/usr/bin/env bash
set -euo pipefail
R=/mnt/data/vmo-ai-task/anhdh35
E="$R/c2-speed-20261011"
while ! grep -q REPRESENTATIVE_DONE "$R/c2-speed-representative.log"; do
 if grep -q -E 'Traceback|SUITE_FAIL|Exited with exit code' "$R/c2-speed-representative.log"; then exit 1; fi
 sleep 10
done
OUT="$E/qualification-e789a13"
mkdir -p "$OUT"
python -m pytest tests/vln -q > "$OUT/pytest.log" 2>&1
python -m scripts.vln.check_episodic --runtime-profile configs/episodic/profile_h2.yaml --model-path "$VLN_MODEL_PATH" --manifest "$R/SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl" --out "$OUT/numerical.json" --arms v0 m1 m2 --replay-steps 64 96 128 > "$OUT/numerical.log" 2>&1
python -m scripts.vln.profile_episodic suite --profile H2 --fixtures "$R/c2-audit-evidence-20261011/fixtures" --out "$OUT/resources" --model-path "$VLN_MODEL_PATH" --gates smoke longest_r2r below_offload above_offload longest_rxr mixed_tail checkpoint --arms v0 m2 --updates 3 --host-guard 460.8 > "$OUT/resources.log" 2>&1
python -m scripts.vln.profile_episodic suite --profile H2 --fixtures "$R/c2-audit-evidence-20261011/fixtures" --out "$OUT/m1" --model-path "$VLN_MODEL_PATH" --gates below_offload longest_rxr --arms m1 --updates 3 --host-guard 460.8 > "$OUT/m1.log" 2>&1
printf 'FINAL_QUALIFICATION_DONE\n'
