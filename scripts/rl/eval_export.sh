#!/usr/bin/env bash
# Evaluate an RL export with exactly the baseline protocol of
# /data/anh/simplemem-eval/run_eval.sh (r2r full): same evaluator, Habitat config,
# dataset, scenes, base model, seed, 500-action cap and 3600 s startup allowance.
# Usage: eval_export.sh CHECKPOINT_DIR PHYSICAL_GPU OUT_DIR [extra evaluator args...]
set -euo pipefail
source /data/anh/simplemem-eval/env.sh
# Model server code of this checkout (M5 episodic serving lives on RL_m5); evaluator unchanged.
SMV_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export SMV_ROOT SIMULATOR_SOURCE="$SMV_ROOT/docs/evaluation_setup" PYTHONPATH="$SMV_ROOT/src:$SMV_ROOT/docs/evaluation_setup"
checkpoint=${1:?checkpoint}; gpu=${2:?physical gpu}; out=${3:?out dir}; shift 3
cd "$SMV_ROOT"
# Habitat resolves gpu_device_id as a CUDA ordinal of the masked process, so the
# single visible device is ordinal 0 (verified on this host; see plans/RL report).
export CUDA_VISIBLE_DEVICES="$gpu"
exec "$EVAL_PYTHON" -u -m qwen_vl.eval.habitat_r2r \
  --dataset-kind r2r --checkpoint "$checkpoint" \
  --model-python "$MODEL_PYTHON" --model-path "$MODEL_PATH" \
  --habitat-config "$R2R_CONFIG" --data-path "$R2R_DATA" \
  --scenes-dir "$MP3D_SCENES" --gpu 0 --split val_unseen \
  --model-startup-timeout 3600 "$@" --out "$out"
