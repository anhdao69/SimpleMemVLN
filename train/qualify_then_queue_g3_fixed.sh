#!/usr/bin/env bash
# Owned GPU step: test constant-weight G3, then resume G3A1 and queue G3.
set -euo pipefail
B=/mnt/data/vmo-ai-task/anhdh35
G="$B/g3-fixed01-seed430-worker2-20261019"
R="$B/g3a1-resume-worker2-20261019"
export SRC="$R/source/SimpleMemVLN-g3a1-877815e" RUN_DIR="$R/production" MODEL="$VLN_MODEL_PATH"
fallback() {
  status=$?
  if [[ "$status" != 0 ]]; then
    echo "Qualification/queue failed ($status); resume G3A1 only; new G3 remains blocked" >&2
    exec bash "$G1_CODE_DIR/train/resume_g3a1_worker2.sh"
  fi
}
trap fallback EXIT
python -m pytest tests/vln -q -rs -p no:cacheprovider --junitxml="$G/gpu-suite.xml" > "$G/logs/gpu-suite.log" 2>&1
python -m scripts.vln.run_episodic_gate --out "$G/smoke/supervisor" --host-guard-gib 691.2 --disk-path "$G" --min-free-disk-gib 220 -- \
 torchrun --standalone --nproc_per_node=4 -m qwen_vl.train.train_qwen \
 --vln_config configs/episodic/science_common.yaml --output_config configs/vln_empty.yaml \
 --memory_config configs/vln_memory_window8.yaml --policy_config configs/vggt/g3_joint_one_epoch.yaml \
 --runtime_profile configs/vggt/g3_fixed01_seed430_h4.yaml --deepspeed configs/deepspeed_zero2_cpu_optimizer.json \
 --manifest "$B/g1-omega-20261013/fixtures/mixed_extreme.jsonl" --model_name_or_path "$MODEL" \
 --output_dir "$G/smoke/train" --run-name g3_fixed01_seed430_smoke \
 --max-optimizer-updates 2 --debug-repeat-episodes --recovery-save-steps 2 > "$G/logs/smoke.log" 2>&1
python reports/vggt_five/G3_fixed01_seed430/operations/prepare_queue.py > "$G/logs/prepare-queue.log" 2>&1
trap - EXIT
python -m scripts.vln.queue_g3a1_then_g3_fixed --config "$G/queue/queue.json" > "$G/queue/controller.log" 2>&1
