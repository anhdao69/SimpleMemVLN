#!/usr/bin/env bash
# G3A1 one-epoch joint run on newton (4 x H100). Independent start from the original
# Qwen3.5-4B base. Reusing the same RUN_DIR resumes complete full-state recovery
# (same 4-GPU topology only). Run from the immutable source snapshot RUN_DIR/source.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside a four-H100 Slurm allocation}"
RUN_DIR="${1:?Absolute run directory}"; [[ "$RUN_DIR" = /* ]] || { echo 'RUN_DIR must be absolute' >&2; exit 2; }
HOST_GUARD_GIB="${HOST_GUARD_GIB:-460.8}"   # 0.9 x 512 GiB allocation
export G3A1_CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS:-1}}"
[[ "$(nvidia-smi -L | wc -l)" == 4 ]] || { echo 'Four visible H100s required' >&2; exit 2; }
MANIFEST=/groups/yshang/an221229/checkpoints/SimpleMemVLN-c2_episodic/artifacts/r2r_rxr15_train.jsonl
MODEL=/groups/yshang/an221229/cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
mkdir -p "$RUN_DIR"
SUPERVISOR="$RUN_DIR/supervisor-$(date -u +%Y%m%dT%H%M%S)-$$"
exec flock -n "$RUN_DIR/launcher.lock" \
 bash "$G3A1_CODE_DIR/train/g3a1_newton_env.sh" python -m scripts.vln.run_episodic_gate \
 --out "$SUPERVISOR" --host-guard-gib "$HOST_GUARD_GIB" --disk-path "$RUN_DIR" --min-free-disk-gib 220 -- \
 torchrun --standalone --nproc_per_node=4 -m qwen_vl.train.train_qwen \
 --vln_config configs/episodic/science_common.yaml --output_config configs/vln_empty.yaml \
 --memory_config configs/vln_memory_window8.yaml --policy_config configs/vggt/g3a1_joint_one_epoch.yaml \
 --runtime_profile configs/vggt/g3a1_newton_h4.yaml --deepspeed configs/deepspeed_zero2_cpu_optimizer.json \
 --manifest "$MANIFEST" --model_name_or_path "$MODEL" \
 --output_dir "$RUN_DIR/train" --run-name g3a1_omega_newton_h4_joint_e1 \
 --recovery-save-steps 100 --first-recovery-save-step 10 --resume_from_checkpoint auto
