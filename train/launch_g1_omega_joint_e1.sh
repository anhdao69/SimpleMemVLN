#!/usr/bin/env bash
# Run inside the user's four-GPU Slurm allocation. Same RUN_DIR resumes safely.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside a four-H100 Slurm allocation}"
G1_RUN_DIR="${1:?Usage: launch_g1_omega_joint_e1.sh /absolute/run/directory}"
[[ "$G1_RUN_DIR" = /* ]] || { echo 'RUN_DIR must be absolute' >&2; exit 2; }
G1_ROOT=/mnt/data/vmo-ai-task/anhdh35
export G1_CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export G1_CUDA_HOME="$G1_ROOT/cuda-12.9.1" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MAX_JOBS=2
mkdir -p "$G1_RUN_DIR"
# Keep supervisor logs from every launch; never overwrite an earlier attempt.
G1_SUPERVISOR="$G1_RUN_DIR/supervisor-$(date -u +%Y%m%dT%H%M%S)-$$"
exec flock -n "$G1_RUN_DIR/launcher.lock" \
  bash "$G1_CODE_DIR/train/g1_omega_env.sh" \
  python -m scripts.vln.run_episodic_gate \
  --out "$G1_SUPERVISOR" --host-guard-gib 691.2 \
  --disk-path "$G1_ROOT" --min-free-disk-gib 220 -- \
  torchrun --standalone --nproc_per_node=4 -m qwen_vl.train.train_qwen \
  --vln_config configs/episodic/science_common.yaml \
  --output_config configs/vln_empty.yaml \
  --memory_config configs/vln_memory_window8.yaml \
  --policy_config configs/vggt/g1_joint_one_epoch.yaml \
  --runtime_profile configs/vggt/g1_omega_h4.yaml \
  --deepspeed configs/deepspeed_zero2_cpu_optimizer.json \
  --manifest "$G1_ROOT/SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl" \
  --model_name_or_path "$G1_ROOT/cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a" \
  --output_dir "$G1_RUN_DIR/train" --run-name g1_omega_h4_joint_e1 \
  --recovery-save-steps 100 --first-recovery-save-step 10 \
  --resume_from_checkpoint auto
