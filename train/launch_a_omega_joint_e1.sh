#!/usr/bin/env bash
# Independent A0/A1 base starts. Reusing the same RUN_DIR resumes full state.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside a four-H100 Slurm allocation}"
A_ARM="${1:?A0 or A1}"; A_RUN_DIR="${2:?Absolute run directory}"
case "$A_ARM" in A0) A_PROFILE=a0_h4;; A1) A_PROFILE=a1_h4;; *) exit 2;; esac
[[ "$A_RUN_DIR" = /* ]] || { echo 'RUN_DIR must be absolute' >&2; exit 2; }
A_QUALIFICATION="${3:?Qualification seal required}"; A_PREDECESSOR="${4:?Successful predecessor required}"
A_ROOT=/mnt/data/vmo-ai-task/anhdh35
export G1_CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export G1_CUDA_HOME="$A_ROOT/cuda-12.9.1" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MAX_JOBS=2
[[ "${SLURMD_NODENAME:-}" == worker-2 ]] || { echo 'Worker-2 required' >&2; exit 2; }
bash "$G1_CODE_DIR/train/g1_omega_env.sh" python -m scripts.vln.queue_geoaddress \
 --check-only --source "$G1_CODE_DIR" --qualification "$A_QUALIFICATION" --predecessor "$A_PREDECESSOR"
mkdir -p "$A_RUN_DIR"
A_SUPERVISOR="$A_RUN_DIR/supervisor-$(date -u +%Y%m%dT%H%M%S)-$$"
exec flock -n "$A_RUN_DIR/launcher.lock" \
 bash "$G1_CODE_DIR/train/g1_omega_env.sh" python -m scripts.vln.run_episodic_gate \
 --out "$A_SUPERVISOR" --host-guard-gib 691.2 --disk-path "$A_ROOT" --min-free-disk-gib 220 -- \
 torchrun --standalone --nproc_per_node=4 -m qwen_vl.train.train_qwen \
 --vln_config configs/episodic/science_common.yaml --output_config configs/vln_empty.yaml \
 --memory_config configs/vln_memory_window8.yaml --policy_config configs/vggt/a_joint_one_epoch.yaml \
 --runtime_profile "configs/vggt/$A_PROFILE.yaml" --deepspeed configs/deepspeed_zero2_cpu_optimizer.json \
 --manifest "$A_ROOT/SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl" \
 --model_name_or_path "$A_ROOT/cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a" \
 --output_dir "$A_RUN_DIR/train" --run-name "${A_ARM,,}_omega_h4_joint_e1" \
 --recovery-save-steps 100 --first-recovery-save-step 10 --resume_from_checkpoint auto
