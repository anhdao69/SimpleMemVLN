#!/usr/bin/env bash
# Run inside the user's four-GPU Slurm allocation. Same RUN_DIR resumes safely.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside a four-H100 Slurm allocation}"
G1_RUN_DIR="${1:?Usage: launch_m5.sh /absolute/run/directory}"
[[ "$G1_RUN_DIR" = /* ]] || { echo 'RUN_DIR must be absolute' >&2; exit 2; }
G1_ROOT=/mnt/data/vmo-ai-task/anhdh35
M5_ROOT="$G1_ROOT/m5-one-epoch-worker2-20261018"
export G1_CODE_DIR="$M5_ROOT/source-f4d1c2f"
export G1_CUDA_HOME="$G1_ROOT/cuda-12.9.1" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MAX_JOBS=2
[[ "${SLURMD_NODENAME:-$(hostname)}" = worker-2 ]] || exit 2
bash "$G1_CODE_DIR/train/g1_omega_env.sh" python - "$M5_ROOT/operations" "$G1_RUN_DIR" <<'CHECK'
import sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from controller import verify_qualification,contract,TRIPLE
from scripts.vln.queue_g3_layers import arm_complete
verify_qualification()
assert arm_complete(TRIPLE,'l8_l16_l24')
assert json.loads((Path(sys.argv[2])/'run_identity.json').read_text())==contract()
CHECK
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
  --memory_config configs/episodic/arm_m5.yaml \
  --policy_config configs/vggt/g3_joint_one_epoch.yaml \
  --runtime_profile "$M5_ROOT/operations/runtime.yaml" \
  --deepspeed configs/deepspeed_zero2_cpu_optimizer.json \
  --manifest "$G1_ROOT/SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl" \
  --model_name_or_path "$G1_ROOT/cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a" \
  --output_dir "$G1_RUN_DIR/train" --run-name "m5_h4_joint_one_epoch_matched_g3" \
  --recovery-save-steps 100 --first-recovery-save-step 10 \
  --resume_from_checkpoint auto
