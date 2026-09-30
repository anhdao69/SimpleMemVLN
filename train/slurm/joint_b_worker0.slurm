#!/usr/bin/env bash
#SBATCH --job-name=smv-joint15-full-b-e2-w0
#SBATCH --partition=main
#SBATCH --nodelist=worker-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/outputs/joint15_worker0_b_bs8_e2_20260930/slurm-%j.log
set -euo pipefail

# Reuse the exact smoke-tested training code, not the mutable repository checkout.
reference=/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/outputs/joint15_b_bs8_e2_20260929
campaign=/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/outputs/joint15_worker0_b_bs8_e2_20260930
root="$reference/source"
run="$campaign/train"
[[ "${SLURMD_NODENAME:?}" == worker-0 ]]
cd "$root"
export PYTHONPATH="$root/src:$root"
export ENV_DIR=/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/.venv
export CUDA_HOME=/mnt/data/vmo-ai-task/anhdh35/cuda-12.8.1
export PATH="$CUDA_HOME/bin:$ENV_DIR/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
TMPDIR=$(mktemp -d "/tmp/smv-joint-w0-${SLURM_JOB_ID:?}-XXXXXX")
export TMPDIR
model=/mnt/data/vmo-ai-task/anhdh35/cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
manifest=/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl

test -f "$reference/smoke/MEMORY_GATE_PASS.json"
sha256sum --status -c "$reference/smoke/manifest.sha256"
sha256sum --status -c "$reference/smoke/source.sha256"
python - "$reference" "$campaign" <<'PY'
import json, os, shutil, sys
from pathlib import Path
import torch
from qwen_vl.train.campaign import check_memory_profile

reference, campaign = map(Path, sys.argv[1:])
assert shutil.disk_usage(campaign).free > 250 * 2**30, 'Insufficient free disk'
assert torch.cuda.device_count() == 4, 'Four visible GPUs required'
assert all('H100' in torch.cuda.get_device_name(i) for i in range(4))
print('MEMORY_GATE', check_memory_profile(reference/'smoke'), flush=True)
provenance = json.loads((reference/'provenance.json').read_text())
provenance.update(
    slurm_job_id=int(os.environ['SLURM_JOB_ID']),
    slurm_node=os.environ['SLURMD_NODENAME'],
    slurm_requested_time_limit_hours=72,
    slurm_time_limit_note='worker-0 Slurm template requests 72 hours',
    smoke_reference=str(reference/'smoke'),
    output_campaign=str(campaign),
)
with (campaign/'provenance.json').open('x') as f:
    json.dump(provenance, f, indent=2)
    f.write('\n')
PY
# Fail closed on an existing output rather than overwriting/resuming implicitly.
mkdir "$run"
nvidia-smi --id="${CUDA_VISIBLE_DEVICES:?}" --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,memory.total --format=csv -l 5 > "$run/gpu_utilization.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
sleep 1
kill -0 "$monitor"
torchrun --standalone --nproc_per_node=4 -m qwen_vl.train.train_qwen \
    --vln_config configs/vln_r2r_v0_base.yaml \
    --output_config configs/vln_joint_b_2epoch_bs8.yaml \
    --model_name_or_path "$model" --manifest "$manifest" \
    --output_dir "$run" --run-name joint15-fullcontext-b-bs8-e2-worker0 \
    > "$run/train.log" 2>&1
python -m scripts.vln.summarize_run "$run" > "$run/summary.log"
