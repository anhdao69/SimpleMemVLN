#!/usr/bin/env bash
#SBATCH --job-name=smv-joint15-win8-text-2gpu
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --nodelist=worker-2
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=768G
#SBATCH --time=72:00:00
# Submit with --dependency=afterany:<interactive-job-id> and an explicit log path.
set -euo pipefail
campaign=${1:?Pass the immutable two-GPU campaign directory}
cd "$campaign/source"
export PYTHONPATH="$PWD/src:$PWD"
export ENV_DIR=/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/.venv
export CUDA_HOME=/mnt/data/vmo-ai-task/anhdh35/cuda-12.8.1
export PATH="$CUDA_HOME/bin:$ENV_DIR/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
TMPDIR=$(mktemp -d "/tmp/smv-text2-${SLURM_JOB_ID:?}-XXXXXX")
export TMPDIR
export TRITON_CACHE_DIR="$TMPDIR/triton"
exec 9>"$campaign/training.lock"
flock -n 9 || { echo 'Another process owns this campaign'; exit 1; }
sha256sum --status -c "$campaign/source.sha256"
sha256sum --status -c "$campaign/manifest.sha256"
sha256sum --status -c "$campaign/runtime.sha256"
python - "$campaign" <<'PY'
import json, sys, shutil, torch
from pathlib import Path
from qwen_vl.train.vln_runtime import resolve_config
from qwen_vl.contracts import validate_config
p = Path(sys.argv[1])
assert shutil.disk_usage(p).free > 450 * 2**30, 'Need 450 GiB checkpoint headroom'
assert torch.cuda.device_count() == 2
assert all('H100' in torch.cuda.get_device_name(i) for i in range(2))
cfg = resolve_config('configs/vln_r2r_v0_base.yaml',
                     'configs/vln_joint_b_2epoch_bs8.yaml',
                     'configs/vln_memory_window8.yaml')
cfg['training']['gradient_accumulation_steps'] = 4
validate_config(cfg, 2)
assert cfg['model']['output_mode'] == 'qwen_text'
assert cfg['memory']['mode'] == 'window8'
t = cfg['training']
assert (t['epochs'], t['nominal_episodes_per_update'], t['backbone_lr']) == (2, 8, 5e-6)
assert (t['expected_total_steps'], t['warmup_steps']) == (7704, 232)
assert t['dataloader_num_workers'] == 4
assert not (p / 'train' / 'final').exists(), 'Campaign already finished'
print('TWO_GPU_TEXT_RECIPE_PASS: 7704 updates / 232 warmup', flush=True)
PY
# Test under the actual 768 GiB allocation. Never reuse a previous PASS.
profile=$(mktemp -d "$campaign/profile-startup-${SLURM_JOB_ID}-XXXXXX")
torchrun --standalone --nproc_per_node=2 -m qwen_vl.train.train_qwen \
    --vln_config configs/vln_r2r_v0_base.yaml \
    --output_config configs/vln_joint_b_2epoch_bs8.yaml \
    --memory_config configs/vln_memory_window8.yaml \
    --gradient_accumulation_steps 4 --deepspeed "$campaign/deepspeed.json" \
    --model_name_or_path /mnt/data/vmo-ai-task/anhdh35/cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
    --manifest "$campaign/worst_manifest.jsonl" \
    --max-optimizer-updates 3 --debug-repeat-episodes --profile-only \
    --output_dir "$profile" --run-name window8-text-2gpu-startup-memory \
    > "$profile/train.log" 2>&1
python "$campaign/verify_two_gpu_profile.py" "$profile" "$campaign/worst_manifest.jsonl"
# The smoke process has exited. Production starts fresh from pretrained weights.
run="$campaign/train"
mkdir -p "$run"
nvidia-smi --id="${CUDA_VISIBLE_DEVICES:?}" --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,memory.total --format=csv -l 5 >> "$run/gpu_utilization.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
sleep 1
kill -0 "$monitor"
torchrun --standalone --nproc_per_node=2 -m qwen_vl.train.train_qwen \
    --vln_config configs/vln_r2r_v0_base.yaml \
    --output_config configs/vln_joint_b_2epoch_bs8.yaml \
    --memory_config configs/vln_memory_window8.yaml \
    --gradient_accumulation_steps 4 --deepspeed "$campaign/deepspeed.json" \
    --model_name_or_path /mnt/data/vmo-ai-task/anhdh35/cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
    --manifest /mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl \
    --output_dir "$run" --run-name joint15-window8-text-2gpu \
    --recovery-save-steps 100 --resume_from_checkpoint auto \
    >> "$run/train.log" 2>&1
python -m scripts.vln.summarize_run "$run" > "$run/summary.log"
