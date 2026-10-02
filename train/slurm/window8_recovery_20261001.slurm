#!/usr/bin/env bash
#SBATCH --job-name=smv-joint15-win8-recovery
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=768G
#SBATCH --time=72:00:00
#SBATCH --output=/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/outputs/window8_recovery_20261001/slurm-%j.log
set -euo pipefail
campaign=/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/outputs/window8_recovery_20261001
cd "$campaign/source"
export PYTHONPATH="$PWD/src:$PWD"
export ENV_DIR=/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/.venv
export CUDA_HOME=/mnt/data/vmo-ai-task/anhdh35/cuda-12.8.1
export PATH="$CUDA_HOME/bin:$ENV_DIR/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
TMPDIR=$(mktemp -d "/tmp/smv-recovery-${SLURM_JOB_ID:?}-XXXXXX")
export TMPDIR
export TRITON_CACHE_DIR="$TMPDIR/triton"
exec 9>"$campaign/training.lock"
flock -n 9 || { echo 'Another training process owns this output'; exit 1; }
sha256sum --status -c "$campaign/source.sha256"
sha256sum --status -c "$campaign/manifest.sha256"
python - "$campaign" <<'PY'
import json,sys,shutil,torch
from pathlib import Path
p=Path(sys.argv[1])
assert shutil.disk_usage(p).free > 450*2**30, 'Need 450 GiB checkpoint headroom'
assert torch.cuda.device_count()==4
assert all('H100' in torch.cuda.get_device_name(i) for i in range(4))
for case in ('failed','worst'):
    gate=json.loads((p/f'profile-{case}'/'PASS.json').read_text())
    assert gate['passed'] and gate['updates']==3 and gate['peak_reserved_gib']<=78
for phase in ('baseline','interrupted','resume'):
    assert json.loads((p/'recovery_test'/f'{phase}-PASS.json').read_text())['passed']
assert not (p/'train'/'final').exists(), 'This campaign already has a final export'
print('VALIDATION_GATES_PASS: input staging, worst batch, distributed recovery',flush=True)
PY
run="$campaign/train"
mkdir -p "$run"
nvidia-smi --id="${CUDA_VISIBLE_DEVICES:?}" --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,memory.total --format=csv -l 5 >> "$run/gpu_utilization.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
sleep 1
kill -0 "$monitor"
torchrun --standalone --nproc_per_node=4 -m qwen_vl.train.train_qwen \
    --vln_config configs/vln_r2r_v0_base.yaml \
    --output_config configs/vln_joint_b_2epoch_bs8.yaml \
    --memory_config configs/vln_memory_window8.yaml \
    --model_name_or_path /mnt/data/vmo-ai-task/anhdh35/cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
    --manifest /mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl \
    --output_dir "$run" --run-name joint15-window8-recovery \
    --recovery-save-steps 100 --resume_from_checkpoint auto \
    >> "$run/train.log" 2>&1
python -m scripts.vln.summarize_run "$run" > "$run/summary.log"
