#!/usr/bin/env bash
set -euo pipefail
campaign=/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/outputs/joint15_b_bs8_e2_20260929
root="$campaign/source"
cd "$root"
export PYTHONPATH="$root/src:$root"
export ENV_DIR=/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/.venv
export CUDA_HOME=/mnt/data/vmo-ai-task/anhdh35/cuda-12.8.1
export PATH="$CUDA_HOME/bin:$ENV_DIR/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
# Multiprocessing sockets/tempfiles must not live on NFS (EBUSY cleanup).
TMPDIR=$(mktemp -d "/tmp/smv-joint-${SLURM_JOB_ID:?}-XXXXXX")
export TMPDIR
model=/mnt/data/vmo-ai-task/anhdh35/cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
export VLN_MODEL_PATH="$model"
manifest=/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl
common=(--vln_config configs/vln_r2r_v0_base.yaml --output_config configs/vln_joint_b_2epoch_bs8.yaml --model_name_or_path "$model" --manifest "$manifest")
mode="${1:?smoke or train}"
[[ "$mode" == smoke || "$mode" == train ]]
run="$campaign/$mode"
mkdir "$run"
df -h "$campaign"
nvidia-smi --query-gpu=name,memory.total --format=csv
python - "$campaign" <<'PY'
import shutil,sys,torch
assert shutil.disk_usage(sys.argv[1]).free > 250 * 2**30, 'Insufficient free disk'
assert torch.cuda.device_count() == 4
assert all('H100' in torch.cuda.get_device_name(i) for i in range(4))
PY
nvidia-smi --id="${CUDA_VISIBLE_DEVICES:?}" --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,memory.total --format=csv -l 5 > "$run/gpu_utilization.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
if [[ "$mode" == smoke ]]; then
    sha256sum "$manifest" > "$run/manifest.sha256"
    find src scripts configs train -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' -o -name '*.slurm' \) -print0 | sort -z | xargs -0 sha256sum > "$run/source.sha256"
    python -m pytest > "$campaign/pytest.log" 2>&1
    torchrun --standalone --nproc_per_node=4 -m scripts.vln.profile_episode "${common[@]}" --out "$run" > "$run/train.log" 2>&1
    python - "$run" <<'PY'
import json,sys
from pathlib import Path
from qwen_vl.train.campaign import check_memory_profile
p=Path(sys.argv[1]); result=check_memory_profile(p)
(p/'MEMORY_GATE_PASS.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
PY
else
    test -f "$campaign/smoke/MEMORY_GATE_PASS.json"
    sha256sum --status -c "$campaign/smoke/manifest.sha256"
    sha256sum --status -c "$campaign/smoke/source.sha256"
    python - "$campaign/smoke" <<'PY'
import sys
from qwen_vl.train.campaign import check_memory_profile
print(check_memory_profile(sys.argv[1]))
PY
    torchrun --standalone --nproc_per_node=4 -m qwen_vl.train.train_qwen "${common[@]}" --output_dir "$run" --run-name joint-fullcontext-b-bs8-e2 > "$run/train.log" 2>&1
    python -m scripts.vln.summarize_run "$run" > "$run/summary.log"
fi
