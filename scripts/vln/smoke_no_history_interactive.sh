#!/usr/bin/env bash
# Run directly in an EXISTING allocated shell; never allocates Slurm resources.
set -euo pipefail
campaign=${1:?Pass a new smoke output directory}
expected_job=${2:?Pass the already-running interactive Slurm job ID}
test "${SLURM_JOB_ID:?Must run inside the existing allocation}" = "$expected_job"
test ! -e "$campaign"
mkdir -p "$campaign"
# The login environment inherits an NFS TMPDIR. Multiprocessing finalizers can
# hit EBUSY there; put process sockets and JIT kernel files on this node instead.
TMPDIR=$(mktemp -d "/tmp/smv-nohistory-${SLURM_JOB_ID}-XXXXXX")
export TMPDIR
export PYTORCH_KERNEL_CACHE_PATH="$TMPDIR/torch-kernels"
mkdir -p "$PYTORCH_KERNEL_CACHE_PATH"
export ENV_DIR=/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/.venv
export CUDA_HOME=/mnt/data/vmo-ai-task/anhdh35/cuda-12.8.1
export PATH="$CUDA_HOME/bin:$ENV_DIR/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export VLN_MODEL_PATH=/mnt/data/vmo-ai-task/anhdh35/cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
manifest=/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/outputs/candidate_joint_full_20261002-W6vveI/smoke_manifest.jsonl
python - "$campaign" "$manifest" <<'PY'
import collections, hashlib, json, os, shutil, socket, sys
from pathlib import Path
import torch, yaml
from qwen_vl.train.vln_runtime import resolve_config
from qwen_vl.contracts import validate_config
p, manifest = map(Path, sys.argv[1:])
assert torch.cuda.device_count() == 4, 'Must use exactly the four already allocated GPUs'
assert shutil.disk_usage(p).free > 20 * 2**30
cfg = resolve_config('configs/vln_r2r_v0_base.yaml', 'configs/vln_joint_b_2epoch_bs8.yaml',
                     policy='configs/vln_candidate_logits_no_history.yaml')
# Exercise nonzero optimizer updates immediately, not three warmup-only updates.
cfg['training']['warmup_steps'] = 0
validate_config(cfg, world_size=4)
assert cfg['memory']['mode'] == 'full_context'
episodes = [json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
assert len(episodes) == 8 and len({e['episode_uid'] for e in episodes}) == 8
(p / 'smoke_recipe.yaml').write_text(yaml.safe_dump(cfg))
record = dict(job=os.environ['SLURM_JOB_ID'], host=socket.gethostname(),
              tmpdir=os.environ['TMPDIR'], kernel_cache=os.environ['PYTORCH_KERNEL_CACHE_PATH'],
              cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
              gpu_names=[torch.cuda.get_device_name(i) for i in range(4)],
              manifest=str(manifest), manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
              dataset_counts=dict(collections.Counter(e['dataset'] for e in episodes)),
              actions=sum(len(e['steps']) for e in episodes),
              longest_episode_steps=max(len(e['steps']) for e in episodes))
(p / 'allocation.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
PY
python -m pytest -q > "$campaign/tests.log" 2>&1
nvidia-smi --id="${CUDA_VISIBLE_DEVICES:?}" --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,memory.total --format=csv -l 5 > "$campaign/gpu_utilization.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
torchrun --standalone --nproc_per_node=4 -m qwen_vl.train.train_qwen \
    --vln_config "$campaign/smoke_recipe.yaml" \
    --output_config configs/vln_candidate_logits_no_history.yaml \
    --model_name_or_path "$VLN_MODEL_PATH" --manifest "$manifest" \
    --output_dir "$campaign/train" --run-name candidate-fullcontext-no-history-smoke \
    --max-optimizer-updates 3 --debug-repeat-episodes --profile-only \
    > "$campaign/train.log" 2>&1
python -m scripts.vln.check_candidate_integration \
    --model-path "$VLN_MODEL_PATH" --manifest "$manifest" \
    --memory full_context --feedback none --out "$campaign/integration.json" \
    > "$campaign/integration.log" 2>&1
printf 'NO_HISTORY_SMOKE_COMMANDS_COMPLETED\n'
