#!/usr/bin/env bash
#SBATCH --job-name=smv-joint15-win8-logits
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --nodelist=worker-1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=768G
#SBATCH --time=72:00:00
# Submit with --dependency=afterok:4623. Independently initialized from pretrained.
set -euo pipefail
campaign=${1:?Pass the immutable campaign directory}
cd "$campaign/source"
export PYTHONPATH="$PWD/src:$PWD"
export ENV_DIR=/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/.venv
export CUDA_HOME=/mnt/data/vmo-ai-task/anhdh35/cuda-12.8.1
export PATH="$CUDA_HOME/bin:$ENV_DIR/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export VLN_MODEL_PATH=/mnt/data/vmo-ai-task/anhdh35/cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
manifest=/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl
TMPDIR=$(mktemp -d "/tmp/smv-win8-logits-${SLURM_JOB_ID:?}-XXXXXX")
export TMPDIR
export TRITON_CACHE_DIR="$TMPDIR/triton"
exec 9>"$campaign/training.lock"
flock -n 9 || { echo 'Another process owns this campaign'; exit 1; }
sha256sum --status -c "$campaign/source.sha256"
sha256sum --status -c "$campaign/manifest.sha256"
sha256sum --status -c "$campaign/runtime.sha256"
python - "$campaign" <<'PY'
import sys, shutil, torch
from pathlib import Path
from qwen_vl.train.vln_runtime import resolve_config
from qwen_vl.contracts import validate_config
p = Path(sys.argv[1])
assert shutil.disk_usage(p).free > 450 * 2**30, 'Need 450 GiB checkpoint headroom'
assert torch.cuda.device_count() == 4
assert all('H100' in torch.cuda.get_device_name(i) for i in range(4))
cfg = resolve_config('configs/vln_r2r_v0_base.yaml', 'configs/vln_joint_b_2epoch_bs8.yaml',
                     memory='configs/vln_memory_window8.yaml',
                     policy='configs/vln_r2r_v0_candidate_logits.yaml')
validate_config(cfg, 4)
assert cfg['memory']['mode'] == 'window8'
assert cfg['memory']['kv_window_steps_including_current'] == 8
assert cfg['runtime']['text_attention'] == 'simplememvln_step_attention'
assert cfg['model']['output_mode'] == 'candidate_logits'
assert cfg['model']['action_head_mode'] == 'lm_rows_trainable'
assert cfg['observations']['feedback_format'] == 'candidate_token'
t = cfg['training']
assert (t['epochs'], t['nominal_episodes_per_update'], t['gradient_accumulation_steps']) == (2, 8, 2)
assert (t['expected_total_steps'], t['expected_warmup_steps'], t['warmup_steps']) == (7704, 232, 232)
assert t['backbone_lr'] == 5e-6 and t['class_weighting'] == 'none'
assert t['dataloader_num_workers'] == 4
assert not (p / 'train' / 'final').exists(), 'Campaign already finished'
print('WINDOW8_CANDIDATE_RECIPE_PASS: 7704 updates / 232 warmup', flush=True)
PY
for phase in baseline interrupted resume; do
    torchrun --standalone --nproc_per_node=4 -m scripts.vln.check_recovery_saving \
        --root "$campaign/recovery_test" --phase "$phase" > "$campaign/recovery-$phase.log" 2>&1
done
python - "$campaign/recovery_test" <<'PY'
import json, sys
from pathlib import Path
for phase in ('baseline', 'interrupted', 'resume'):
    assert json.loads((Path(sys.argv[1]) / f'{phase}-PASS.json').read_text())['passed']
PY
# Fresh three-update mixed-length and all-longest profiles on this allocation.
for case in failed worst; do
    profile="$campaign/profile-${case}-${SLURM_JOB_ID}-$(date +%s)"
    torchrun --standalone --nproc_per_node=4 -m scripts.vln.profile_input_staging \
        --case "$case" --out "$profile" \
        --vln_config configs/vln_r2r_v0_base.yaml \
        --output_config configs/vln_joint_b_2epoch_bs8.yaml \
        --policy_config configs/vln_r2r_v0_candidate_logits.yaml \
        --memory_config configs/vln_memory_window8.yaml \
        --model_name_or_path "$VLN_MODEL_PATH" --manifest "$manifest" \
        > "$profile.log" 2>&1
    python -m scripts.vln.profile_input_staging --verify-only \
        --case "$case" --out "$profile" --manifest "$manifest" >> "$profile.log" 2>&1
done
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
    --policy_config configs/vln_r2r_v0_candidate_logits.yaml \
    --memory_config configs/vln_memory_window8.yaml \
    --model_name_or_path "$VLN_MODEL_PATH" --manifest "$manifest" \
    --output_dir "$run" --run-name joint15-window8-candidate \
    --recovery-save-steps 100 --resume_from_checkpoint auto >> "$run/train.log" 2>&1
python -m scripts.vln.summarize_run "$run" > "$run/summary.log"
