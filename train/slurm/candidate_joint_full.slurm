#!/usr/bin/env bash
#SBATCH --job-name=smv-joint15-full-logits
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=768G
#SBATCH --time=72:00:00
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
TMPDIR=$(mktemp -d "/tmp/smv-logits-${SLURM_JOB_ID:?}-XXXXXX")
export TMPDIR
export TRITON_CACHE_DIR="$TMPDIR/triton"
exec 9>"$campaign/training.lock"
flock -n 9 || { echo 'Another process owns this campaign'; exit 1; }
sha256sum --status -c "$campaign/source.sha256"
sha256sum --status -c "$campaign/manifest.sha256"
python - "$campaign" <<'PY'
import json, sys, shutil, torch
from pathlib import Path
from qwen_vl.train.vln_runtime import resolve_config
p = Path(sys.argv[1])
assert shutil.disk_usage(p).free > 450 * 2**30, 'Need 450 GiB checkpoint headroom'
assert torch.cuda.device_count() == 4
assert all('H100' in torch.cuda.get_device_name(i) for i in range(4))
gate = json.loads((p / 'interactive-PASS.json').read_text())
assert gate['passed'] and gate['updates'] == 3 and gate['peak_reserved_gib'] <= 78
cfg = resolve_config('configs/vln_r2r_v0_base.yaml', 'configs/vln_joint_b_2epoch_bs8.yaml',
                     policy='configs/vln_r2r_v0_candidate_logits.yaml')
assert cfg['memory']['mode'] == 'full_context'
assert cfg['model']['output_mode'] == 'candidate_logits'
assert cfg['model']['action_head_mode'] == 'lm_rows_trainable'
assert cfg['observations']['feedback_format'] == 'candidate_token'
t = cfg['training']
assert (t['epochs'], t['nominal_episodes_per_update'], t['gradient_accumulation_steps']) == (2, 8, 2)
assert (t['expected_total_steps'], t['expected_warmup_steps'], t['warmup_steps']) == (7704, 232, 232)
assert t['backbone_lr'] == 5e-6 and t['class_weighting'] == 'none'
assert t['dataloader_num_workers'] == 4
assert not (p / 'train' / 'final').exists(), 'Campaign already finished'
print('RECIPE_AND_INTERACTIVE_GATE_PASS', flush=True)
PY

# Revalidate recovery and the failed/worst production batches on the actual
# four-GPU allocation. No production updates occur unless all gates pass.
for phase in baseline interrupted resume; do
    if [ ! -f "$campaign/recovery_test/$phase-PASS.json" ]; then
        torchrun --standalone --nproc_per_node=4 -m scripts.vln.check_recovery_saving \
            --root "$campaign/recovery_test" --phase "$phase" > "$campaign/recovery-$phase.log" 2>&1
    fi
done
python - "$campaign/recovery_test" <<'PY'
import json, sys
from pathlib import Path
for phase in ('baseline', 'interrupted', 'resume'):
    assert json.loads((Path(sys.argv[1]) / f'{phase}-PASS.json').read_text())['passed']
PY
for case in failed worst; do
    if [ ! -f "$campaign/profile-$case/PASS.json" ]; then
        torchrun --standalone --nproc_per_node=4 -m scripts.vln.profile_input_staging \
            --case "$case" --out "$campaign/profile-$case" \
            --vln_config configs/vln_r2r_v0_base.yaml \
            --output_config configs/vln_joint_b_2epoch_bs8.yaml \
            --policy_config configs/vln_r2r_v0_candidate_logits.yaml \
            --model_name_or_path "$VLN_MODEL_PATH" --manifest "$manifest" \
            > "$campaign/profile-$case.log" 2>&1
    fi
    python -m scripts.vln.profile_input_staging --verify-only \
        --case "$case" --out "$campaign/profile-$case" --manifest "$manifest" \
        >> "$campaign/profile-$case.log" 2>&1
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
    --model_name_or_path "$VLN_MODEL_PATH" --manifest "$manifest" \
    --output_dir "$run" --run-name joint15-fullcontext-candidate \
    --recovery-save-steps 100 --resume_from_checkpoint auto \
    >> "$run/train.log" 2>&1
python -m scripts.vln.summarize_run "$run" > "$run/summary.log"
