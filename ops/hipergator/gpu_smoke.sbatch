#!/bin/bash
#SBATCH --job-name=svln-smoke
#SBATCH --account=yu954774.ucf
#SBATCH --qos=yu954774.ucf
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/blue/yu954774.ucf/an221229.ucf/svln_logs/smoke_%j.out
# GPU development gates on 2 x RTX PRO 6000, in order (stops at the first failure):
#  1. device/cgroup visibility, torch arch list and a flash-attn kernel on sm_120,
#  2. S8N realised-memory replay gate (scripts/rl/check_s8n_replay.py) on GPU 0,
#  3. 2-step trainer smokes (2 prompts x G 2, 40-step budget) for GRPO and GRPO+F-B on S8N,
#     checked by scripts/rl/check_fork_smoke.py.
set -euo pipefail
source /blue/yu954774.ucf/an221229.ucf/code/svln/ops/env_hpg.sh
cd "$SMV_ROOT"
SMOKE=$SVLN_BASE/svln_runs/smoke/$SLURM_JOB_ID
mkdir -p "$SMOKE"
echo "host=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES start=$(date -Is)"
provenance
gpu_health
nvidia-smi -L
echo "unmasked device count: $(env -u CUDA_VISIBLE_DEVICES nvidia-smi -L | wc -l)"
"$MODEL_PYTHON" -c "import torch; print('torch sees', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], torch.cuda.get_arch_list())"
"$MODEL_PYTHON" - <<'PY'
import torch, flash_attn
from flash_attn import flash_attn_func
q = torch.randn(1, 64, 4, 128, device='cuda', dtype=torch.bfloat16)
print('flash-attn', flash_attn.__version__, 'ok', tuple(flash_attn_func(q, q, q, causal=True).shape))
PY

echo "== 2. S8N replay gate"
"$MODEL_PYTHON" scripts/rl/check_s8n_replay.py "$S8N_CHECKPOINT" "$MODEL_PATH" 0 "$SMOKE/replay" 3

export OMP_NUM_THREADS=4 PYTORCH_ALLOC_CONF= PYTORCH_CUDA_ALLOC_CONF=
port=$((29500 + SLURM_JOB_ID % 2000))
for name in activevln_grpo_s8n_r2r1k hico_grpo_fork_bs_s8n_r2r1k; do
  echo "== 3. trainer smoke $name"
  .venv/bin/torchrun --nproc_per_node=2 --master_port "$port" -m qwen_vl.rl.train_grpo \
    --config "configs/rl/$name.yaml" --output-dir "$SMOKE/$name" \
    --max-steps 2 --prompts-per-step 2 --group-size 2 --budget-limit 40 --export-steps 2 --save-every 2
  flag=(); [[ $name == activevln_* ]] && flag=(--no-forks)
  "$MODEL_PYTHON" scripts/rl/check_fork_smoke.py "$SMOKE/$name" "${flag[@]}"
  port=$((port + 1))
done
echo "end=$(date -Is) smoke=$SMOKE"
