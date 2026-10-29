# HiPerGator environment for SimpleMemVLN S8N RL/evaluation (EVALUATION_SETUP.md variable names).
# Usage: source /blue/yu954774.ucf/an221229.ucf/code/svln/ops/env_hpg.sh   (SMV_ROOT may be preset)
export SVLN_BASE=/blue/yu954774.ucf/an221229.ucf
export SMV_ROOT=${SMV_ROOT:-$SVLN_BASE/code/svln/RL_s8n}
export DATA_ROOT=/orange/yu954774.ucf/an221229.ucf/svln_data
export MODEL_PYTHON=$SMV_ROOT/.venv/bin/python
export EVAL_PYTHON=$SVLN_BASE/envs/habitat-vln310/bin/python
export MODEL_PATH=$SVLN_BASE/checkpoints/svln/qwen-base
export S8N_CHECKPOINT=$SVLN_BASE/checkpoints/svln/M5_S/S8N
export R2R_DATA="$DATA_ROOT/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz"
export MP3D_SCENES=$DATA_ROOT/scene_datasets
export R2R_CONFIG=$SMV_ROOT/docs/evaluation_setup/vln_r2r.yaml
export SIMULATOR_SOURCE=$SMV_ROOT/docs/evaluation_setup
export PYTHONPATH=$SMV_ROOT/src:$SIMULATOR_SOURCE
# Caches off the small home quota.
export XDG_CACHE_HOME=$SVLN_BASE/.cache
export HF_HOME=$SVLN_BASE/.cache/huggingface HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=$SVLN_BASE/.cache/triton TORCHINDUCTOR_CACHE_DIR=$SVLN_BASE/.cache/inductor
export TILELANG_CACHE_DIR=$SVLN_BASE/.cache/tilelang UV_CACHE_DIR=$SVLN_BASE/.cache/uv
export TOKENIZERS_PARALLELISM=false
# EVALUATION_SETUP.md 11.1 runtime variables. Quiet simulator logs also keep habitat-sim's C++ logger
# (it writes to fd 1) off the RL env-server protocol pipe.
unset VIRTUAL_ENV
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 HF_HUB_DISABLE_PROGRESS_BARS=1
export HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet
# Code provenance: commit + dirty files in a git worktree; in a freeze_code.sh snapshot (no .git) its
# origin record and a check that no file changed since freezing.
provenance() {
  if [[ -f $SMV_ROOT/FROZEN_FROM.txt ]]; then
    head -3 "$SMV_ROOT/FROZEN_FROM.txt"
    (cd "$SMV_ROOT" && sha256sum --quiet -c FROZEN_SHA256SUMS) || { echo "snapshot changed since freezing"; return 1; }
    echo "snapshot checksums ok"
  else
    git -C "$SMV_ROOT" rev-parse --short HEAD && git -C "$SMV_ROOT" status --short
  fi
}
# Nodes with faulty GPUs (one per line) excluded from every submission: c0800a-s5 GPU 1 raised
# "uncorrectable ECC error" at torch.cuda.set_device (2026-10-27, smoke 48012174).
exclude_bad_nodes() { local n; n=$(grep -v '^\s*$' "$SVLN_BASE/code/svln/ops/bad_nodes.txt" 2>/dev/null | paste -sd,); [[ -n $n ]] && echo "--exclude=$n" || true; }
# Fail fast on a GPU that cannot be used (ECC faults surface at the first CUDA call on the device).
gpu_health() {
  nvidia-smi --query-gpu=index,uuid,ecc.errors.uncorrected.volatile.total --format=csv,noheader
  "$MODEL_PYTHON" -c "import torch; n = torch.cuda.device_count(); [torch.ones(1, device=f'cuda:{i}').sum().item() for i in range(n)]; print('gpu health ok:', n, 'devices')" \
    || { echo "GPU health check failed on $(hostname): add it to ops/bad_nodes.txt"; return 1; }
}
