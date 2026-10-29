#!/bin/bash
#SBATCH --job-name=svln-envs
#SBATCH --account=yu954774.ucf
#SBATCH --qos=yu954774.ucf-b
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/blue/yu954774.ucf/an221229.ucf/svln_logs/envs_%j.out
set -euo pipefail
BASE=/blue/yu954774.ucf/an221229.ucf
SMV=$BASE/code/svln/RL_s8n
ENVS=$BASE/envs
OPS=$BASE/code/svln/ops
export PATH=$BASE/.local/bin:$PATH
source /apps/lmod/lmod/init/bash
echo "host=$(hostname) start=$(date -Is)"

# ---- model environment: the branch lock, unchanged ----
if [[ "${SKIP_MODEL:-0}" != 1 ]]; then
  cd "$SMV"
  unset VIRTUAL_ENV
  module load cuda/12.9.1            # deepspeed's build reads CUDA_HOME/bin/nvcc (no ops are prebuilt)
  export CUDA_HOME DS_BUILD_OPS=0
  uv python install 3.12.13
  uv sync --locked --python 3.12.13
  .venv/bin/python - <<'PY'
import importlib.metadata as m
for name in ('torch', 'torchvision', 'transformers', 'flash-attn', 'flash-linear-attention', 'fla-core',
             'tilelang', 'deepspeed', 'numpy'):
    print(name, m.version(name))
import torch; print('wheel CUDA', torch.version.cuda)
PY
fi

# ---- simulator environment: habitat-sim 0.2.4 headless (commit f179b584) + habitat-lab v0.2.4 ----
if [[ "${SKIP_SIM:-0}" != 1 ]]; then
  module load conda/25.7.0
  SIM=$ENVS/habitat-vln39
  if [[ ! -x $SIM/bin/python ]]; then
    conda create -y -p "$SIM" -c conda-forge -c aihabitat --override-channels python=3.9 pip \
      "habitat-sim=0.2.4=py3.9_headless_linux_f179b584bcd713c5a2a998132211e2cae881d6d1"
  fi
  SRC=$BASE/code/svln/habitat-source
  mkdir -p "$SRC"
  if [[ ! -d $SRC/habitat-lab ]]; then
    git clone -q https://github.com/facebookresearch/habitat-lab.git "$SRC/habitat-lab"
  fi
  git -C "$SRC/habitat-lab" checkout -q 1639e1ae732ba1e84199a1a04b79c7243c3f8586
  "$SIM/bin/python" -m pip install -q -c "$OPS/habitat-constraints-py39.txt" -e "$SRC/habitat-lab/habitat-lab"
  "$SIM/bin/python" -m pip install -q -c "$OPS/habitat-constraints-py39.txt" fastdtw
  "$SIM/bin/python" -m pip check || true
  "$SIM/bin/python" - <<'PY'
import sys, importlib.metadata as m
import habitat, habitat_sim
print(sys.version)
for name in ('habitat-lab', 'habitat-sim', 'numpy', 'gym', 'hydra-core', 'omegaconf', 'fastdtw', 'scipy'):
    print(name, m.version(name))
print('cuda interop', habitat_sim.cuda_enabled, 'bullet', habitat_sim.built_with_bullet)
PY
fi
echo "end=$(date -Is)"
