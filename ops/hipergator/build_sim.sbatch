#!/bin/bash
#SBATCH --job-name=svln-sim
#SBATCH --account=yu954774.ucf
#SBATCH --qos=yu954774.ucf-b
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/blue/yu954774.ucf/an221229.ucf/svln_logs/sim_%j.out
# Simulator environment, EVALUATION_SETUP.md 6.1-6.3 and 7, Route B (aihabitat has no py3.10 build of
# habitat-sim 0.2.4): Python 3.10.20, habitat-sim f179b584 source build --headless --no-bullet,
# habitat-lab 1639e1ae, the project's constraints file, RxR registrations from docs/evaluation_setup.
set -euo pipefail
BASE=/blue/yu954774.ucf/an221229.ucf
SMV=$BASE/code/svln/RL_s8n
SRC=$BASE/code/svln/habitat-source
SIM=$BASE/envs/habitat-vln310
CONSTRAINTS=$SMV/docs/evaluation_setup/habitat-constraints.txt
source /apps/lmod/lmod/init/bash
module load conda/25.7.0
echo "host=$(hostname) start=$(date -Is)"
if [[ ! -x $SIM/bin/python ]]; then
  conda create -y -p "$SIM" -c conda-forge --override-channels python=3.10.20 pip
fi
PY=$SIM/bin/python
"$PY" --version
mkdir -p "$SRC"
if [[ ! -d $SRC/habitat-sim ]]; then
  git clone -q https://github.com/facebookresearch/habitat-sim.git "$SRC/habitat-sim"
fi
git -C "$SRC/habitat-sim" checkout -q f179b584bcd713c5a2a998132211e2cae881d6d1
# gitlab.com refused every Eigen clone from HiPerGator ("unable to handle this request due to load",
# 2026-10-27). Fetch the identical pinned commit 3147391d (Eigen 3.4.0) from its GitHub mirror instead;
# git object hashes make the checked-out tree byte-identical, and the commit is asserted below.
git -C "$SRC/habitat-sim" submodule init -q
git -C "$SRC/habitat-sim" config submodule.src/deps/eigen.url https://github.com/eigen-mirror/eigen.git
for attempt in 1 2 3 4 5 6 7 8; do
  git -C "$SRC/habitat-sim" submodule update --init --recursive -q && break
  [[ $attempt == 8 ]] && { echo 'submodule update failed 8 times'; exit 1; }
  sleep $((attempt * 120))
done
git -C "$SRC/habitat-sim" describe --tags --always
# Aborted clone attempts can leave a submodule at its pinned HEAD with an empty index: check out every
# submodule's recorded commit and refuse to build if any tracked file is missing.
git -C "$SRC/habitat-sim" submodule foreach --quiet --recursive 'git reset -q --hard "$sha1"'
git -C "$SRC/habitat-sim" submodule foreach --quiet --recursive \
  '[ "$(git ls-files | wc -l)" -gt 0 ] && [ -z "$(git ls-files --deleted)" ] || { echo "empty submodule $displaypath"; exit 1; }'
[[ $(git -C "$SRC/habitat-sim/src/deps/eigen" rev-parse HEAD) == 3147391d946bb4b6c68edd901f2add6ac1f31f8c ]]
git -C "$SRC/habitat-sim" submodule status --recursive | awk '$1 ~ /^[-+U]/ {bad=1; print "submodule not at pinned commit:", $0} END {exit bad}'
"$PY" -m pip install -q 'setuptools<70' wheel 'cmake>=3.14,<4' ninja
"$PY" -m pip install -q -c "$CONSTRAINTS" -r "$SRC/habitat-sim/requirements.txt"
if ! "$PY" -c 'import habitat_sim' 2>/dev/null; then
  cd "$SRC/habitat-sim"
  export CMAKE_BUILD_PARALLEL_LEVEL=$SLURM_CPUS_PER_TASK MAX_JOBS=$SLURM_CPUS_PER_TASK
  export PATH=$SIM/bin:$PATH
  # --force-cmake: a configure that failed earlier leaves a cache setup.py would otherwise trust.
  echo "compiler $(command -v c++): $(c++ --version | head -1)"; env | grep -iE 'cpath|include|flags|^cc=|^cxx=' || true
  if ! "$PY" setup.py install --headless --no-bullet --force-cmake; then
    obj=build/deps/magnum-bindings/src/python/corrade/CMakeFiles/corrade.dir/corrade.cpp.o
    line=$(grep -F "-o $obj -c" build/compile_commands.json 2>/dev/null | head -1 || true)
    [[ -n $line ]] && (cd build && eval "$(echo "$line" | sed -E 's/^ *"command": "//; s/",? *$//; s/\\"/"/g; s/ -o [^ ]+/ /; s/ -c / -fsyntax-only -H /')" 2>&1 | grep -E "unicodeobject|Python.h|error" | head -20)
    exit 1
  fi
  cd -
fi
if [[ ! -d $SRC/habitat-lab ]]; then
  git clone -q https://github.com/facebookresearch/habitat-lab.git "$SRC/habitat-lab"
fi
git -C "$SRC/habitat-lab" checkout -q 1639e1ae732ba1e84199a1a04b79c7243c3f8586
git -C "$SRC/habitat-lab" describe --tags --always
"$PY" -m pip install -q -c "$CONSTRAINTS" -e "$SRC/habitat-lab/habitat-lab"
"$PY" -m pip install -q -c "$CONSTRAINTS" fastdtw
"$PY" -m pip check || true
cd /
"$PY" - <<'PY'
import sys, importlib.metadata as m
import habitat, habitat_sim
print(sys.executable, sys.version)
for name in ('habitat-lab', 'habitat-sim', 'numpy', 'gym', 'hydra-core', 'omegaconf', 'fastdtw'):
    print(name, m.version(name))
assert m.version('habitat-lab') == '0.2.4'
assert m.version('habitat-sim') == '0.2.4'
assert m.version('numpy') == '1.26.4'
print('Habitat-Lab:', habitat.__file__)
print('Habitat-Sim:', habitat_sim.__file__)
print('Simulator CUDA interop:', habitat_sim.cuda_enabled)
print('Bullet:', habitat_sim.built_with_bullet)
PY
PYTHONPATH="$SMV/src:$SMV/docs/evaluation_setup" "$PY" - <<'PY'
from habitat_extensions import task, measures
from habitat.core.registry import registry
assert registry.get_dataset('RxRVLNCE-v1') is not None
assert registry.get_measure('NDTW') is not None
print('RxR dataset and nDTW registrations imported successfully')
PY
echo "end=$(date -Is)"
