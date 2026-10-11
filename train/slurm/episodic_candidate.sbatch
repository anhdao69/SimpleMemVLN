#!/usr/bin/env bash
# One candidate attempt of one logical episodic run.  Resources, job name and
# campaign comment are supplied by scripts/vln/submit_episodic.py; never submit
# this file by hand with a reused attempt directory.
#SBATCH --partition=highgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
attempt_dir=${1:?attempt directory}
manifest="$attempt_dir/launch_manifest.json"
read_field() { python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(d$2)" "$3"; }
campaign=$(read_field "$manifest" "['campaign_root']" "$manifest")
source_dir=$(read_field "$manifest" "['source_dir']" "$manifest")
cm="$campaign/campaign_manifest.json"
env_dir=$(read_field "$cm" "['env_dir']" "$cm")
export CUDA_HOME=$(read_field "$cm" "['cuda_home']" "$cm")
export CXX=$(read_field "$cm" "['cxx']" "$cm")
export CC=${CXX%g++}gcc
export PATH="$env_dir/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$(dirname "$(dirname "$CXX")")/lib64:${LD_LIBRARY_PATH:-}"
export HF_HOME=$(read_field "$cm" "['hf_home']" "$cm")
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$(read_field "$manifest" "['resources']['omp']" "$manifest")
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS=1
export TMPDIR=$(mktemp -d "/tmp/epi-${SLURM_JOB_ID:?}-XXXXXX")
export TRITON_CACHE_DIR="$TMPDIR/triton" TORCH_EXTENSIONS_DIR="$TMPDIR/torch_extensions"
export PYTHONPATH="$source_dir/src:$source_dir"
cd "$source_dir"
echo "EPISODIC_JOB ${SLURM_JOB_ID} $(hostname) attempt=$attempt_dir gpus=${CUDA_VISIBLE_DEVICES:-}"
exec python -m scripts.vln.episodic_job "$attempt_dir"
