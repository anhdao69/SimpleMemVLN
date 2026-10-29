#!/bin/bash
#SBATCH --job-name=svln-tests
#SBATCH --account=yu954774.ucf
#SBATCH --qos=yu954774.ucf-b
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/blue/yu954774.ucf/an221229.ucf/svln_logs/tests_%j.out
# CPU test suite of the checkout (no GPU, no simulator). Usage: [TESTS="tests/rl ..."] sbatch cpu_tests.sbatch
set -euo pipefail
source /blue/yu954774.ucf/an221229.ucf/code/svln/ops/env_hpg.sh
export CHECKPOINT=$S8N_CHECKPOINT OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SMV_ROOT"
echo "host=$(hostname) start=$(date -Is)"
provenance
.venv/bin/python -m pytest -q -p no:cacheprovider --durations=15 ${TESTS:-tests/rl tests/memory_sweep} 2>&1
echo "end=$(date -Is)"
