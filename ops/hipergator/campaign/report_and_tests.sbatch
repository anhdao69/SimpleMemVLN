#!/bin/bash
#SBATCH --job-name=svln-report
#SBATCH --account=yu954774.ucf
#SBATCH --qos=yu954774.ucf-b
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/blue/yu954774.ucf/an221229.ucf/svln_logs/report_%j.out
# S8N RL report tables (scripts/rl/s8n_report.py), then the CPU test suite of the RL_s8n checkout.
set -euo pipefail
source /blue/yu954774.ucf/an221229.ucf/code/svln/ops/env_hpg.sh
export CHECKPOINT=$S8N_CHECKPOINT OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SMV_ROOT"
echo "host=$(hostname) start=$(date -Is)"
provenance
"$MODEL_PYTHON" scripts/rl/s8n_report.py
echo "report end=$(date -Is)"
"$MODEL_PYTHON" -m pytest -q -p no:cacheprovider --durations=15 tests/rl tests/memory_sweep 2>&1 || echo "pytest rc=$?"
echo "end=$(date -Is)"
