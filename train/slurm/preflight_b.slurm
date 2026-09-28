#!/usr/bin/env bash
#SBATCH --job-name=smvln-b-preflight
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/artifacts/slurm/%x-%j.out
#SBATCH --error=/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/artifacts/slurm/%x-%j.err
set -euo pipefail
exec bash "${1:?source root}/train/slurm/run_b.sh" "$1" "${2:?campaign path}" preflight
