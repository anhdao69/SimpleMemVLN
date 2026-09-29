#!/usr/bin/env bash
#SBATCH --job-name=smv-joint15-full-b-e2
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/outputs/joint15_b_bs8_e2_20260929/slurm-%j.log
set -euo pipefail
bash /mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/outputs/joint15_b_bs8_e2_20260929/source/train/slurm/run_joint_b.sh train
