#!/usr/bin/env bash
#SBATCH --job-name=smvln-b-upload
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/artifacts/slurm/%x-%j.out
#SBATCH --error=/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/artifacts/slurm/%x-%j.err
set -euo pipefail
root="${1:?source root}"
campaign="${2:?campaign}"
variant="${3:?variant}"
cd "$root"
export PYTHONPATH="$root/src:$root"
export HF_TOKEN_PATH=/mnt/data/vmo-ai-task/anhdh35/.simplememvln-hf/token
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export OMP_NUM_THREADS=2 HF_HUB_DISABLE_PROGRESS_BARS=1
python_bin=/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/.venv/bin/python
for attempt in 1 2 3 4 5; do
    if "$python_bin" -m scripts.vln.upload_campaign --campaign "$campaign" --variant "$variant"; then
        exit 0
    fi
    echo "Upload/verification failed on attempt $attempt; training dependencies are unaffected."
    if (( attempt < 5 )); then sleep $((attempt * 60)); fi
done
exit 1
