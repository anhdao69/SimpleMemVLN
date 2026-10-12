# Resume the existing two-H100 attempts

Only run these after the previous job is terminal and a complete recovery checkpoint exists. Keep the exact source, arm, manifest, topology and attempt directory. The coordinator automatically resumes from the latest valid full-state checkpoint and refuses a live previous owner. A canceled run before its first complete save must start again. V0 4877/a1 is superseded: use 4880/a2 below. Record the new job ID in the campaign ledger; lifecycle events are also recorded automatically.

## P1-M1-s429

```bash
sbatch --parsable --job-name=epi-P1-M1-s429-H2 --comment=fddde5b1-0f38-4185-a751-f96d4a0e2975/P1-M1-s429/P1-M1-s429-H2-a1 --partition=main --nodes=1 --ntasks=1 --gres=gpu:nvidia_h100_80gb_hbm3:2 --cpus-per-task=30 --mem=512G --time=5-00:00:00 --signal=B:USR1@1800 --output=/mnt/data/vmo-ai-task/anhdh35/checkpoints/SimpleMemVLN-c2-h2-speed-20261011/runs/P1-M1-s429/P1-M1-s429-H2-a1/slurm-%j.log /mnt/data/vmo-ai-task/anhdh35/checkpoints/SimpleMemVLN-c2-h2-speed-20261011/source/e789a13f8b6ea3abc29086b82205a19024a6e7b5/train/slurm/episodic_candidate.sbatch /mnt/data/vmo-ai-task/anhdh35/checkpoints/SimpleMemVLN-c2-h2-speed-20261011/runs/P1-M1-s429/P1-M1-s429-H2-a1
```

## P1-M2-s429

```bash
sbatch --parsable --job-name=epi-P1-M2-s429-H2 --comment=fddde5b1-0f38-4185-a751-f96d4a0e2975/P1-M2-s429/P1-M2-s429-H2-a1 --partition=main --nodes=1 --ntasks=1 --gres=gpu:nvidia_h100_80gb_hbm3:2 --cpus-per-task=30 --mem=512G --time=5-00:00:00 --signal=B:USR1@1800 --output=/mnt/data/vmo-ai-task/anhdh35/checkpoints/SimpleMemVLN-c2-h2-speed-20261011/runs/P1-M2-s429/P1-M2-s429-H2-a1/slurm-%j.log /mnt/data/vmo-ai-task/anhdh35/checkpoints/SimpleMemVLN-c2-h2-speed-20261011/source/e789a13f8b6ea3abc29086b82205a19024a6e7b5/train/slurm/episodic_candidate.sbatch /mnt/data/vmo-ai-task/anhdh35/checkpoints/SimpleMemVLN-c2-h2-speed-20261011/runs/P1-M2-s429/P1-M2-s429-H2-a1
```

## P1-V0-s429

```bash
sbatch --parsable --job-name=epi-P1-V0-s429-H2 --comment=fddde5b1-0f38-4185-a751-f96d4a0e2975/P1-V0-s429/P1-V0-s429-H2-a2 --partition=main --nodes=1 --ntasks=1 --gres=gpu:nvidia_h100_80gb_hbm3:2 --cpus-per-task=30 --mem=512G --time=5-00:00:00 --signal=B:USR1@1800 --output=/mnt/data/vmo-ai-task/anhdh35/checkpoints/SimpleMemVLN-c2-h2-speed-20261011/runs/P1-V0-s429/P1-V0-s429-H2-a2/slurm-%j.log /mnt/data/vmo-ai-task/anhdh35/checkpoints/SimpleMemVLN-c2-h2-speed-20261011/source/e789a13f8b6ea3abc29086b82205a19024a6e7b5/train/slurm/episodic_candidate.sbatch /mnt/data/vmo-ai-task/anhdh35/checkpoints/SimpleMemVLN-c2-h2-speed-20261011/runs/P1-V0-s429/P1-V0-s429-H2-a2
```

