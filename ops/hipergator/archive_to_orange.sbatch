#!/bin/bash
#SBATCH --job-name=svln-archive
#SBATCH --account=yu954774.ucf
#SBATCH --qos=yu954774.ucf-b
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/blue/yu954774.ucf/an221229.ucf/svln_logs/archive_%j.out
# Move run artifacts from /blue (group quota 1 TB, shared) to /orange and leave a symlink at the old path,
# so every reference still resolves. Each source file is removed only after rsync copied it.
# Usage: sbatch archive_to_orange.sbatch DIR [DIR ...]   (DIR under /blue/.../an221229.ucf/svln_runs)
# A DIR given as RUN/resumable/LATEST is resolved to the run's newest resumable checkpoint at run time.
set -euo pipefail
BLUE=/blue/yu954774.ucf/an221229.ucf/svln_runs
ORANGE=/orange/yu954774.ucf/an221229.ucf/svln_archive
for src in "$@"; do
  if [[ $(basename "$src") == LATEST ]]; then
    src=$(ls -d "$(dirname "$src")"/step_* | sort | tail -1)
  fi
  [[ $src == "$BLUE"/* && -d $src && ! -L $src ]] || { echo "skip $src"; continue; }
  dst=$ORANGE/${src#"$BLUE"/}
  mkdir -p "$dst"
  echo "$(date -Is) $src -> $dst ($(du -sh "$src" | cut -f1))"
  rsync -a --remove-source-files "$src"/ "$dst"/
  find "$src" -depth -type d -empty -delete
  [[ -e $src ]] && { echo "not empty after rsync: $src" >&2; exit 1; }
  ln -s "$dst" "$src"
done
blue_quota | sed -n 3,4p
echo "end=$(date -Is)"
