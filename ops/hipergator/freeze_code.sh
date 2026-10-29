#!/bin/bash
# Immutable code snapshot for a campaign: runs and evaluations use SMV_ROOT=<snapshot>, so later edits
# of the worktree never reach running jobs (env-server restarts re-import from disk).
# Usage: freeze_code.sh TAG   -> prints the snapshot path
set -euo pipefail
TAG=${1:?tag}
SRC=${SRC:-/blue/yu954774.ucf/an221229.ucf/code/svln/RL_s8n}
DEST=/blue/yu954774.ucf/an221229.ucf/svln_runs/code/$TAG
[[ -e $DEST ]] && { echo "exists: $DEST" >&2; exit 1; }
mkdir -p "$DEST"
rsync -a --exclude .venv --exclude .git --exclude '__pycache__' --exclude '*.pyc' "$SRC/" "$DEST/"
ln -s "$SRC/.venv" "$DEST/.venv"       # PYTHONPATH=$DEST/src precedes the venv's editable path
{
  echo "source=$SRC"
  echo "frozen=$(date -Is)"
  git -C "$SRC" rev-parse HEAD
  git -C "$SRC" status --porcelain
  git -C "$SRC" diff
} > "$DEST/FROZEN_FROM.txt"
(cd "$DEST" && find src configs scripts data docs -type f ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum) > "$DEST/FROZEN_SHA256SUMS"
chmod -R a-w "$DEST"
echo "$DEST"
