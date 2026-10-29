#!/bin/bash
#SBATCH --job-name=svln-eval
#SBATCH --account=yu954774.ucf
#SBATCH --qos=yu954774.ucf
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=24:00:00
#SBATCH --output=/blue/yu954774.ucf/an221229.ucf/svln_logs/eval_%j.out
# Unchanged evaluator (qwen_vl.eval.habitat_r2r, greedy, seed 42, 500-action cap, HFOV 79) in
# WORKERS_PER_GPU processes per GPU (default 3: one evaluator is CPU-bound and leaves its GPU ~40% idle),
# each on a strided slice order[i::N] of the sorted episode list (balanced loads, scene order kept within a
# slice), merged with scripts/rl/eval_shards.py. Episodes are scored independently (fresh session and
# Habitat reset per episode), so the merged records do not depend on N. A worker whose process dies
# resumes from its journal (eval_shards.py retry first), at most 3 attempts; every restart is logged.
# Usage: CHECKPOINT=... OUT=... [SPLIT=val_unseen|val_seen] [EPISODES=ids.json] [WORKERS_PER_GPU=3] sbatch eval_r2r.sbatch
set -euo pipefail
source /blue/yu954774.ucf/an221229.ucf/code/svln/ops/env_hpg.sh
: "${CHECKPOINT:?}" "${OUT:?}"
# afterany chains: a missing export (failed training) ends the job at once instead of retrying shards.
[[ -f $CHECKPOINT/navigation.json ]] || { echo "no checkpoint at $CHECKPOINT" >&2; exit 1; }
SPLIT=${SPLIT:-val_unseen}
mkdir -p "$OUT"
cd "$SMV_ROOT"
export PYTORCH_ALLOC_CONF=expandable_segments:True   # EVALUATION_SETUP.md 11.1 (evaluation only; RL IPC needs it unset)
echo "host=$(hostname) gpus=$CUDA_VISIBLE_DEVICES start=$(date -Is) checkpoint=$CHECKPOINT split=$SPLIT"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
GPUS=(${CUDA_VISIBLE_DEVICES//,/ })
N=$(( ${#GPUS[@]} * ${WORKERS_PER_GPU:-3} ))
"$EVAL_PYTHON" -I - "$R2R_DATA" "$SPLIT" "${EPISODES:-}" "$OUT" "$N" <<'PY'
import gzip, json, sys
from pathlib import Path
data, split, wanted, out, n = sys.argv[1:6]
n = int(n)
episodes = json.load(gzip.open(data.format(split=split)))['episodes']
if wanted:
    keep = set(json.loads(Path(wanted).read_text()))
    episodes = [e for e in episodes if str(e['episode_id']) in keep]
order = [str(e['episode_id']) for e in sorted(episodes, key=lambda e: (e['scene_id'], str(e['episode_id'])))]
base = Path(out) / 'order'
base.mkdir(exist_ok=True)
(base / 'evaluation_contract.json').write_text(json.dumps(dict(episode_ids=order)))
for i in range(n):
    (Path(out) / f'shard{i}.json').write_text(json.dumps(order[i::n]))
print(len(order), 'episodes in', n, 'strided shards')
PY
run_shard() {   # shard index, GPU
  local i=$1 rc=1 attempt
  for attempt in 1 2 3; do
    "$MODEL_PYTHON" scripts/rl/eval_shards.py retry "$OUT/shard$i" 2>&1 || true
    CUDA_VISIBLE_DEVICES=$2 "$EVAL_PYTHON" -u -m qwen_vl.eval.habitat_r2r \
      --dataset-kind r2r --checkpoint "$CHECKPOINT" \
      --model-python "$MODEL_PYTHON" --model-path "$MODEL_PATH" \
      --habitat-config "$R2R_CONFIG" --data-path "$R2R_DATA" \
      --scenes-dir "$MP3D_SCENES" --gpu 0 --split "$SPLIT" --episode-list "$OUT/shard$i.json" \
      --model-startup-timeout 3600 --out "$OUT/shard$i" >> "$OUT/shard$i.log" 2>&1 && return 0
    rc=$?
    echo "$(date -Is) shard$i attempt $attempt exited rc=$rc after $(wc -l < "$OUT/shard$i/episodes.jsonl" 2>/dev/null || echo 0) episodes"
  done
  return $rc
}
pids=()
for ((i = 0; i < N; i++)); do
  run_shard "$i" "${GPUS[$((i % ${#GPUS[@]}))]}" &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
"$MODEL_PYTHON" scripts/rl/eval_shards.py merge "$OUT/order" "$OUT/merged" $(for ((i = 0; i < N; i++)); do echo "$OUT/shard$i"; done)
cat "$OUT/merged/summary.json"
echo "end=$(date -Is) status=$status"
exit $status
