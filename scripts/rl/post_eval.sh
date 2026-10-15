#!/usr/bin/env bash
# Wait for supervised evaluations to finish and write the standard analyses into plans/RL/research/results.
# Usage: post_eval.sh "LABEL EVAL_DIR" ["LABEL EVAL_DIR" ...]
#   EVAL_DIR is /data/anh/rl_baseline/evals/<run>-stepNNNN (done when its comparison.json exists).
# For each: paired comparisons vs SFT, both GRPO seeds and GRPO+F step 125; failure analysis vs GRPO s2;
# then one memory analysis over every finished policy.
set -uo pipefail
cd /data/anh/code/SimpleMem
E=/data/anh/rl_baseline/evals
OUT=plans/RL/research/results
D=/data/anh/datasets/simplemem/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen
SFT=/data/anh/simplemem-eval/results/r2r-full/epoch-2
log=/data/anh/rl_baseline/logs/post_eval.log
declare -A BASES=([sft]=$SFT [grpo_s1]=$E/grpo_r2r1k-step0125/val_unseen
                  [grpo_s2]=$E/grpo_r2r1k_seed2-step0125/val_unseen
                  [grpo_fork125]=$E/hico_grpo_fork_r2r1k-step0125/val_unseen)
specs=("$@")
for spec in "${specs[@]}"; do
  set -- $spec
  label=$1; dir=$2
  until [ -f "$dir/comparison.json" ]; do sleep 300; done
  echo "$(date -Is) analysing $label ($dir)" >> "$log"
  cp "$dir/val_unseen/summary.json" "$OUT/${label}_val_unseen_summary.json"
  for base in "${!BASES[@]}"; do
    [ "${BASES[$base]}" = "$dir/val_unseen" ] && continue
    python3 scripts/rl/compare_evals.py --base "${BASES[$base]}" --new "$dir/val_unseen" --name-base "$base" \
      --name-new "$label" --out "$OUT/${label}_vs_${base}_comparison.json" >> "$log" 2>&1
  done
  .venv/bin/python scripts/rl/analyze_failures.py "${BASES[grpo_s2]}" "$dir/val_unseen" "$D/val_unseen.json.gz" \
    "$D/val_unseen_gt.json.gz" "$OUT/${label}_failure_analysis_vs_grpo_s2.json" > /dev/null 2>> "$log"
done
args=(sft=$SFT)
for d in grpo_r2r1k-step0050:grpo_s1_step50 grpo_r2r1k-step0125:grpo_s1 grpo_r2r1k_seed2-step0125:grpo_s2 \
         hico_v3_r2r1k-step0125:hico_v3 hico_grpo_stop_r2r1k-step0125:grpo_stop \
         hico_grpo_fork_r2r1k-step0050:grpo_fork_step50 hico_grpo_fork_r2r1k-step0075:grpo_fork_step75 \
         hico_grpo_fork_r2r1k-step0125:grpo_fork hico_grpo_fork_bs_r2r1k-step0125:grpo_fork_bs \
         hico_v2a_r2r1k-step0125:hico_v2a; do
  [ -f "$E/${d%%:*}/val_unseen/summary.json" ] && args+=("${d##*:}=$E/${d%%:*}/val_unseen")
done
.venv/bin/python scripts/rl/analyze_memory.py "$OUT/memory_analysis_all.json" "${args[@]}" >> "$log" 2>&1
echo "$(date -Is) post-eval done: ${specs[*]}" >> "$log"
