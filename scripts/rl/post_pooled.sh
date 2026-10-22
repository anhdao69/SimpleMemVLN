#!/usr/bin/env bash
# Seed-pooled comparisons once the second GRPO+F-B seed (and then GRPO G=8) are evaluated.
# Writes plans/RL/research/results/pooled_*.json; log /data/anh/rl_baseline/logs/post_eval.log.
set -uo pipefail
cd /data/anh/code/SimpleMem
E=/data/anh/rl_baseline/evals
OUT=plans/RL/research/results
log=/data/anh/rl_baseline/logs/post_eval.log
G1=$E/grpo_r2r1k-step0125/val_unseen; G2=$E/grpo_r2r1k_seed2-step0125/val_unseen
F1=$E/hico_grpo_fork_bs_r2r1k-step0125/val_unseen; F2=$E/hico_grpo_fork_bs_r2r1k_seed2-step0125/val_unseen
G8=$E/grpo_g8_r2r1k-step0125/val_unseen
until [ -f "$E/hico_grpo_fork_bs_r2r1k_seed2-step0125/comparison.json" ]; do sleep 300; done
python3 scripts/rl/pooled_compare.py "$OUT/pooled_grpo2seeds_vs_grpo_fork_bs2seeds.json" \
  --a grpo "$G1" "$G2" --b grpo_fork_bs "$F1" "$F2" >> "$log" 2>&1
echo "$(date -Is) pooled: grpo (2 seeds) vs grpo_fork_bs (2 seeds) written" >> "$log"
until [ -f "$E/grpo_g8_r2r1k-step0125/comparison.json" ]; do sleep 300; done
python3 scripts/rl/pooled_compare.py "$OUT/pooled_grpo_g8_vs_grpo_fork_bs2seeds.json" \
  --a grpo_g8 "$G8" --b grpo_fork_bs "$F1" "$F2" >> "$log" 2>&1
echo "$(date -Is) pooled: grpo_g8 vs grpo_fork_bs (2 seeds) written" >> "$log"
