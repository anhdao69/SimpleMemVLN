#!/bin/bash
# LOCAL 2026-11-13. User: "train the CoRe for one more seed for me and ActiveVLN for one more seed with R2R + RxR mix".
# Both runs: seed 20261414, from SFR (Stage 2, e3ada114), the mix1k pool (data/rl/mix_r2r1k_rxr1k.jsonl, 6ab04818:
# 500 R2R + 500 RxR alternating, = rows 0-999 of the A+ 8k pool), 125 steps, both local GPUs, one after the other:
#   1. GRPO (ActiveVLN recipe): configs/rl/local/activevln_grpo_sfr_mix1k_seed1414.yaml -> grpo_sfr_mix1k_seed1414
#   2. CoRe (rewind):           configs/rl/local/rewind_grpo_sfr_mix1k_seed1414.yaml   -> rewind_sfr_mix1k_seed1414
# Each goes through ops/local/hard2k_chain.sh: freeze the dev tree (STABLE-V1 code; defaults = the old code path:
# adv_scale group, constant LR), 2-step smoke with provenance checks, 125 steps with the drift guard, R2R val_unseen
# of exports/step_0125, verify, paired compares. RxR is evaluated later from the kept export (user's rule: R2R unless
# asked). Nothing is removed; every export and the last resumable state are kept.
# Usage: setsid nohup bash /nvme/public/svln/analysis/paper/seed1414_chain.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
DEV=/nvme/public/SimpleMemVLN_dev RUNS=/nvme/public/svln/runs
exec >> "$RUNS/seed1414_chain.log" 2>&1
say() { echo "$(date -Is) $*"; }
say "seed-1414 chain start"
while pgrep -u "$(id -u)" -f "[s]upervise_run.sh|[t]rain_grpo|[f]inal_evals.sh|[h]abitat_r2r" > /dev/null; do sleep 60; done
say "no trainer or evaluation of ours is running"
COMMON="MILESTONES=125 BASE_PREFIX=e3ada114 POOL_FILE=data/rl/mix_r2r1k_rxr1k.jsonl POOL_PREFIX=6ab04818"
run_one() {   # NAME CONFIG BASES LABEL
  say "start $1"
  env $COMMON NAME="$1" CONFIG="$2" BASES="$3" LABEL="$4" \
    TABLE=/nvme/public/svln/analysis/clean2k/rewind8k_turns_s790.tsv bash "$DEV/ops/local/hard2k_chain.sh"
  say "$1 chain ended rc=$?: $(tail -1 "$RUNS/$1.chain.log" | cut -c1-220)"
  grep -E "MILESTONE|verify|compare" "$RUNS/$1.chain.log" | cut -c1-260
}
run_one grpo_sfr_mix1k_seed1414 configs/rl/local/activevln_grpo_sfr_mix1k_seed1414.yaml \
  "sfr grpo_sfr_mix8k_aplus_s125 rewind_sfr_mix1k_s125" "prompts, GRPO seed 20261414, mix1k pool from SFR"
while pgrep -u "$(id -u)" -f "[s]upervise_run.sh|[t]rain_grpo|[f]inal_evals.sh|[h]abitat_r2r" > /dev/null; do sleep 60; done
run_one rewind_sfr_mix1k_seed1414 configs/rl/local/rewind_grpo_sfr_mix1k_seed1414.yaml \
  "sfr rewind_sfr_mix1k_s125 grpo_sfr_mix8k_aplus_s125 grpo_sfr_mix1k_seed1414_s125" "prompts, CoRe seed 20261414, mix1k pool from SFR"
say "seed-1414 chain finished"
