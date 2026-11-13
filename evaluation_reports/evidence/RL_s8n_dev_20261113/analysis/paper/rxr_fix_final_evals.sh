#!/bin/bash
# LOCAL 2026-11-13. RxR val_unseen with the 99 oLBMNvg9in8 episodes on the RxR-CE agent navmesh (rxr99_fix.md) for every
# local model with RxR results, and full RxR evaluations of the seeds that have none. User: "you can stop the training a
# little bit and run these eval for me and update the latest result about RxR", "Maybe you can run it for all the seed you
# got so far". The file name contains "final_evals.sh" on purpose: abl_local_chain.sh (the local CoRe ablations) waits
# while a process with that name runs, so the ablations start only after these evaluations.
#   0. wait for GRPO run 1's 99-episode re-evaluation (rxr99_eval.sh); rebuild its RxR results (full, A, B, C, 2k)
#   1. 52-episode re-evaluations (the 99 within sets A + B) of sfr, rewind_path_sfr_mix1k s125, rewind_sfr s125 on GPU 1
#      with 3 workers next to the CoRe seed-1414 training (as the 99-episode runs did; no pause needed); rebuild their
#      RxR results (A, B, 2k)
#   2. after the CoRe seed-1414 chain ends (training + R2R evaluation): full RxR (3,570 episodes with the evaluator config
#      unchanged + the 99 with the RxR-CE agent navmesh, combined) of GRPO seed 20261414 (grpo_sfr_mix1k_seed1414 s125),
#      CoRe seed 20261414 (rewind_sfr_mix1k_seed1414 s125) and CoRe seed 20261111 (rewind_sfr_mix1k_seed2 s125), both GPUs
# Evaluations: frozen tree svln/code/rxrce_agent_eval_20261113 (src identical to every earlier RxR evaluation), both GPUs,
# no MPS. Nothing is deleted; original evaluations stay as they are.
# Usage: setsid nohup bash /nvme/public/svln/analysis/paper/rxr_fix_final_evals.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
A=/nvme/public/svln/analysis/paper E=/nvme/public/svln/evals C=/nvme/public/svln/code/rxrce_agent_eval_20261113
R=/nvme/public/svln/runs/rewind_sfr_mix1k_seed1414
LOG=$E/rxr99_eval.log
say() { echo "$(date -Is) [local] $*" >> "$LOG"; }
workers() {   # per-GPU worker counts for GPUs 0,1 from free memory: about 11 GiB per worker, at most 5
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -2 |
    awk '{w = int(($1 / 1024 - 6) / 11); if (w > 5) w = 5; if (w < 1) w = 1; printf "%s%d", (NR > 1 ? "," : ""), w}'
}
ev() {   # LAUNCHER CHECKPOINT EPISODE_LIST OUT GPUS WORKERS_PER_GPU
  local g=$5 w=$6
  if [[ -f $4/merged/summary.json ]]; then say "$4 already evaluated"; return 0; fi
  [[ -e $4 ]] && { say "$4 exists without a merged summary: not touched"; return 1; }
  say "evaluate $(basename "$4") ($2, $(python3 -I -c "import json; print(len(json.load(open('$3'))))") episodes, GPUs $g, workers $w, $1)"
  (cd "$C" && USE_MPS=0 SMV_ROOT=$C CUDA_VISIBLE_DEVICES=$g WORKERS_PER_GPU=$w CHECKPOINT=$2 OUT=$4 EPISODES=$3 \
     bash "$C/ops/local/$1" > "$4.log" 2>&1)
  local rc=$?
  say "$(basename "$4") rc=$rc: $(python3 -I -c "
import json; d=json.load(open('$4/merged/summary.json')); print({k: d[k] for k in ('episodes', 'failures', 'sr', 'spl', 'oracle_success', 'failure_reasons')})" 2>&1)"
  return $rc
}
merge() {   # MODE BASE REEVAL OUT
  [[ -e $4 ]] && { say "$(basename "$4") exists"; return 0; }
  (cd /tmp && python3 -I "$A/rxr_fix_merge.py" "$@" >> "$LOG" 2>&1) || { say "merge FAILED: $4"; return 1; }
}
fix_model() {   # MODEL REEVAL: rebuild every RxR result of MODEL that exists
  local s
  for s in rxrfull rxr rxrb_rxr rxrc_rxr rxr2k; do
    [[ -f $E/$1_${s}_val_unseen/merged/episodes.jsonl ]] && merge replace "$E/$1_${s}_val_unseen" "$2" "$E/$1_${s}_fixed_val_unseen"
  done
}
full_rxr() {   # MODEL CHECKPOINT: 3,570 + 99, combined
  local w; w=$(workers)
  ev eval_rxr.sh "$2" "$A/rxr3570_episode_ids.json" "$E/$1_rxr3570_val_unseen" 0,1 "$w" &&
  ev eval_rxr_rxrce_agent.sh "$2" "$A/rxr99_episode_ids.json" "$E/$1_rxr99_rxrce_agent_val_unseen" 0,1 "$w" &&
  merge combine "$E/$1_rxr3570_val_unseen" "$E/$1_rxr99_rxrce_agent_val_unseen" "$E/$1_rxrfull_fixed_val_unseen"
}

say "rxr_fix_final_evals start"
# 0.
while pgrep -u "$(id -u)" -f "[r]xr99_eval.sh" > /dev/null; do sleep 30; done
fix_model grpo_sfr_mix8k_aplus_s125 "$E/grpo_sfr_mix8k_aplus_s125_rxr99_rxrce_agent_val_unseen"
# 1.
for m in sfr:/nvme/public/svln/checkpoints/stage2_S8N/S8N_DAGGER_SF_REMAINING \
         rewind_path_sfr_mix1k_s125:/nvme/public/svln/runs/rewind_path_sfr_mix1k/exports/step_0125 \
         rewind_sfr_s125:/nvme/public/svln/runs/rewind_sfr/exports/step_0125; do
  M=${m%%:*} K=${m#*:}
  ev eval_rxr_rxrce_agent.sh "$K" "$A/rxr52_episode_ids.json" "$E/${M}_rxr52_rxrce_agent_val_unseen" 1 3 &&
    fix_model "$M" "$E/${M}_rxr52_rxrce_agent_val_unseen"
  pkill -u "$(id -u)" -f "$C/.*[h]abitat_r2r"; sleep 5
done
# 2.
say "waiting for the CoRe seed-1414 chain (training + R2R evaluation) to end"
until grep -qE "hard-2k chain finished|chain ends" "$R.chain.log" 2> /dev/null || ! pgrep -u "$(id -u)" -f "[s]eed1414_chain.sh" > /dev/null; do sleep 60; done
while pgrep -u "$(id -u)" -f "[s]upervise_run.sh|[t]rain_grpo|[h]abitat_r2r" > /dev/null; do sleep 60; done
say "seed-1414 chain: $(tail -1 "$R.chain.log" | cut -c1-200)"
full_rxr grpo_sfr_mix1k_seed1414_s125 /nvme/public/svln/runs/grpo_sfr_mix1k_seed1414/exports/step_0125
if [[ -f $R/exports/step_0125/pytorch_model.bin || -f $R/exports/step_0125/model.safetensors ]] || ls "$R"/exports/step_0125/*.safetensors > /dev/null 2>&1; then
  full_rxr rewind_sfr_mix1k_seed1414_s125 "$R/exports/step_0125"
else
  say "rewind_sfr_mix1k_seed1414 has no step-125 export: its RxR is not evaluated"
fi
full_rxr rewind_sfr_mix1k_seed2_s125 /nvme/public/svln/runs/rewind_sfr_mix1k_seed2/exports/step_0125
say "rxr_fix_final_evals finished"
