#!/bin/bash
# LOCAL 2026-11-12. User: "Is there any way to fix it in the training? so make sure the infer result will stable?" ->
# option 1. STABLE-V1 seed A (configs/rl/local/rewind_grpo_sfr_mix1k_stable_a.yaml: mix1k's config + adv_scale batch,
# linear_tail LR decay over the last 50 steps, exports every 5 steps from 65; seed 20261212) on both local GPUs:
#   0. waits for the soup evaluations (soups/soup_evals.log) to finish; runs the dev unit tests;
#   1. ops/local/hard2k_chain.sh (freeze, 2-step smoke with provenance SFR e3ada114 + pool 6ab04818, 125 steps with the
#      drift guard on the gpu2 Rewind 8k table (mix1k's prompts), R2R val_unseen of exports/step_0125, compares);
#      after its smoke, stable_check.py must find ONE batch advantage scale per step, else the run is stopped;
#   2. stable_check.py on the run (advantage scales + LR schedule), the uniform average of the 13 tail exports
#      (make_soup.py -> exports/avg_065_125), its R2R val_unseen, verify, compares.
# Usage: setsid nohup bash /nvme/public/svln/analysis/stability/stable_a_chain.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
DEV=/nvme/public/SimpleMemVLN_dev; ST=/nvme/public/svln/analysis/stability
RUNS=/nvme/public/svln/runs EVALS=/nvme/public/svln/evals CMP=/nvme/public/svln/compare
PY=/nvme/public/SimpleMemVLN_rl_s8n/.venv/bin/python
EVALUATOR=/nvme/public/svln/code/sfr_campaign_20261102_2034
export NAME=rewind_sfr_mix1k_stable_a CONFIG=configs/rl/local/rewind_grpo_sfr_mix1k_stable_a.yaml MILESTONES=125 \
  BASE_PREFIX=e3ada114 POOL_FILE=data/rl/mix_r2r1k_rxr1k.jsonl POOL_PREFIX=6ab04818 \
  TABLE=/nvme/public/svln/analysis/clean2k/rewind8k_turns_s790.tsv \
  BASES="sfr rewind_sfr_mix1k_s125 rewind_sfr_mix1k_seed2_s125 rewind_sfr_hard1k_s125" \
  LABEL="STABLE-V1 seed A, mix1k prompts from SFR"
R=$RUNS/$NAME
exec >> "$RUNS/stable_a_chain.log" 2>&1
say() { echo "$(date -Is) $*"; }
say "stable-A chain start"
until grep -q "soup evals finished" "$RUNS/soups/soup_evals.log" 2>/dev/null; do sleep 120; done
while pgrep -f "[f]inal_evals.sh|[h]abitat_r2r" > /dev/null; do sleep 60; done
say "soup evaluations finished; GPUs free of our evaluations"
(cd "$DEV" && PYTHONPATH=$DEV/src timeout 600 "$PY" -m pytest -q tests/rl/test_algos.py 2>&1 | tail -1 | sed "s/^/$(date -Is) dev tests: /")
bash "$DEV/ops/local/hard2k_chain.sh" &
chain=$!
until grep -qE "smoke passed|smoke FAILED|exists: this chain|not starting" "$R.chain.log" 2>/dev/null || ! kill -0 $chain 2>/dev/null; do sleep 30; done
if grep -q "smoke passed" "$R.chain.log"; then
  SM=$(ls -d "$RUNS"/smoke_${NAME}_* | tail -1)
  if python3 -I "$ST/stable_check.py" "$SM" > "$SM.stable_check.txt" 2>&1; then
    say "smoke stable_check OK: $(tail -2 "$SM.stable_check.txt" | head -1)"
  else
    say "smoke stable_check FAILED ($(tail -1 "$SM.stable_check.txt")): stopping the run"
    until [[ -f $R/run_config.json ]]; do sleep 20; done
    RUN=$R REASON="STABLE-V1 smoke check failed: advantages not batch-scaled" bash "$DEV/ops/local/stop_run.sh"
  fi
fi
wait $chain
say "hard2k_chain ended: $(tail -1 "$R.chain.log" | cut -c1-200)"
[[ -f $R/COMPLETE ]] || { say "$NAME has no COMPLETE; chain ends (nothing removed)"; exit 1; }
python3 -I "$ST/stable_check.py" "$R" --lr 1e-6 --total 125 --decay 50 | tail -3 | sed "s/^/$(date -Is) run check: /"
SNAPS=$(for s in $(seq 65 5 125); do printf '%s ' "$R/exports/step_$(printf %04d "$s")"; done)
for d in $SNAPS; do [[ -s $d/pytorch_model.bin ]] || { say "missing tail export $d"; exit 1; }; done
AVG=$R/exports/avg_065_125
[[ -d $AVG ]] || "$PY" -I "$ST/make_soup.py" "$AVG" $SNAPS | sed "s/^/$(date -Is) average: /"
[[ -s $AVG/pytorch_model.bin ]] || { say "average failed"; exit 1; }
EP=${NAME}_avg
bash "$EVALUATOR/ops/local/final_evals.sh" "$EP=$AVG"
w=$(sha256sum "$AVG/pytorch_model.bin" | cut -d' ' -f1)
python3 -I "$EVALUATOR/scripts/rl/s8n_analysis/verify_eval.py" "$EVALS/${EP}_r2r_val_unseen" "$EVALS/base_r2r_val_unseen" "$w" | head -4 | sed "s/^/$(date -Is) verify $EP: /"
for b in sfr rewind_sfr_mix1k_s125 rewind_sfr_mix1k_seed2_s125 ${NAME}_s125; do
  out=$CMP/${b}_vs_${EP}_r2r_val_unseen.json
  (cd "$EVALUATOR" && $PY scripts/rl/compare_evals.py --base "$EVALS/${b}_r2r_val_unseen/merged" --new "$EVALS/${EP}_r2r_val_unseen/merged" \
     --name-base "$b" --name-new "$EP" --out "$out" > /dev/null) && python3 -c "
import json; d = json.load(open('$out')); m = d['metrics']; p = d['paired_success']
print('$(date -Is) compare $b -> $EP: SR %.2f -> %.2f (%+.2f), SPL %+.2f, only_base %d only_new %d, McNemar p %.3g'
      % (100 * m['SR']['base'], 100 * m['SR']['new'], 100 * m['SR']['abs_delta'], 100 * m['SPL']['abs_delta'], p['only_base'], p['only_new'], p['mcnemar_exact_p']))"
done
say "stable-A chain finished"
