#!/bin/bash
# 2026-11-12, user: "Deeply analysis why the running is not stable ... propose a solution ... SR stable around 71-72".
# R2R val_unseen of two weight averages (model soups) of RL exports from SFR, with the evaluator tree of every journal,
# then verify + paired compares with each ingredient, mix1k and SFR.
set -uo pipefail
EVALUATOR=/nvme/public/svln/code/sfr_campaign_20261102_2034
EVALS=/nvme/public/svln/evals CMP=/nvme/public/svln/compare PY=/nvme/public/SimpleMemVLN_rl_s8n/.venv/bin/python
SOUPS=/nvme/public/svln/runs/soups LOG=$SOUPS/soup_evals.log
exec >> "$LOG" 2>&1
say() { echo "$(date -Is) $*"; }
compare() {
  local b=$EVALS/$1_r2r_val_unseen/merged n=$EVALS/$2_r2r_val_unseen/merged out=$CMP/$1_vs_$2_r2r_val_unseen.json
  (cd "$EVALUATOR" && $PY scripts/rl/compare_evals.py --base "$b" --new "$n" --name-base "$1" --name-new "$2" --out "$out" > /dev/null) || { say "compare failed $1 / $2"; return; }
  python3 -c "
import json; d = json.load(open('$out')); m = d['metrics']; p = d['paired_success']
print('$(date -Is) compare $1 -> $2: SR %.2f -> %.2f (%+.2f), SPL %+.2f, only_base %d only_new %d, McNemar p %.3g'
      % (100 * m['SR']['base'], 100 * m['SR']['new'], 100 * m['SR']['abs_delta'], 100 * m['SPL']['abs_delta'], p['only_base'], p['only_new'], p['mcnemar_exact_p']))"
}
for s in soup_mix1k_seed12 soup_rewind4; do
  bash "$EVALUATOR/ops/local/final_evals.sh" "$s=$SOUPS/$s"
  w=$(sha256sum "$SOUPS/$s/pytorch_model.bin" | cut -d' ' -f1)
  python3 -I "$EVALUATOR/scripts/rl/s8n_analysis/verify_eval.py" "$EVALS/${s}_r2r_val_unseen" "$EVALS/base_r2r_val_unseen" "$w" | head -4 | sed "s/^/$(date -Is) verify $s: /"
  for b in sfr rewind_sfr_mix1k_s125 rewind_sfr_mix1k_seed2_s125 rewind_path_sfr_mix1k_s125 rewind_sfr_hard1k_s125; do compare "$b" "$s"; done
  say "SOUP DONE $s"
done
say "soup evals finished"
