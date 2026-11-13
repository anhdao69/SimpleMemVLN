#!/bin/bash
# LOCAL 2026-11-13. User: "Train these config in the 1k R2R that we test before where the CoRe score 68.7 ... Just finish
# training first. I will run eval later." Local share of the 8 CoRe ablations (the others run on gpu2, s137, s119), after
# the seed-1414 chain (GRPO seed 1414 + R2R eval, then CoRe seed 1414 + R2R eval) has finished. For each NAME:CONFIG:
# freeze; 2-step smoke on both GPUs (as the local runs, with MPS) with provenance (SFR e3ada114, pool a1f93b80) and
# step-0 parity with rewind_sfr (the CoRe 68.73 run); 125 steps under supervise_run.sh (both GPUs, as rewind_sfr);
# turn-share monitor LOG ONLY (margin 1.0); final state check. No evaluation; exports + last resumable state kept.
# Usage: setsid nohup bash /nvme/public/svln/analysis/paper/abl_local_chain.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
DEV=/nvme/public/SimpleMemVLN_dev RUNS=/nvme/public/svln/runs GREF=/nvme/public/svln/runs/rewind_sfr
T=/tmp/claude-1001/-nvme-public-SimpleMemVLN/dced37af-913d-422f-b0e4-f110c3584dd3/scratchpad/ablation
TOOLS=/nvme/public/svln/tools/ablation
QUEUE="abl_eta05_sfr_r2r1k:configs/rl/local/rewind_eta05_grpo_sfr_r2r1k.yaml abl_eta2_sfr_r2r1k:configs/rl/local/rewind_eta2_grpo_sfr_r2r1k.yaml"
export TRAIN_GPUS=0,1 MAX_RESTARTS=20
exec >> "$RUNS/ablation_chain.log" 2>&1
say() { echo "$(date -Is) [local] $*"; }
mkdir -p "$TOOLS" && cp "$T/step0_parity.py" "$TOOLS/"
say "local ablation chain start: queue $QUEUE; waiting for the seed-1414 chain"
while pgrep -u "$(id -u)" -f "[s]eed1414_chain.sh|[s]upervise_run.sh|[t]rain_grpo|[f]inal_evals.sh|[h]abitat_r2r" > /dev/null; do sleep 120; done
say "no chain, trainer or evaluation of ours is running"
helpers_down() { pkill -u "$(id -u)" -f "multiprocessing.spawn import spawn_main"; pkill -u "$(id -u)" -f "qwen_vl.rl.env_server"; sleep 10; }
for item in $QUEUE; do
  NAME=${item%%:*} CONFIG=${item#*:} R=$RUNS/${item%%:*}
  if [[ -e $R ]]; then say "$NAME exists: not touched; next"; continue; fi
  while pgrep -u "$(id -u)" -f "[s]upervise_run.sh|[t]rain_grpo|[f]inal_evals.sh" > /dev/null; do sleep 120; done
  FROZEN=$(bash "$DEV/ops/local/freeze_code.sh" "${NAME}_$(date +%Y%m%d_%H%M)") || { say "$NAME: freeze failed: chain ends"; exit 1; }
  say "$NAME: frozen $FROZEN (config $CONFIG sha256 $(sha256sum "$FROZEN/$CONFIG" | cut -c1-16))"
  SM=$RUNS/smoke_${NAME}_$(date +%Y%m%d_%H%M)
  say "$NAME: smoke start $SM"
  (cd "$FROZEN" && SMV_ROOT=$FROZEN CUDA_VISIBLE_DEVICES=$TRAIN_GPUS CONFIG=$CONFIG RUN=$SM \
    EXTRA="--max-steps 2 --export-steps 2 --save-every 1" timeout 5400 bash "$FROZEN/ops/local/train_rl.sh" > "$SM.log" 2>&1)
  rc=$?; helpers_down
  say "$NAME: smoke end rc=$rc: $(python3 -I -c "
import json
for line in open('$SM/metrics.jsonl').read().splitlines():
    r = json.loads(line)
    print({k: r[k] for k in r if k in ('step', 'success_rate', 'trained_per_rank', 'train_peak_gib', 'step_seconds', 'grad_norm', 'lr', 'rescues', 'rescue_success_rate', 'takeover_mean')})" 2>&1 | tr '\n' ' ')"
  if grep -q '"step": 1' "$SM/metrics.jsonl" 2> /dev/null && [[ -f $SM/exports/step_0002/rl_provenance.json ]] \
     && python3 -I -c "
import json, sys
p = json.load(open('$SM/exports/step_0002/rl_provenance.json'))
sys.exit(0 if p['base_weight_sha256'].startswith('e3ada114') and p['pool_sha256'].startswith('a1f93b80') else f'provenance {p}')" \
     && python3 -I "$TOOLS/step0_parity.py" "$SM" "$GREF" "$CONFIG"; then
    say "$NAME: smoke and step-0 parity passed; removing its weights and resumable state (logs, metrics, rollouts kept)"
    rm -rf "$SM/exports" "$SM/resumable"
  else
    say "$NAME: smoke or parity FAILED: not started (see $SM.log); chain ends"; exit 1
  fi
  echo "$(date -Is) run=$R code=$FROZEN config=$CONFIG" >> "$RUNS/LAUNCHES.txt"
  (cd "$FROZEN" && SMV_ROOT=$FROZEN CONFIG=$CONFIG RUN=$R setsid nohup bash "$FROZEN/ops/local/supervise_run.sh" > /dev/null 2>&1 < /dev/null &)
  say "$NAME: launched (125 steps, GPUs $TRAIN_GPUS)"
  until [[ -f $R/run_config.json ]]; do sleep 30; done
  setsid nohup python3 -I "$FROZEN/ops/local/drift_guard.py" "$R" --reference "$GREF" --r2r-only --margin 1.0 > /dev/null 2>&1 < /dev/null &
  say "$NAME: turn-share monitor started (log only; vs rewind_sfr on the same prompts)"
  until [[ -f $R/COMPLETE || -f $R/STOPPED ]]; do
    grep -q "supervise: giving up" "$R.log" 2> /dev/null && { say "$NAME: supervisor gave up; chain ends (nothing removed)"; exit 1; }
    sleep 120
  done
  while pgrep -u "$(id -u)" -f "[s]upervise_run.sh" > /dev/null; do sleep 20; done
  sleep 30; helpers_down
  say "$NAME ended: COMPLETE $(cat "$R/COMPLETE" 2> /dev/null) STOPPED $(cat "$R/STOPPED" 2> /dev/null); restarts $(grep -c 'supervise: train exited' "$R.log")"
  python3 -I - "$R" <<'PY' || { say "$NAME: final state check FAILED; chain ends (nothing removed)"; exit 1; }
import json, sys
from pathlib import Path
r = Path(sys.argv[1])
assert (r / 'COMPLETE').exists() and not (r / 'STOPPED').exists(), 'not COMPLETE'
e = r / 'exports' / 'step_0125'
prov = json.loads((e / 'rl_provenance.json').read_text())
assert prov['rl_step'] == 125 and (e / 'pytorch_model.bin').stat().st_size > 8e9, prov.get('rl_step')
print(f'final state ok: exports {sorted(p.name for p in (r / "exports").iterdir())}; resumable {sorted(p.name for p in (r / "resumable").iterdir())}')
PY
  say "TRAINED $NAME: $R/exports/step_0125 (sha256 $(sha256sum "$R/exports/step_0125/pytorch_model.bin" | cut -c1-16)); not evaluated"
done
say "local ablation chain finished"
