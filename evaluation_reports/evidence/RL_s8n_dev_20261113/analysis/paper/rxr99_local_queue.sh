#!/bin/bash
# LOCAL 2026-11-13: after CoRe run 1's 99-episode RxR re-evaluation, run GRPO run 1's (same script, GPU 1, no MPS).
while pgrep -u "$(id -u)" -f "[r]xr99_eval.sh rewind_sfr_mix1k_s125" > /dev/null; do sleep 60; done
setsid nohup bash /nvme/public/svln/analysis/paper/rxr99_eval.sh grpo_sfr_mix8k_aplus_s125 /nvme/public/svln/runs/grpo_sfr_mix8k_aplus/exports/step_0125 > /dev/null 2>&1 < /dev/null &
