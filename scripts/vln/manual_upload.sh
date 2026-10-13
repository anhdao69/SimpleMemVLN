#!/usr/bin/env bash
# Publish a COMPLETE attempt's epoch-1 export + full training state to the Hub, by hand.
# Needed for attempts launched from source a340e13, whose in-job upload runs offline
# (fixed in 5975e8b). Run on a node with internet (login node is fine):
#   bash scripts/vln/manual_upload.sh <attempt_dir>
set -euo pipefail
A=$(readlink -f "${1:?attempt dir}")
L="$A/launch_manifest.json"
field() { python3 -c "import json,sys;print(json.load(open(sys.argv[1]))$2)" "$1"; }
run=$(field "$L" "['run_id']"); attempt=$(field "$L" "['attempt']"); src=$(field "$L" "['source_dir']")
cm="$(field "$L" "['campaign_root']")/campaign_manifest.json"
env_dir=$(field "$cm" "['env_dir']"); repo=$(field "$cm" "['hf_repo']"); hf_home=$(field "$cm" "['hf_home']")
T="$A/train"
ck=$(ls -d "$T"/checkpoint-3852 2>/dev/null) || { echo "no checkpoint-3852 in $T"; exit 2; }
[ -f "$ck/RECOVERY_COMPLETE.json" ] || { echo "checkpoint-3852 not complete"; exit 2; }
export_dir="$ck"
if python3 -c "import json,sys;sys.exit(0 if json.load(open('$T/final/export_info.json')).get('global_step')==3852 else 1)" 2>/dev/null; then
  export_dir="$T/final"
fi
cd "$src"
env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_HOME="$hf_home" PYTHONPATH="$src/src:$src" \
  "$env_dir/bin/python" -m scripts.vln.upload_episodic --repo "$repo" --folder "$run/$attempt" \
  --checkpoint "$ck" --export "$export_dir" --attempt-dir "$A" 2>&1 | tee "$A/upload_manual.log"
