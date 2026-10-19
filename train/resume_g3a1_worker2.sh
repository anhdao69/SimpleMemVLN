# Worker-2 continuation of verified HF checkpoint2100. One extra recovery save at2110; science unchanged.
#!/usr/bin/env bash
# Continue G3A1 from the run directory prepared by tools/prepare_resume.py on a 4 x H100 node.
# Same command as the original newton/worker-2 launchers: host-memory supervisor, 4 ranks, GAS 2,
# recovery saves every 100 updates, --resume_from_checkpoint auto (newest COMPLETE checkpoint).
#
#   SRC=/YOUR/src/SimpleMemVLN-g3a1-877815e RUN_DIR=/YOUR/g3a1_joint_e1 \
#   MODEL=/YOUR/.../snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a bash tools/resume_g3a1_h4.sh
#
# ENV_WRAPPER sets CUDA_HOME, the venv PATH and PYTHONPATH (incl. vggt-omega) then execs its args;
# default is the source tree's worker-2 wrapper train/g1_omega_env.sh (export G1_CUDA_HOME first).
set -euo pipefail
[[ "${SLURMD_NODENAME:-$(hostname)}" = worker-2 ]] || { echo "worker-2 only" >&2; exit 2; }
: "${SLURM_JOB_ID:?Run inside a four-H100 Slurm allocation}"
SRC="${SRC:?extracted training source (SimpleMemVLN-g3a1-877815e)}"
RUN_DIR="${RUN_DIR:?run directory prepared by tools/prepare_resume.py}"
MODEL="${MODEL:?Qwen3.5-4B snapshot 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}"
MANIFEST="${MANIFEST:-$RUN_DIR/data/r2r_rxr15_train.rebased.jsonl}"
PROFILE="${PROFILE:-$RUN_DIR/g3a1_resume_h4.yaml}"
ENV_WRAPPER="${ENV_WRAPPER:-$SRC/train/g1_omega_env.sh}"
HOST_GUARD_GIB="${HOST_GUARD_GIB:-691.2}"   # 0.9 x job RAM (691.2 for 768G, 460.8 for 512G)
for path in "$SRC" "$RUN_DIR" "$MODEL" "$MANIFEST" "$PROFILE"; do
  [[ "$path" = /* && -e "$path" ]] || { echo "absolute existing path required: $path" >&2; exit 2; }
done
[[ "$(cat "$SRC/source.commit")" == 877815e97e46a8bcc37cafee919b6348c18d0f35 ]] || { echo 'wrong source tree' >&2; exit 2; }
[[ -f "$RUN_DIR/train/checkpoint-2100/REBASE_RECORD.json" || -n "$(ls -d "$RUN_DIR"/train/checkpoint-* 2>/dev/null | grep -v checkpoint-2100)" ]] \
  || { echo 'run dir not prepared: run tools/prepare_resume.py first' >&2; exit 2; }
[[ "$(nvidia-smi -L | wc -l)" == 4 ]] || { echo 'Exactly four visible H100s required (world size 4)' >&2; exit 2; }
export G1_CODE_DIR="$SRC" G3A1_CODE_DIR="$SRC"
SUPERVISOR="$RUN_DIR/supervisor-$(date -u +%Y%m%dT%H%M%S)-$$"
exec flock -n "$RUN_DIR/launcher.lock" \
 bash "$ENV_WRAPPER" python -m scripts.vln.run_episodic_gate \
 --out "$SUPERVISOR" --host-guard-gib "$HOST_GUARD_GIB" --disk-path "$RUN_DIR" --min-free-disk-gib 220 -- \
 torchrun --standalone --nproc_per_node=4 -m qwen_vl.train.train_qwen \
 --vln_config configs/episodic/science_common.yaml --output_config configs/vln_empty.yaml \
 --memory_config configs/vln_memory_window8.yaml --policy_config configs/vggt/g3a1_joint_one_epoch.yaml \
 --runtime_profile "$PROFILE" --deepspeed configs/deepspeed_zero2_cpu_optimizer.json \
 --manifest "$MANIFEST" --model_name_or_path "$MODEL" \
 --output_dir "$RUN_DIR/train" --run-name g3a1_omega_newton_h4_joint_e1 \
 --recovery-save-steps 100 --first-recovery-save-step 2110 --resume_from_checkpoint auto
