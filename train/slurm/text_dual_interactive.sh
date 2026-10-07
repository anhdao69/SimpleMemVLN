#!/usr/bin/env bash
# Run inside an existing four-GPU Slurm step with the pinned environment active.
# Usage: bash train/slurm/text_dual_interactive.sh RUN_ROOT MODEL_PATH MANIFEST
set -euo pipefail
run_root=$(realpath "$1")
model_path=$(realpath "$2")
manifest=$(realpath "$3")
cd "$run_root/source"
export PYTHONPATH="$PWD/src:$PWD"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python - "$run_root" "$manifest" <<'PY'
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys
import torch,yaml
from qwen_vl.contracts import validate_config
import qwen_vl.stream.window_attention as attention
root=Path(sys.argv[1]); manifest=Path(sys.argv[2])
assert Path(attention.__file__).resolve().is_relative_to(Path.cwd()), 'Wrong source imported'
assert os.environ.get('SLURM_JOB_ID'), 'An existing Slurm allocation is required'
assert torch.cuda.device_count()==4, 'Exactly four visible GPUs required'
q=json.loads((root/'qualification.json').read_text())
assert q['passed'] and q['source_commit']==subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
for name,expected in q['source_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==q['manifest_sha256']
c=validate_config(yaml.safe_load(Path('configs/vln_dual_window8_text_joint.yaml').read_text()),4)
assert c['model']['output_mode']=='qwen_text' and c['model']['step_lane']['enabled']
assert c['memory']['mode']=='window8' and c['observations']['append_action_tokens']
assert (c['training']['epochs'],c['training']['expected_total_steps'],c['training']['warmup_steps'])==(2,7704,232)
assert not (root/'train').exists(), 'Fresh output directory required'
assert shutil.disk_usage(root).free>400*2**30, 'Insufficient checkpoint headroom'
assert sum(1 for line in manifest.open() if line.strip())==30815
(root/'empty.yaml').write_text('{}\n')
print('TEXT_DUAL_PRODUCTION_PREFLIGHT_PASS',flush=True)
PY
python -m scripts.vln.run_step_lane_gate \
  --host-guard-gib 700 --out "$run_root/supervisor" -- \
  torchrun --standalone --nproc_per_node=4 -m qwen_vl.train.train_qwen \
  --vln_config configs/vln_dual_window8_text_joint.yaml \
  --output_config "$run_root/empty.yaml" --manifest "$manifest" \
  --model_name_or_path "$model_path" --output_dir "$run_root/train" \
  --run-name window8_text_dual_from_base_e2 --recovery-save-steps 250 \
  --deepspeed deepspeed.json
