"""Require H100 smoke acceptance before constructing the replacement queue."""
import json,math,os,statistics,xml.etree.ElementTree as ET
from pathlib import Path
from transformers import AutoProcessor
from qwen_vl.train.vln_runtime import resolve_config
from qwen_vl.data.episode_serializer import EpisodeSerializer
from qwen_vl.train.recovery import validate_complete,latest_checkpoint
from scripts.vln.profile_episodic import verify
from scripts.vln.queue_g3a1_then_g3_fixed import source_files,file_hash,atomic_json
B=Path('/mnt/data/vmo-ai-task/anhdh35');G=B/'g3-fixed01-seed430-worker2-20261019';R=B/'g3a1-resume-worker2-20261019'
S=Path(os.environ['G1_CODE_DIR']);Q=G/'queue';train=G/'smoke/train'
for suite in ET.parse(G/'gpu-suite.xml').getroot().iter('testsuite'):
 assert all(int(suite.get(k,0))==0 for k in ('failures','errors','skipped'))
check=verify(train,4,2,8,78);assert check['passed'],check
resource=json.loads((G/'smoke/supervisor/resources.json').read_text());assert resource['status']=='PASS' and resource['new_oom_events']==0
rows=[json.loads(line) for rank in range(4) for line in (train/f'exposures_rank{rank}.jsonl').read_text().splitlines()]
assert len(rows)==16 and all(x['aux_weight']==.1 and math.isfinite(x['aux_loss_sum']) for x in rows)
marker=validate_complete(train/'checkpoint-2');assert marker['step']==2 and marker['world_size']==4
import torch
state=torch.load(train/'checkpoint-2/pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
bias_updates={k:int(v.count_nonzero()) for k,v in state.items() if k.startswith('research_aux.') and k.endswith('bias')}
assert sum(bias_updates.values())>0,'Readout biases did not receive an optimizer update'
del state
cfg=json.loads((train/'resolved_config.json').read_text());assert cfg['research']['ramp_updates']==0 and cfg['research']['lambda_max']==.1
assert cfg['research']['sample_seed']==cfg['research']['head_seed']==cfg['training']['seed']==430
qualification=dict(passed=True,source=str(S),source_files=source_files(S),check=check,resources=resource,exposures=len(rows),constant_aux_weight=.1,seed=430,recovery_files=len(marker['files']),head_bias_updates=bias_updates)
atomic_json(G/'qualification.json',qualification)
model=B/'cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
nav=EpisodeSerializer(AutoProcessor.from_pretrained(str(model)),cfg).metadata()
old_config=json.loads((B/'g3a1-then-g1-future-20261019/queue.json').read_text());first=old_config['stages'][0]
first['command']=['bash',str(S/'train/resume_g3a1_worker2.sh')]
first['pinned_files']={p:h for p,h in first['pinned_files'].items() if '/g1-future-worker2-' not in p}
first['pinned_files'][str(G/'qualification.json')]=file_hash(G/'qualification.json')
first['pinned_files'][str(S/'train/resume_g3a1_worker2.sh')]=file_hash(S/'train/resume_g3a1_worker2.sh')
# The old predecessor source/checkpoint identity remains unchanged.
current=latest_checkpoint(R/'production/train');assert current is not None
assert json.loads((current/'navigation.json').read_text())==first['navigation']
assert not (R/'production/train/REQUEST_CHECKPOINT_AND_STOP').exists()
common=dict(G1_CUDA_HOME=str(B/'cuda-12.9.1'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MAX_JOBS='2',PYTHONDONTWRITEBYTECODE='1')
stage=dict(name='g3_fixed',source=str(S),source_files=source_files(S),run=str(G/'production'),steps=3852,navigation=nav,command=['bash',str(S/'train/launch_g3_fixed01_seed430_joint_e1.sh'),str(G/'production')],environment=common,pinned_files={str(p):file_hash(p) for p in [G/'qualification.json',B/'SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl']})
Q.mkdir(exist_ok=True);assert not (Q/'queue.json').exists()
atomic_json(Q/'queue.json',dict(root=str(Q),stages=[first,stage]))
print('G3_FIXED_QUALIFICATION_AND_QUEUE_PASS',current,flush=True)
