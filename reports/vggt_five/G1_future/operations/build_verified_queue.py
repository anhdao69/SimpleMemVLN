"""Seal completed qualification and create (never start) the two-stage queue."""
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
from transformers import AutoProcessor
from qwen_vl.train.vln_runtime import resolve_config
from qwen_vl.data.episode_serializer import EpisodeSerializer
from scripts.vln.queue_g3a1_then_g1_future import file_hash, source_files, atomic_json
B=Path('/mnt/data/vmo-ai-task/anhdh35')
G=B/'g1-future-worker2-20261019';R=B/'g3a1-resume-worker2-20261019'
S=G/'production/source-1fcc2af';OLD=R/'source/SimpleMemVLN-g3a1-877815e'
Q=B/'g3a1-then-g1-future-20261019'
assert (G/'final-checks.completed').is_file()
assert (G/'export-check.completed').is_file()
assert (R/'prepare.completed').is_file()
assert (R/'download.completed').is_file()
for name in ['checkpoint_audit.json','sampling_audit.json','image_check_parallel.json','dataset_equivalence.json']:
    assert json.loads((R/name).read_text())['passed'],name
assert json.loads((R/'production/train/checkpoint-2100/REBASE_RECORD.json').read_text())['preflight']['PREFLIGHT']=='PASS'
results={}
for name in ['mixed-b8','mixed-b4','terminal8','r2r-representative','rxr-representative','r2r-longest','rxr-longest','mixed-extreme','recovery-reference','recovery-resumed']:
    directory=G/'experiments'/name
    result=json.loads((directory/'result.json').read_text());assert result['passed'],name
    assert result['resources']['new_oom_events']==0,name
    hashes=json.loads((directory/'source_manifest.json').read_text())
    for path,digest in hashes.items():
        if path.startswith('src/'):assert file_hash(S/path)==digest,(name,path)
    results[name]=dict(result_sha256=file_hash(directory/'result.json'),seconds_per_update=result['steady_seconds'],peak_reserved_gib=result['check']['peak_reserved_gib'])
for path in [G/'recovery_check.json',G/'serving-export-smoke/g1_export_check.json',G/'config_comparison.json']:
    assert json.loads(path.read_text())['passed'],path
suites={}
for path in [G/'final-suite.xml',R/'gpu-suite.xml',G/'export-unit-suite.xml']:
    suites[str(path)]={}
    for suite in ET.parse(path).getroot().iter('testsuite'):
        assert int(suite.get('failures',0))==int(suite.get('errors',0))==0
        assert int(suite.get('skipped',0))==0,(path,'skipped tests')
        suites[str(path)].update(suite.attrib)
assert 'DISTRIBUTED_G1_REDUCTION_PASS' in (G/'logs/distributed-reduction.log').read_text()
atomic_json(G/'qualification.json',dict(passed=True,source=str(S),source_files=source_files(S),experiments=results,suites=suites,recovery_sha256=file_hash(G/'recovery_check.json'),export_sha256=file_hash(G/'serving-export-smoke/g1_export_check.json'),note='Smoke greedy output may remain invalid; export/replay/reset equivalence is mandatory.'))
shutil.copy2(R/'production/train/checkpoint-2100/REBASE_RECORD.json',R/'REBASE_RECORD.json')
model=B/'cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
cfg=resolve_config(S/'configs/episodic/science_common.yaml',S/'configs/vln_empty.yaml',S/'configs/vln_memory_window8.yaml',S/'configs/vggt/g1_joint_one_epoch.yaml',S/'configs/vggt/g1_future_h4.yaml')
nav=EpisodeSerializer(AutoProcessor.from_pretrained(str(model)),cfg).metadata()
assert nav['config']['training']['epochs']==1
common=dict(G1_CUDA_HOME=str(B/'cuda-12.9.1'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MAX_JOBS='2',PYTHONDONTWRITEBYTECODE='1')
def pins(paths):return {str(p):file_hash(p) for p in paths}
stages=[dict(name='g3a1',source=str(OLD),source_files=source_files(OLD),run=str(R/'production'),steps=3852,navigation=json.loads((R/'production/train/checkpoint-2100/navigation.json').read_text()),command=['bash',str(S/'train/resume_g3a1_worker2.sh')],environment=dict(common,SRC=str(OLD),RUN_DIR=str(R/'production'),MODEL=str(model)),pinned_files=pins([R/'production/data/r2r_rxr15_train.rebased.jsonl',R/'production/g3a1_resume_h4.yaml',R/'checkpoint_audit.json',R/'sampling_audit.json',R/'REBASE_RECORD.json',G/'qualification.json',S/'train/resume_g3a1_worker2.sh'])),dict(name='g1_future',source=str(S),source_files=source_files(S),run=str(G/'production/g1_future_joint_e1'),steps=3852,navigation=nav,command=['bash',str(S/'train/launch_g1_future_joint_e1.sh'),str(G/'production/g1_future_joint_e1')],environment=common,pinned_files=pins([B/'SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl',G/'qualification.json']))]
config=dict(root=str(Q),stages=stages)
Q.mkdir(exist_ok=True)
assert not (Q/'queue.json').exists(),'Never overwrite an existing queue'
atomic_json(Q/'queue.json',config)
print('QUALIFICATION_AND_QUEUE_READY',Q)
