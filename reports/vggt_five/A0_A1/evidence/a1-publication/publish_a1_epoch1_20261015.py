import os,json,sys,time,hashlib,shutil,fcntl
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ.pop('HF_HUB_OFFLINE',None)
base=Path('/mnt/data/vmo-ai-task/anhdh35')
run=base/'a1-omega-worker2-20261014-repair/a1_joint_e1'
code=base/'a1-omega-worker2-20261014-repair/source-6e71778'
sys.path[:0]=[str(code/'src'),str(code)]
staging=run/'publication-a1-e1';staging.mkdir(exist_ok=True)
lock=(staging/'upload.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
import torch
torch.set_num_threads(2)
from qwen_vl.train.recovery import validate_complete
from qwen_vl.research.vggt_g1 import serving_config,serving_state_dict
from qwen_vl.research.geoaddress_contract import parse_geoaddress
from scripts.vln.queue_geoaddress import source_files, check_qualification
repo='anhdao69/Streaming_VGGT';api=HfApi()
assert api.whoami()['name']=='anhdao69'
info=api.model_info(repo);parent=info.sha
assert not any(x.rfilename.startswith('A1/') for x in info.siblings),'A1 already exists; inspect before replacing'
resources=json.loads((sorted(run.glob('supervisor-*'))[-1]/'resources.json').read_text())
assert resources['status']=='PASS' and resources['exit_code']==0
assert resources['new_oom_events']==0 and resources['new_oom_kill_events']==0
check_qualification(code,run.parent/'qualification/qualification_pass.json',arm='A1')
checkpoint=run/'train/checkpoint-3852';final=run/'train/final'
complete=validate_complete(checkpoint)
assert complete['step']==3852 and complete['world_size']==4
trainer=json.loads((checkpoint/'trainer_state.json').read_text())
assert trainer['global_step']==3852 and trainer['epoch']>=.999
metadata=json.loads((final/'navigation.json').read_text())
assert metadata['config']['research']['run_id']=='A1'
assert parse_geoaddress(metadata['config']) is not None
assert metadata['serializer']=='vln_append_only_chat_geoaddress_v1'
state=torch.load(final/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
ckpt=torch.load(checkpoint/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
assert state.keys()==ckpt.keys()
prefix='backbone.model.language_model.layers.24.linear_attn.geoaddress.'
expected={prefix+n+'.weight':shape for n,shape in [('q_down',(8,2560)),('k_down',(8,2560)),('q_up',(256,8)),('k_up',(256,8))]}
assert {k:tuple(v.shape) for k,v in state.items() if '.geoaddress.' in k}==expected
assert not any(k.startswith('research_aux.') for k in state)
print('CHECKPOINT_VALIDATED',len(state),'tensors; checking exact final weights and finiteness',flush=True)
for i,(key,value) in enumerate(state.items()):
 assert value.dtype==ckpt[key].dtype and value.shape==ckpt[key].shape,key
 assert torch.equal(value,ckpt[key]),key
 assert torch.isfinite(value).all().item(),key
 if i%100==0:print('CHECKED_TENSORS',i,flush=True)
assert all(state[prefix+n+'.weight'].count_nonzero().item()>0 for n in ['q_up','k_up'])
assert serving_state_dict(state).keys()==state.keys()
export=staging/'A1/epoch-1';export.mkdir(parents=True,exist_ok=True)
allowed=['pytorch_model.bin','config.json','processor_config.json','preprocessor_config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja','generation_config.json','export_info.json','reload_reference.json']
for name in allowed:
 src=final/name;dst=export/name
 if src.exists() and not dst.exists():os.link(src,dst)
research=metadata['config']['research'];metadata['config']=serving_config(metadata['config'])
assert 'research' not in metadata['config'] and parse_geoaddress(metadata['config']) is not None
(export/'navigation.json').write_text(json.dumps(metadata,indent=2)+'\n')
(export/'research_training.json').write_text(json.dumps(dict(run_id='A1',research=research,source_commit=(code/'source.commit').read_text().strip()),indent=2)+'\n')
prov=staging/'A1/provenance';prov.mkdir(parents=True,exist_ok=True)
for name in ['resolved_config.json','run_timing.json','train_results.json','trainer_state.json']:
 shutil.copy2(run/'train'/name,prov/name)
shutil.copy2(final/'navigation.json',prov/'training_navigation.json')
shutil.copy2(code/'source.commit',prov/'source.commit')
(prov/'source_manifest.json').write_text(json.dumps(source_files(code),indent=2)+'\n')
shutil.copy2(run.parent/'qualification/qualification_pass.json',prov/'qualification_pass.json')
for p in (run/'train').rglob('epoch-1.json'):
 shutil.copy2(p,prov/'epoch-1.json');break
validation=dict(passed=True,epoch=1,step=3852,world_size=4,checkpoint_files=len(complete['files']),serving_state_tensors_exact=True,serving_tensors_finite=True,geoaddress_adapter_parameters=45056,geoaddress_adapter_preserved=True,supervisor_status='PASS',new_oom_events=resources['new_oom_events'],source_commit=(code/'source.commit').read_text().strip(),validation_device='CPU',note='All final tensor values equal complete checkpoint3852. Serving weight file hardlinked unchanged; only research configuration removed. No new GPU navigation evaluation.')
(prov/'publication_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
(prov/'supervisor_resources.json').write_text(json.dumps(resources,indent=2)+'\n')
readme=Path(hf_hub_download(repo,'README.md',revision=parent)).read_text()
readme+="\n## A1 — online VGGT-Omega native Q/K relation supervision, joint epoch 1\n\n`A1/epoch-1/` contains A1 independently trained from original Qwen3.5-4B\nbase for one joint R2R + English RxR15 epoch: 30,815 episodes, 3,852 updates,\nseed429, global batch8 on four H100s. One-epoch cosine schedule with minimum\nLR ratio0.1,116 warmup updates, backbone peak LR5e-6 and Q/K adapter peak LR5e-5.\n\nA1 uses frozen resident FP32 online VGGT-Omega-1B-512 during training, with\n**direct native Q/K relation KL supervision**, no PCA, no pre-extracted features,\nand no dimension-conversion MLP. Auxiliary weight ramps to0.1 over771 updates.\nThe policy uses Window8, canonical text action history and full episode BPTT.\nImage-token-only rank8 Q/K adapters at native GDN decoder24, raw heads0/1,\nadd45,056 parameters. These trained adapters are retained and **required at\ninference**. VGGT-Omega is not required for inference.\nSerializer: `vln_append_only_chat_geoaddress_v1`.\n\n```python\nfrom huggingface_hub import snapshot_download\nfrom qwen_vl.train.vln_runtime import load_checkpoint\nsnapshot = snapshot_download('anhdao69/Streaming_VGGT', allow_patterns=['A1/epoch-1/*'])\nmodel, serializer = load_checkpoint(snapshot + '/A1/epoch-1', base_model_path)\n```\n\nUse the GeoAddress-capable SimpleMemVLN wrapper and pinned environment, not\nan unmodified AutoModel or the older G1/G3-only runtime. Training source:\nhttps://github.com/anhdao69/SimpleMemVLN/tree/6e717787313b2218cd2fd40b059f2942fee1820f\nBase revision: Qwen/Qwen3.5-4B at851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a.\n\nEvery serving tensor is finite and bitwise identical to complete checkpoint3852.\nOnly training research configuration is removed; all policy weights are retained.\nAll16 qualification gates passed before production. Training supervisorPASS,\nzero new OOM events. Optimizer/RNG recovery checkpoints remain on the cluster.\nProvenance and publication validation are in `A1/provenance/`. This publication\ndoes not establish Habitat navigation quality; evaluation is separate.\n"
(staging/'README.md').write_text(readme)
files={str(p.relative_to(staging)):p for p in sorted((staging/'A1').rglob('*')) if p.is_file()}
files['README.md']=staging/'README.md'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
checksums=json.loads(Path(hf_hub_download(repo,'SHA256SUMS.json',revision=parent)).read_text())
local_hashes={name:sha(path) for name,path in files.items()}
checksums.update(local_hashes)
(staging/'SHA256SUMS.json').write_text(json.dumps(checksums,indent=2,sort_keys=True)+'\n')
files['SHA256SUMS.json']=staging/'SHA256SUMS.json'
local_hashes['SHA256SUMS.json']=sha(staging/'SHA256SUMS.json')
print('UPLOAD_START',len(files),'files',sum(p.stat().st_size for p in files.values()),flush=True)
commit=api.create_commit(repo_id=repo,repo_type='model',parent_commit=parent,operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=str(p)) for name,p in sorted(files.items())],commit_message='Publish verified A1 online Omega relation-supervised joint epoch 1',num_threads=2)
(staging/'upload_commit.json').write_text(json.dumps(dict(repo=repo,revision=commit.oid),indent=2))
print('UPLOAD_COMMITTED',commit.oid,flush=True)
remote={f.rfilename:f for f in api.model_info(repo,revision=commit.oid,files_metadata=True).siblings}
verified=[]
for name,expected_hash in local_hashes.items():
 obj=remote[name];assert obj.size==files[name].stat().st_size,name
 if obj.lfs:
  assert obj.lfs.sha256==expected_hash,name
  method='Hub LFS SHA256 and byte size'
 else:
  actual=Path(hf_hub_download(repo,name,revision=commit.oid))
  assert sha(actual)==expected_hash,name
  method='Downloaded content SHA256'
 verified.append(dict(path=name,sha256=expected_hash,method=method))
 print('VERIFIED',name,flush=True)
result=dict(passed=True,repo=repo,revision=commit.oid,folder='A1/epoch-1',files=verified,bytes=sum(p.stat().st_size for p in files.values()),verified_unix=time.time())
(staging/'UPLOAD_VERIFIED.json').write_text(json.dumps(result,indent=2)+'\n')
print('UPLOAD_VERIFIED',repo,commit.oid,flush=True)
