"""CPU-only completed-M5-one-epoch export, immutable publication, and integrity audit."""
import os, sys, json, hashlib, shutil, datetime
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'):os.environ.pop(key,None)
base=Path('/mnt/data/vmo-ai-task/anhdh35')
campaign=base/'m5-one-epoch-worker2-20261018'
source=campaign/'source-f4d1c2f'
run=campaign/'production/m5_joint_e1'
root=campaign/'publication-m5-epoch1-20261019'
root.mkdir(exist_ok=True)
export=root/'epoch-1'
prefix='M5_E1'
repo='anhdao69/Streaming_VGGT'
sys.path[:0]=[str(source/'src'),str(source)]

def write(path,data):
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

sys.path.insert(0,str(campaign/'operations'))
from controller import verify_qualification, m5_complete, contract
from qwen_vl.train.recovery import validate_complete
seal=campaign/'qualification/QUALIFIED.json'
verify_qualification()
assert m5_complete(run)
checkpoint=run/'train/checkpoint-3852'
record=validate_complete(checkpoint)
assert record['step']==3852 and record['world_size']==4 and record['epoch_boundary']
trainer=json.loads((checkpoint/'trainer_state.json').read_text())
assert trainer['global_step']==3852 and trainer['epoch']==1 and trainer['max_steps']==3852
training=json.loads((checkpoint/'navigation.json').read_text())
assert training['config']==contract()['config'] and 'research' not in training['config']
supervisor=sorted(run.glob('supervisor-*'))[-1]
resources=json.loads((supervisor/'resources.json').read_text())
assert resources['status']=='PASS' and resources['new_oom_events']==resources['new_oom_kill_events']==0
print('COMPLETED_M5_CHECKPOINT_VERIFIED',flush=True)

# Plain M5 has no teacher or auxiliary heads; publish an exact serving copy.
serving_files={'pytorch_model.bin','navigation.json','config.json',
    'generation_config.json','preprocessor_config.json','processor_config.json',
    'tokenizer_config.json','tokenizer.json','chat_template.jinja'}
if not export.exists():
    export.mkdir()
    for p in checkpoint.iterdir():
        if p.is_file() and p.name in serving_files:shutil.copy2(p,export/p.name)
import torch
assert not torch.cuda.is_available()
metadata=json.loads((export/'navigation.json').read_text())
assert metadata==training
assert metadata['config']['memory']==training['config']['memory']
assert 'research' not in metadata['config']
before=torch.load(checkpoint/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
after=torch.load(export/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
assert not any(k.startswith('research_aux.') for k in before)
keys=set(before)
assert set(after)==keys and all(torch.equal(after[k],before[k]) for k in keys)
removed=sorted(set(before)-set(after))
assert not removed
print('ALL_SERVING_TENSORS_BITWISE_EQUAL',len(keys),flush=True)

# Construct the real navigation architecture on meta, then strictly assign the
# mmap-backed CPU tensors. No GPU allocations and no second full model copy.
from transformers import AutoConfig,AutoProcessor,Qwen3_5ForConditionalGeneration
from qwen_vl.models.nav_model import SimpleMemVLNForNavigation
from qwen_vl.data.episode_serializer import EpisodeSerializer
model_path=base/'cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
config=AutoConfig.from_pretrained(model_path,local_files_only=True)
with torch.device('meta'):
    backbone=Qwen3_5ForConditionalGeneration(config)
    model=SimpleMemVLNForNavigation(backbone,metadata['config'])
model.load_state_dict(after,strict=True,assign=True)
assert not hasattr(model,'research_aux')
assert all(p.device.type=='cpu' for p in model.parameters())
processor=AutoProcessor.from_pretrained(export,local_files_only=True)
assert EpisodeSerializer(processor,metadata['config']).metadata()==metadata
audit=dict(passed=True,cpu_only=True,strict_architecture_reload=True,
           exact_serving_tensors=len(keys),removed_training_tensors=removed,
           serializer_metadata_exact=True,memory_metadata_exact=True,
           completed_model_gpu_replay=False,
           qualification='qualification/QUALIFIED.json')
write(root/'export_validation.json',audit)
del model,backbone,before,after
print('CPU_STRICT_EXPORT_VALIDATED',flush=True)

provenance=root/'provenance';provenance.mkdir(exist_ok=True)
for src,name in [(run/'train/reports/epoch-1.json','epoch-1.json'),
    (run/'train/resolved_config.json','resolved_config.json'),
    (run/'train/run_timing.json','run_timing.json'),
    (checkpoint/'trainer_state.json','trainer_state.json'),
    (supervisor/'resources.json','completion_resources.json'),
    (root/'export_validation.json','publication_validation.json'),
    (seal,'qualification_pass.json'),
    (run/'run_identity.json','run_identity.json')]:
    shutil.copy2(src,provenance/name)
if not (provenance/'source_and_completion.json').exists():
    write(provenance/'source_and_completion.json',dict(
        source_commit='f4d1c2f216e68a960a5a691d21cbf1f08ea4a901',
        source_directory=str(source),checkpoint=str(checkpoint),completed_epoch=1,
        optimizer_updates=3852,global_batch=8,supervised_layers=[],teacher=None,
        recovery_inventory=record,export_source_weights_sha256=sha(checkpoint/'pytorch_model.bin'),
        export_weights_sha256=sha(export/'pytorch_model.bin'),
        verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
(root/'README.md').write_text('''# Plain M5 — completed epoch 1 with G3-matched one-epoch decay

`epoch-1/` contains the completed plain-M5 navigation model: 3,852 optimizer
updates, one joint R2R + English RxR15deg pass (30,815 episodes), initialized
from original Qwen3.5-4B revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

This is the matched one-epoch-schedule control, not epoch 1 of the older
two-epoch schedule. Training used no VGGT teacher, no G3 supervision,
no auxiliary heads, no PCA and no preextracted teacher features.
It preserves M5 R4/B1200 committed-mean-L2 episodic memory, canonical text
action history, causal state handling and full-episode BPTT.

Four H100s on worker-2; global batch eight (one episode/rank, accumulation two),
seed 429. One-epoch cosine decay, 116 warmup updates, final LR ratio 0.1;
backbone LR 5e-6, weight decay 0.01, Adam betas 0.9/0.95, gradient clip 1.
Runtime matches optimized G3: ZeRO2 CPU optimizer offload, CPUAdam, pinned
activation arena, BF16 staged pixels, vision batch eight, attention batch 32,
eight persistent data workers per rank. Exact configuration is in provenance.

All 3,852 updates completed with action-weighted training loss
0.12657747317027537 and no GPU/host OOM events. Full wall time including
model load and checkpoint saves was 48,849.65 seconds (13h34m10s).
Training loss is not navigation evaluation accuracy.

The serving files are copied without modifying the checkpoint's model state
or navigation configuration. Every serving tensor is bitwise identical, and
the real navigation model strictly reloads on CPU with exact serializer and
memory metadata. No completed-model GPU replay is claimed. The earlier
qualification covered mixed R2R/RxR, longest RxR repetitions and full
checkpoint recovery; its source-bound seal is included in provenance.

Use SimpleMemVLN branch `c2_vggt_omega_g3_m5_layers`, scientific source
`f4d1c2f216e68a960a5a691d21cbf1f08ea4a901`, with
`qwen_vl.train.vln_runtime.load_checkpoint(epoch_directory, base_model_path)`.
These are navigation-wrapper weights. Full optimizer/RNG recovery is retained
on the cluster; the Hub export is for inference, not full training resume.
Published SHA256/size records and completion/validation evidence accompany
this model. Other repository model folders remain unchanged.
''')

allowed={'pytorch_model.bin','navigation.json','research_training.json','config.json',
         'generation_config.json','preprocessor_config.json','processor_config.json',
         'tokenizer_config.json','tokenizer.json','chat_template.jinja'}
files={f'{prefix}/epoch-1/{p.name}':p for p in export.iterdir() if p.is_file() and p.name in allowed}
assert all(f'{prefix}/epoch-1/{name}' in files for name in ('pytorch_model.bin','navigation.json','config.json','tokenizer.json','processor_config.json'))
files.update({f'{prefix}/provenance/{p.name}':p for p in provenance.iterdir() if p.is_file()})
files[f'{prefix}/README.md']=root/'README.md'
digests={name:dict(sha256=sha(p),size=p.stat().st_size) for name,p in files.items()}
write(root/'SHA256SUMS.json',digests)
files[f'{prefix}/SHA256SUMS.json']=root/'SHA256SUMS.json'
digests[f'{prefix}/SHA256SUMS.json']=dict(sha256=sha(root/'SHA256SUMS.json'),size=(root/'SHA256SUMS.json').stat().st_size)
print('HASHED',len(files),flush=True)
from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
api=HfApi();assert api.whoami()['name'].lower()=='anhdao69'
receipt=root/'PUBLISHED.json'
if receipt.exists():
    saved=json.loads(receipt.read_text());assert saved['digests']==digests
    revision=saved['revision']
else:
    info=api.model_info(repo,files_metadata=True)
    assert not any(f.rfilename.startswith(prefix+'/') for f in info.siblings),'Destination exists; inspect before overwriting'
    prior=dict(revision=info.sha,files={f.rfilename:dict(blob_id=f.blob_id,size=f.size) for f in info.siblings})
    write(root/'prior_repo_files.json',prior)
    commit=api.create_commit(repo_id=repo,parent_commit=info.sha,
        operations=[CommitOperationAdd(path_in_repo=n,path_or_fileobj=str(p)) for n,p in sorted(files.items())],
        commit_message='Publish plain M5 one-epoch decay, completed step 3852, verified serving weights')
    revision=commit.oid;write(receipt,dict(repo=repo,revision=revision,digests=digests))
    print('PUBLISHED',revision,flush=True)
info=api.model_info(repo,revision=revision,files_metadata=True)
remote={f.rfilename:f for f in info.siblings};prior=json.loads((root/'prior_repo_files.json').read_text())
attribute_change=None
for name,old in prior['files'].items():
    if name=='.gitattributes' and remote[name].blob_id!=old['blob_id']:
        before=Path(hf_hub_download(repo,name,revision=prior['revision'])).read_text()
        after=Path(hf_hub_download(repo,name,revision=revision)).read_text()
        line=f'{prefix}/epoch-1/tokenizer.json filter=lfs diff=lfs merge=lfs -text\n'
        assert after==before+line;attribute_change=line.strip()
    else:assert remote[name].blob_id==old['blob_id'] and remote[name].size==old['size'],name
verified=[]
for name,expected in sorted(digests.items()):
    f=remote[name];assert f.size==expected['size'],name
    if f.lfs:
        assert f.lfs.sha256==expected['sha256'],name;method='Hub LFS SHA256 and size'
    else:
        assert sha(hf_hub_download(repo,name,revision=revision))==expected['sha256'],name
        method='Downloaded SHA256 and size'
    verified.append(dict(path=name,**expected,method=method));print('VERIFIED',name,flush=True)
write(root/'UPLOAD_VERIFIED.json',dict(passed=True,repo=repo,revision=revision,
    folder=prefix+'/epoch-1',files=verified,prior_files_unchanged=len(prior['files'])-int(attribute_change is not None),
    automatic_metadata_change=attribute_change,verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
print('UPLOAD_VERIFIED',revision,flush=True)
