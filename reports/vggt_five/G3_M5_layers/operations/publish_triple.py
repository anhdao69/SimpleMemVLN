"""CPU-only completed-triple export, immutable publication, and integrity audit."""
import os, sys, json, hashlib, shutil, datetime
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'):os.environ.pop(key,None)
base=Path('/mnt/data/vmo-ai-task/anhdh35')
campaign=base/'g3-m5-layers-worker2-20261017'
source=campaign/'source-f4d1c2f-l8_l16_l24'
run=campaign/'production/l8_l16_l24'
root=campaign/'publication-triple-epoch1-20261018'
root.mkdir(exist_ok=True)
export=root/'epoch-1'
prefix='G3_M5_L8_L16_L24'
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

from scripts.vln.queue_g3_layers import check_seal,arm_complete
from qwen_vl.train.recovery import validate_complete
seal=campaign/'qualification-f4d1c2f/qualification_pass.json'
check_seal(source,seal)
assert arm_complete(run,'l8_l16_l24')
checkpoint=run/'train/checkpoint-3852'
record=validate_complete(checkpoint)
assert record['step']==3852 and record['world_size']==4 and record['epoch_boundary']
trainer=json.loads((checkpoint/'trainer_state.json').read_text())
assert trainer['global_step']==3852 and trainer['epoch']==1 and trainer['max_steps']==3852
training=json.loads((checkpoint/'navigation.json').read_text())
assert training['config']['research']['student_layers']==[8,16,24]
supervisor=sorted(run.glob('supervisor-*'))[-1]
resources=json.loads((supervisor/'resources.json').read_text())
assert resources['status']=='PASS' and resources['new_oom_events']==resources['new_oom_kill_events']==0
print('COMPLETED_TRIPLE_CHECKPOINT_VERIFIED',flush=True)

from scripts.vln.export_vggt_g1 import export_checkpoint
from qwen_vl.research.vggt_g1 import serving_config
if not export.exists():export_checkpoint(checkpoint,export)
import torch
assert not torch.cuda.is_available()
metadata=json.loads((export/'navigation.json').read_text())
assert metadata['config']==serving_config(training['config'])
assert metadata['config']['memory']==training['config']['memory']
assert 'research' not in metadata['config']
before=torch.load(checkpoint/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
after=torch.load(export/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
keys={k for k in before if not k.startswith('research_aux.')}
assert set(after)==keys and all(torch.equal(after[k],before[k]) for k in keys)
removed=sorted(set(before)-set(after))
assert removed and all(any(k.startswith(f'research_aux.state_heads.{i}.') for i in (8,16,24)) for k in removed)
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
           qualification_gpu_export='qualification-f4d1c2f/l8_l16_l24/serving-export/g3_m5_export_check.json')
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
    (campaign/'qualification-f4d1c2f/l8_l16_l24/serving-export/g3_m5_export_check.json','qualification_serving_replay.json')]:
    shutil.copy2(src,provenance/name)
if not (provenance/'source_and_completion.json').exists():
    write(provenance/'source_and_completion.json',dict(
        source_commit='f4d1c2f216e68a960a5a691d21cbf1f08ea4a901',
        source_directory=str(source),checkpoint=str(checkpoint),completed_epoch=1,
        optimizer_updates=3852,global_batch=8,supervised_layers=[8,16,24],
        recovery_inventory=record,export_source_weights_sha256=sha(checkpoint/'pytorch_model.bin'),
        export_weights_sha256=sha(export/'pytorch_model.bin'),
        verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
(root/'README.md').write_text('''# G3 + M5, L8+L16+L24 supervision — completed joint epoch 1

`epoch-1/` is the serving export of the completed zero-based decoder-layer-[8,16,24]
ablation: 3,852 optimizer updates, epoch 1.0, joint R2R + English RxR15deg
(30,815 episodes). Initialized independently from original Qwen3.5-4B,
revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

The L24 baseline remains in `G3_M5/`. This triple-layer model preserves M5 episodic
memory (four recent groups, 1,200 selected older visual tokens, committed-mean-L2
scoring), canonical text actions with action history, and full-episode BPTT.
Training used online frozen VGGT-Omega, the original G3 targets/sampling and
independent native-state query/readout MLP for each supervised layer, with no PCA or preextracted features.

The epoch completed at update 3852 with action-weighted training loss
0.12598709897034086 and no GPU/host OOM events. A user-authorized pause at
update 2227 qualified the next plain-M5 run, then restored the complete
optimizer/scheduler/RNG checkpoint. The run_timing.json measures the final
resumed invocation; epoch-1.json includes the whole epoch.

Four H100s, global batch 8 (one episode/rank, accumulation 2), seed 429.
One-epoch cosine schedule: 3,852 updates, 116 warmup, minimum LR ratio 0.1.
Backbone LR 5e-6, auxiliary LR 1e-4, maximum auxiliary weight 0.1, ramp 771.
The three layer losses are averaged: (L8 + L16 + L24) / 3, preserving the
total auxiliary weight of 0.1.

Only training-only `research_aux.*` tensors and research configuration were
removed. Every serving tensor is bitwise identical to the completed checkpoint.
The native navigation architecture strictly reloaded on CPU, and serializer
and M5 memory metadata were verified. The earlier qualification checkpoint
passed both-dataset GPU streaming/offline/export tests; those are explicitly
qualification evidence, not a new GPU replay of the completed model. Plain M5 with the matched one-epoch schedule
continued training while this CPU-only export/upload ran. No navigation
success-rate results are claimed; training loss is not evaluation accuracy.

Use SimpleMemVLN branch `c2_vggt_omega_g3_m5_layers`, scientific source
`f4d1c2f216e68a960a5a691d21cbf1f08ea4a901`, and
`qwen_vl.train.vln_runtime.load_checkpoint(epoch_directory, base_model_path)`.
These are navigation-wrapper weights, not a plain AutoModel state dictionary.
No teacher or auxiliary head is required for inference. Full optimizer/RNG
recovery remains on the cluster; this export is for serving, not training resume.
`provenance/` contains completion and validation evidence; `SHA256SUMS.json`
records published hashes and sizes. Existing repository models are preserved.
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
        commit_message='Publish completed G3 M5 L8+L16+L24 epoch 1, step 3852, verified serving weights')
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
