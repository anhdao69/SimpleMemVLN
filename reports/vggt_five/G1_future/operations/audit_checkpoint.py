import json,torch,pathlib,hashlib
r=pathlib.Path('/mnt/data/vmo-ai-task/anhdh35/g3a1-resume-worker2-20261019');cp=r/'download/checkpoint-2100';report={}
state=json.loads((cp/'trainer_state.json').read_text());report['trainer']={k:state[k] for k in ('global_step','max_steps','epoch')};assert state['global_step']==2100 and state['max_steps']==3852
scheduler=torch.load(cp/'scheduler.pt',map_location='cpu',weights_only=False);report['scheduler']={k:v for k,v in scheduler.items() if k!='lr_lambdas'};assert scheduler['last_epoch']==2100
report['optimizer']=[]
for rank in range(4):
 p=cp/f'global_step2100/bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt';s=torch.load(p,map_location='cpu',weights_only=False,mmap=True);o=s['optimizer_state_dict'];a=o['base_optimizer_state']
 assert o['zero_stage']==2 and all(n==4 for n in o['partition_count'])
 steps=[v['step'] for v in a['state'].values()];assert len(steps)==5 and set(steps)=={2100}
 groups=[{k:g[k] for k in ('lr','weight_decay','betas','eps')} for g in a['param_groups']]
 assert [g['lr'] for g in groups]==scheduler['_last_lr']
 assert all(g['betas']==(.9,.95) and g['eps']==1e-8 for g in groups)
 report['optimizer'].append(dict(rank=rank,steps=steps,groups=groups,zero_stage=o['zero_stage'],partition_count=o['partition_count'],overflow=o['overflow']))
 assert not o['overflow']
 del s,o,a
m=torch.load(cp/'global_step2100/mp_rank_00_model_states.pt',map_location='cpu',weights_only=False,mmap=True)
report['deepspeed_model_state']={k:m.get(k) for k in ('global_steps','global_samples','skipped_steps','dp_world_size','mp_world_size')};assert m['global_steps']==2100 and m['dp_world_size']==4
if m['lr_scheduler'] is not None: assert m['lr_scheduler']['last_epoch']==2100
report['scheduler_owner']='HF scheduler.pt (validated against all optimizer groups)' if m['lr_scheduler'] is None else 'DeepSpeed'
del m
weights=torch.load(cp/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
report['model']={kind:{k:list(v.shape) for k,v in weights.items() if pattern in k} for kind,pattern in [('adapters','geoaddress.'),('head','research_aux.')]}
assert len(report['model']['adapters'])==4 and len(report['model']['head'])>0
report['model']['tensor_count']=len(weights);report['model']['parameter_elements']=sum(v.numel() for v in weights.values());del weights
report['rng_files']={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in cp.glob('rng_state_*.pth')};assert len(report['rng_files'])==4
report['passed']=True;(r/'checkpoint_audit.json').write_text(json.dumps(report,indent=2)+'\n');print('CHECKPOINT_INTERNAL_AUDIT_PASS',report['trainer'],flush=True)
