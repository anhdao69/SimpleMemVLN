from pathlib import Path
import json,os,time,subprocess,statistics,yaml
from scripts.vln.profile_episodic import verify
R=Path('/mnt/data/vmo-ai-task/anhdh35');E=R/'c2-speed-20261011';E.mkdir(exist_ok=True)
fixture=R/'c2-audit-evidence-20261011/fixtures/medium_joint.jsonl'
base=yaml.safe_load(Path('configs/episodic/profile_h2.yaml').read_text())
cases=[('baseline',2,2,True,8,4),('threads4',4,2,True,8,4),('threads8',8,2,True,8,4),('no_trim',4,2,False,8,4),('workers4',4,4,False,8,4),('batch16_vision8',4,4,False,16,8),('batch32_vision16',4,4,False,32,16)]
results=[]
for name,threads,workers,trim,batch,vision in cases:
 cfg={k:dict(v) for k,v in base.items()};cfg['training'].update(dataloader_num_workers=workers,dataloader_prefetch_factor=2)
 cfg['runtime'].update(malloc_trim_each_update=trim,window_attention_batch_steps=batch,vision_microbatch_images=vision)
 config=E/f'{name}.yaml';config.write_text(yaml.safe_dump(cfg))
 out=E/'screen'/name;out.mkdir(parents=True,exist_ok=False)
 command=['python','-m','scripts.vln.run_episodic_gate','--out',str(out/'supervisor'),'--host-guard-gib','680','--disk-path',str(E),'--','torchrun','--standalone','--nproc_per_node=2','-m','qwen_vl.train.train_qwen','--vln_config','configs/episodic/science_common.yaml','--output_config','configs/episodic/arm_v0.yaml','--runtime_profile',str(config),'--deepspeed','configs/deepspeed_zero2_cpu_optimizer.json','--manifest',str(fixture),'--model_name_or_path',os.environ['VLN_MODEL_PATH'],'--max-optimizer-updates','4','--debug-repeat-episodes','--output_dir',str(out/'train'),'--run-name',f'speed-{name}','--profile-only']
 env=dict(os.environ,OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads))
 start=time.time();rc=subprocess.run(command,env=env).returncode
 checks=verify(out/'train',2,4,8,78)
 profiles=[[json.loads(x) for x in (out/'train'/f'profile_rank{rank}.jsonl').read_text().splitlines()] for rank in range(2)] if all((out/'train'/f'profile_rank{r}.jsonl').exists() for r in range(2)) else []
 ends=[max(p[u]['unix_end'] for p in profiles) for u in range(min(map(len,profiles)))] if profiles else []
 intervals=[b-a for a,b in zip(ends,ends[1:])]
 resource=json.loads((out/'supervisor/resources.json').read_text())
 result=dict(name=name,threads=threads,config=str(config),passed=rc==0 and checks['passed'],training=checks,resources=resource,wall_seconds=time.time()-start,steady_intervals=intervals,steady_seconds=statistics.mean(intervals) if intervals else None)
 results.append(result);(E/'screen-results.json').write_text(json.dumps(results,indent=2)+'\n');print('SCREEN',json.dumps(result),flush=True)
print('SCREEN_DONE',flush=True)
