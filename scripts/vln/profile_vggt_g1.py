"""Owned four-H100 G1 experiments, complete episodes and real cached Omega targets."""
import argparse,json,os,statistics,subprocess,time
from pathlib import Path
import yaml
from scripts.vln.profile_episodic import verify, _rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--name',required=True)
    p.add_argument('--fixture',required=True);p.add_argument('--updates',type=int,default=3)
    p.add_argument('--threads',type=int,default=8);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--attention-batch',type=int,default=8);p.add_argument('--vision-batch',type=int,default=4)
    p.add_argument('--teacher-batch',type=int,default=8);p.add_argument('--teacher-residency',choices=['resident','cpu_between'],default='resident')
    p.add_argument('--cpu-optimizer',action='store_true')
    p.add_argument('--save',action='store_true');p.add_argument('--resume');p.add_argument('--first-update',type=int,default=1)
    p.add_argument('--baseline',action='store_true',help='Disposable native-off timing diagnostic only')
    a=p.parse_args();root=Path(a.root);out=root/'experiments'/a.name
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    config=yaml.safe_load(Path('configs/episodic/profile_h4.yaml').read_text())
    config['training'].update(dataloader_num_workers=a.workers,dataloader_persistent_workers=a.workers>0,
        dataloader_prefetch_factor=2 if a.workers else None)
    config['runtime'].update(window_attention_batch_steps=a.attention_batch,vision_microbatch_images=a.vision_batch,
                            episodic_diagnostics=False,malloc_trim_each_update=False)
    if not a.baseline:
        from qwen_vl.research.vggt_omega import teacher_identity
        source='/mnt/data/vmo-ai-task/anhdh35/third_party/vggt-omega'
        weights='/mnt/data/vmo-ai-task/anhdh35/models/vggt-omega/vggt_omega_1b_512.pt'
        config['research']=dict(run_id='G1',teacher='VGGT-Omega-1B-512',student_layer=23,
            teacher_source=source,teacher_weights=weights,teacher_identity=teacher_identity(source,weights),
            teacher_batch_size=a.teacher_batch,teacher_residency=a.teacher_residency,sample_seed=429,
            head_lr=1e-4,lambda_max=.2,ramp_updates=771)
    config_path=out/'runtime.yaml';config_path.write_text(yaml.safe_dump(config))
    model='/mnt/data/vmo-ai-task/anhdh35/cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
    command=['torchrun','--standalone','--nproc_per_node=4','-m','qwen_vl.train.train_qwen',
        '--vln_config','configs/episodic/science_common.yaml','--output_config','configs/vln_empty.yaml',
        '--memory_config','configs/vln_memory_window8.yaml','--runtime_profile',str(config_path),
        '--deepspeed',('configs/deepspeed_zero2_cpu_optimizer.json' if a.cpu_optimizer else 'deepspeed.json'),'--manifest',a.fixture,
        '--model_name_or_path',model,'--max-optimizer-updates',str(a.updates),'--debug-repeat-episodes',
        '--output_dir',str(out/'train'),'--run-name','g1-diagnostic-'+a.name]
    if a.save: command+=['--recovery-save-steps','1']
    else: command+=['--profile-only']
    if a.resume: command+=['--resume_from_checkpoint',a.resume]
    os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=str(a.threads)
    start=time.time()
    status=subprocess.run(['python','-m','scripts.vln.run_episodic_gate','--out',str(out/'supervisor'),
        '--host-guard-gib','691.2','--disk-path',str(root),'--min-free-disk-gib','220','--',*command]).returncode
    resources=json.loads((out/'supervisor/resources.json').read_text())
    check=verify(out/'train',4,a.updates,8,78,first_update=a.first_update)
    ranks=[_rows(out/'train'/f'profile_rank{r}.jsonl') for r in range(4)]
    ends=[max(rows[i]['unix_end'] for rows in ranks) for i in range(min(map(len,ranks)))]
    intervals=[b-a for a,b in zip(ends,ends[1:])]
    exposures=[_rows(out/'train'/f'exposures_rank{r}.jsonl') for r in range(4)]
    if not a.baseline:
        import math
        if not all(math.isfinite(e.get('aux_loss_sum',float('nan'))) and e.get('aux_frame_count',0)>0
                   for rank in exposures for e in rank):
            check['passed']=False;check['problems'].append('Missing/nonfinite auxiliary evidence')
    result=dict(name=a.name,passed=status==0 and check['passed'] and resources['status']=='PASS',
        check=check,resources=resources,steady_seconds=statistics.mean(intervals) if intervals else None,
        intervals=intervals,wall_seconds=time.time()-start,threads=a.threads,workers=a.workers,
        attention_batch=a.attention_batch,vision_batch=a.vision_batch,fixture=a.fixture,
        complete_episode_exposures=sum(map(len,exposures)),aux_frames=sum(e.get('aux_frame_count',0) for rank in exposures for e in rank),
        teacher_batch=a.teacher_batch,teacher_residency=a.teacher_residency,cpu_optimizer=a.cpu_optimizer,
        source_note='diagnostic mutable development worktree; final source commit recorded separately')
    (out/'result.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps(result),flush=True)
    if not result['passed']: raise SystemExit(1)

if __name__=='__main__': main()
