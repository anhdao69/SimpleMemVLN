"""Owned four-H100 G1 experiments, complete episodes and online Omega targets."""
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
    p.add_argument('--future-offset',type=int,help='Opt-in joint current/future G1 targets; positive trajectory offset')
    p.add_argument('--policy-config',help='Optional training schedule overlay')
    p.add_argument('--cpu-optimizer',action='store_true')
    p.add_argument('--cpu-adam',action='store_true',help='Vectorized DeepSpeed CPU AdamW; implies CPU optimizer offload')
    p.add_argument('--save',action='store_true');p.add_argument('--resume');p.add_argument('--first-update',type=int,default=1)
    p.add_argument('--baseline',action='store_true',help='Disposable native-off timing diagnostic only')
    a=p.parse_args();a.cpu_optimizer=a.cpu_optimizer or a.cpu_adam
    if not Path(a.fixture).is_file(): raise FileNotFoundError(a.fixture)
    if a.future_offset is not None:
        from qwen_vl.research.vggt_future import future_frame_pairs
        future_frame_pairs((),0,a.future_offset)
        if a.baseline: raise ValueError('Future teacher requires G1 auxiliary supervision')
        if a.policy_config is None: a.policy_config='configs/vggt/g1_joint_one_epoch.yaml'
    root=Path(a.root);out=root/'experiments'/a.name
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    import hashlib
    source_files={str(f):hashlib.sha256(f.read_bytes()).hexdigest()
        for directory in ('src','scripts','configs','train') for f in sorted(Path(directory).rglob('*'))
        if f.is_file() and f.suffix in ('.py','.yaml','.json','.sh')}
    source_digest=hashlib.sha256(json.dumps(source_files,sort_keys=True).encode()).hexdigest()
    (out/'source_manifest.json').write_text(json.dumps(source_files,indent=2))
    config=yaml.safe_load(Path('configs/episodic/profile_h4.yaml').read_text())
    config['training'].update(dataloader_num_workers=a.workers,dataloader_persistent_workers=a.workers>0,
        dataloader_prefetch_factor=2 if a.workers else None)
    if a.cpu_adam:config['training']['cpu_optimizer_impl']='deepspeed_cpu_adam'
    config['runtime'].update(window_attention_batch_steps=a.attention_batch,vision_microbatch_images=a.vision_batch,
                            episodic_diagnostics=False,malloc_trim_each_update=False)
    if not a.baseline:
        from qwen_vl.research.vggt_omega import teacher_identity
        source='/mnt/data/vmo-ai-task/anhdh35/third_party/vggt-omega'
        weights='/mnt/data/vmo-ai-task/anhdh35/models/vggt-omega/vggt_omega_1b_512.pt'
        config['research']=dict(run_id='G1',teacher='VGGT-Omega-1B-512',student_layer=23,
            teacher_source=source,teacher_weights=weights,teacher_identity=teacher_identity(source,weights),
            teacher_batch_size=a.teacher_batch,teacher_residency=a.teacher_residency,sample_seed=429,
            sampling_identity='collected-rgb-paths-v1',
            head_lr=1e-4,lambda_max=.2,ramp_updates=771)
    if a.future_offset is not None:
        from qwen_vl.research.vggt_future import future_teacher_identity
        config['research'].update(teacher_context='current_future_v1',teacher_future_offset=a.future_offset,
            teacher_identity=future_teacher_identity(source,weights,a.future_offset))
    config_path=out/'runtime.yaml';config_path.write_text(yaml.safe_dump(config))
    model='/mnt/data/vmo-ai-task/anhdh35/cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
    command=['torchrun','--standalone','--nproc_per_node=4','-m','qwen_vl.train.train_qwen',
        '--vln_config','configs/episodic/science_common.yaml','--output_config','configs/vln_empty.yaml',
        '--memory_config','configs/vln_memory_window8.yaml','--runtime_profile',str(config_path),
        '--deepspeed',('configs/deepspeed_zero2_cpu_optimizer.json' if a.cpu_optimizer else 'deepspeed.json'),'--manifest',a.fixture,
        '--model_name_or_path',model,'--max-optimizer-updates',str(a.updates),'--debug-repeat-episodes',
        '--output_dir',str(out/'train'),'--run-name','g1-diagnostic-'+a.name]
    if a.policy_config: command+=['--policy_config',a.policy_config]
    if a.save: command+=['--recovery-save-steps','1']
    else: command+=['--profile-only']
    if a.resume: command+=['--resume_from_checkpoint',a.resume]
    if a.cpu_adam:
        # Separate matching toolkit for this optional extension; no driver or
        # package changes to the environment used by existing training jobs.
        toolkit=Path('/mnt/data/vmo-ai-task/anhdh35/cuda-12.9.1')
        if not (toolkit/'bin/nvcc').is_file():raise FileNotFoundError(toolkit/'bin/nvcc')
        os.environ['CUDA_HOME']=str(toolkit)
        os.environ['PATH']=str(toolkit/'bin')+os.pathsep+os.environ['PATH']
    os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=str(a.threads)
    start=time.time()
    status=subprocess.run(['python','-m','scripts.vln.run_episodic_gate','--out',str(out/'supervisor'),
        '--host-guard-gib','691.2','--disk-path',str(root),'--min-free-disk-gib','220','--',*command]).returncode
    resources=json.loads((out/'supervisor/resources.json').read_text())
    check=verify(out/'train',4,a.updates,8,78,first_update=a.first_update)
    ranks=[_rows(out/'train'/f'profile_rank{r}.jsonl') for r in range(4)]
    check['peak_allocated_gib']=max((row['peak_allocated_gib'] for rows in ranks for row in rows),default=None)
    ends=[max(rows[i]['unix_end'] for rows in ranks) for i in range(min(map(len,ranks)))]
    intervals=[b-a for a,b in zip(ends,ends[1:])]
    exposures=[_rows(out/'train'/f'exposures_rank{r}.jsonl') for r in range(4)]
    if not a.baseline:
        import math
        if not all(math.isfinite(e.get('aux_loss_sum',float('nan'))) and e.get('aux_frame_count',-1)>=(0 if a.future_offset is not None else 1)
                   for rank in exposures for e in rank):
            check['passed']=False;check['problems'].append('Missing/nonfinite auxiliary evidence')
        if a.future_offset is not None:
            from qwen_vl.research.vggt_sampling import sample_frames,collection_trajectory_key
            expected={}
            for line in Path(a.fixture).read_text().splitlines():
                ep=json.loads(line);n=len(ep['steps'])
                expected[ep['episode_uid']]=len(future_frame_pairs(
                    sample_frames(collection_trajectory_key(ep),n,429),n,a.future_offset))
            totals={}
            for rank in exposures:
                for e in rank:
                    if e['aux_frame_count']!=expected[e['episode_uid']]:
                        check['problems'].append('Future frame support mismatch')
                    totals[e['update']]=totals.get(e['update'],0)+e['aux_frame_count']
            if any(e['update_aux_count']!=totals[e['update']] for rank in exposures for e in rank):
                check['problems'].append('Global auxiliary denominator mismatch')
            if check['problems']:check['passed']=False
    result=dict(name=a.name,passed=status==0 and check['passed'] and resources['status']=='PASS',
        check=check,resources=resources,steady_seconds=statistics.mean(intervals) if intervals else None,
        intervals=intervals,wall_seconds=time.time()-start,threads=a.threads,workers=a.workers,
        attention_batch=a.attention_batch,vision_batch=a.vision_batch,fixture=a.fixture,
        complete_episode_exposures=sum(map(len,exposures)),aux_frames=sum(e.get('aux_frame_count',0) for rank in exposures for e in rank),
        teacher_batch=a.teacher_batch,teacher_residency=a.teacher_residency,cpu_optimizer=a.cpu_optimizer,
        cpu_adam=a.cpu_adam,future_offset=a.future_offset,policy_config=a.policy_config,
        cuda_toolkit=os.environ.get('CUDA_HOME'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_status=subprocess.check_output(['git','status','--porcelain'],text=True),
        source_tree_sha256=source_digest,
        teacher_identity=config.get('research',{}).get('teacher_identity'),
        fixture_sha256=__import__('hashlib').sha256(Path(a.fixture).read_bytes()).hexdigest())
    (out/'result.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps(result),flush=True)
    if not result['passed']: raise SystemExit(1)

if __name__=='__main__': main()
