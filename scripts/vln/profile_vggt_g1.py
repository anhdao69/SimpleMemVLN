"""Owned four-H100 G1 experiments, complete episodes and online Omega targets."""
import argparse,json,os,statistics,subprocess,time
from pathlib import Path
import yaml
from scripts.vln.profile_episodic import verify, _rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['G1','G3','G3_M5'],default='G1');p.add_argument('--root',required=True);p.add_argument('--name',required=True)
    p.add_argument('--fixture',required=True);p.add_argument('--updates',type=int,default=3)
    p.add_argument('--threads',type=int,default=8);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--attention-batch',type=int,default=8);p.add_argument('--vision-batch',type=int,default=4)
    p.add_argument('--teacher-batch',type=int,default=8);p.add_argument('--teacher-residency',choices=['resident','cpu_between'],default='resident')
    p.add_argument('--cpu-optimizer',action='store_true')
    p.add_argument('--cpu-adam',action='store_true',help='Vectorized DeepSpeed CPU AdamW; implies CPU optimizer offload')
    p.add_argument('--save',action='store_true');p.add_argument('--resume');p.add_argument('--first-update',type=int,default=1)
    p.add_argument('--baseline',action='store_true',help='Disposable native-off timing diagnostic only')
    a=p.parse_args();a.cpu_optimizer=a.cpu_optimizer or a.cpu_adam
    if not Path(a.fixture).is_file():raise FileNotFoundError(a.fixture)
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
        if a.arm in ('G3','G3_M5'):
            from qwen_vl.research.vggt_g3_teacher import g3_teacher_identity as teacher_identity
        source='/mnt/data/vmo-ai-task/anhdh35/third_party/vggt-omega'
        weights='/mnt/data/vmo-ai-task/anhdh35/models/vggt-omega/vggt_omega_1b_512.pt'
        config['research']=dict(run_id=a.arm,teacher='VGGT-Omega-1B-512',student_layer=23 if a.arm=='G1' else 24,
            teacher_source=source,teacher_weights=weights,teacher_identity=teacher_identity(source,weights),
            teacher_batch_size=a.teacher_batch,teacher_residency=a.teacher_residency,sample_seed=429,
            sampling_identity='collected-rgb-paths-v1',
            head_lr=1e-4,lambda_max=.2 if a.arm=='G1' else .1,ramp_updates=771)
    config_path=out/'runtime.yaml';config_path.write_text(yaml.safe_dump(config))
    model='/mnt/data/vmo-ai-task/anhdh35/cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
    command=['torchrun','--standalone','--nproc_per_node=4','-m','qwen_vl.train.train_qwen',
        '--vln_config','configs/episodic/science_common.yaml','--output_config','configs/vln_empty.yaml',
        '--memory_config',('configs/episodic/arm_m5.yaml' if a.arm=='G3_M5' else 'configs/vln_memory_window8.yaml'),'--runtime_profile',str(config_path),
        '--deepspeed',('configs/deepspeed_zero2_cpu_optimizer.json' if a.cpu_optimizer else 'deepspeed.json'),'--manifest',a.fixture,
        '--model_name_or_path',model,'--max-optimizer-updates',str(a.updates),'--debug-repeat-episodes',
        '--output_dir',str(out/'train'),'--run-name',a.arm.lower()+'-diagnostic-'+a.name]
    if a.arm=='G3_M5':command+=['--policy_config','configs/vggt/g3_joint_one_epoch.yaml']
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
        if not all(math.isfinite(e.get('aux_loss_sum',float('nan'))) and e.get('aux_pair_count',e.get('aux_frame_count',-1))>=0
                   for rank in exposures for e in rank):
            check['passed']=False;check['problems'].append('Missing/nonfinite auxiliary evidence')
    result=dict(arm=a.arm,name=a.name,passed=status==0 and check['passed'] and resources['status']=='PASS',
        check=check,resources=resources,steady_seconds=statistics.mean(intervals) if intervals else None,
        intervals=intervals,wall_seconds=time.time()-start,threads=a.threads,workers=a.workers,
        attention_batch=a.attention_batch,vision_batch=a.vision_batch,fixture=a.fixture,
        complete_episode_exposures=sum(map(len,exposures)),aux_frames=sum(e.get('aux_frame_count',0) for rank in exposures for e in rank),
        aux_pairs=sum(e.get('aux_pair_count',0) for rank in exposures for e in rank),
        teacher_batch=a.teacher_batch,teacher_residency=a.teacher_residency,cpu_optimizer=a.cpu_optimizer,
        cpu_adam=a.cpu_adam,
        cuda_toolkit=os.environ.get('CUDA_HOME'),
        source_commit=(Path('SOURCE_COMMIT').read_text().strip() if Path('SOURCE_COMMIT').is_file() else subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),
        source_status=('immutable archive' if Path('SOURCE_COMMIT').is_file() else subprocess.check_output(['git','status','--porcelain'],text=True)),
        source_tree_sha256=source_digest,
        teacher_identity=config.get('research',{}).get('teacher_identity'),
        fixture_sha256=__import__('hashlib').sha256(Path(a.fixture).read_bytes()).hexdigest())
    (out/'result.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps(result),flush=True)
    if not result['passed']: raise SystemExit(1)

if __name__=='__main__': main()
