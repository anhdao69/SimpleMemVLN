"""Worker-resident plain-M5 qualification -> resumed triple -> one-epoch M5."""
import fcntl,hashlib,json,os,subprocess,sys,threading,time
from pathlib import Path

BASE=Path('/mnt/data/vmo-ai-task/anhdh35')
ROOT=BASE/'m5-one-epoch-worker2-20261018'
CODE=ROOT/'source-f4d1c2f'
OPS=ROOT/'operations'
LAYERS=BASE/'g3-m5-layers-worker2-20261017'
TRIPLE=LAYERS/'production/l8_l16_l24'
SEAL=LAYERS/'qualification-f4d1c2f/qualification_pass.json'
MANIFEST=BASE/'SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl'
MANIFEST_SHA='5deb425d2594ce96931dd6ce12bd6084b066cd44f04808c1dd3ee2e4b3672933'

def execute_sequence(qualify,resume_triple,triple_complete,launch_m5):
    error=None
    try:qualify()
    except Exception as exc:error=exc
    # A failed diagnostic must not leave the user's current training stopped.
    resume_triple()
    if not triple_complete():raise RuntimeError('Triple exited without full epoch')
    if error is not None:raise RuntimeError('M5 qualification failed; triple resumed; M5 blocked') from error
    launch_m5()

def ready_to_qualify(active,own,allocation):
    return not (set(active)-{own,*[allocation+s for s in ('.0','.extern','.batch')]})

def write(path,data):
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def contract():
    from scripts.vln.queue_g3_layers import check_seal
    from qwen_vl.train.vln_runtime import resolve_config
    check_seal(CODE,SEAL)
    cfg=resolve_config(CODE/'configs/episodic/science_common.yaml',CODE/'configs/vln_empty.yaml',
        CODE/'configs/episodic/arm_m5.yaml',CODE/'configs/vggt/g3_joint_one_epoch.yaml',OPS/'runtime.yaml')
    ref=resolve_config(CODE/'configs/episodic/science_common.yaml',CODE/'configs/vln_empty.yaml',
        CODE/'configs/episodic/arm_m5.yaml',CODE/'configs/vggt/g3_joint_one_epoch.yaml',CODE/'configs/vggt/g3_m5_omega_h4.yaml')
    ref.pop('research');assert cfg==ref and 'research' not in cfg
    assert (cfg['training']['epochs'],cfg['training']['expected_total_steps'],cfg['training']['warmup_steps'])==(1,3852,116)
    assert cfg['training']['gradient_accumulation_steps']==2 and cfg['training']['seed']==429
    assert sha(MANIFEST)==MANIFEST_SHA
    # Verify the actual scheduler API, including warmup and final LR floor.
    import torch
    from transformers import get_scheduler
    param=torch.nn.Parameter(torch.zeros(1));opt=torch.optim.AdamW([param],lr=5e-6)
    schedule=get_scheduler('cosine_with_min_lr',opt,num_warmup_steps=116,num_training_steps=3852,scheduler_specific_kwargs={'min_lr_rate':.1})
    factor=schedule.lr_lambdas[0]
    assert factor(0)==0 and factor(116)==1 and abs(factor(3852)-.1)<1e-12
    return dict(config=cfg,manifest_sha256=MANIFEST_SHA,scientific_source='f4d1c2f216e68a960a5a691d21cbf1f08ea4a901',
        operation_files={p.name:sha(p) for p in sorted(OPS.iterdir()) if p.suffix in ('.py','.sh','.yaml')})

def verify_qualification():
    data=json.loads((ROOT/'qualification/QUALIFIED.json').read_text())
    assert data['passed'] and data['contract']==contract()
    assert set(data['gates'])=={'mixed-reference','mixed-resumed','longest-rxr','recovery-compare'}
    return data

def qualify(record):
    q=ROOT/'qualification';q.mkdir(exist_ok=True)
    if (q/'QUALIFIED.json').exists():verify_qualification();return
    identity=contract();gates={}
    def profile(name,fixture,updates=3,extra=()):
        record('qualifying',stage=name)
        out=q/'experiments'/name
        if (out/'result.json').exists():
            result=json.loads((out/'result.json').read_text());assert result['passed']
        else:
            if out.exists():raise RuntimeError(f'Preserve/rename interrupted diagnostic before retry: {out}')
            args=[sys.executable,'-m','scripts.vln.profile_vggt_g1','--baseline','--arm','G3_M5',
                '--root',str(q),'--name',name,'--fixture',str(BASE/'g1-omega-20261013/fixtures'/fixture),
                '--updates',str(updates),'--cpu-adam','--threads','4','--workers','8',
                '--attention-batch','32','--vision-batch','8',*extra]
            with (q/(name+'.log')).open('w') as log:
                subprocess.run(args,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
            result=json.loads((out/'result.json').read_text());assert result['passed']
        assert json.loads((out/'train/resolved_config.json').read_text())==identity['config']
        assert result['teacher_identity'] is None and result['aux_pairs']==0
        gates[name]=dict(passed=True,steady_seconds=result['steady_seconds'],result=str(out/'result.json'))
    profile('mixed-reference','mixed_tail.jsonl',extra=['--save'])
    ref=q/'experiments/mixed-reference/train';res=q/'experiments/mixed-resumed/train'
    profile('mixed-resumed','mixed_tail.jsonl',extra=['--save','--resume',str(ref/'checkpoint-2'),'--first-update','3'])
    record('qualifying',stage='recovery-compare')
    import torch
    from scripts.vln.check_vggt_recovery import compare
    from qwen_vl.train.recovery import validate_complete
    for path in (ref/'checkpoint-2',ref/'checkpoint-3',res/'checkpoint-3'):
        assert validate_complete(path)['world_size']==4
    states=[torch.load(p/'final/pytorch_model.bin',weights_only=True,map_location='cpu',mmap=True) for p in (ref,res)]
    assert not any(k.startswith('research_aux.') for k in states[0])
    comparison=compare(states);assert comparison['relative_l2']<=1e-5
    losses=[next(x['loss'] for x in reversed(json.loads((p/'trainer_state.json').read_text())['log_history']) if x.get('step')==3 and 'loss' in x) for p in (ref,res)]
    assert abs(losses[0]-losses[1])<=1e-4
    gates['recovery-compare']=dict(passed=True,weights=comparison,losses=losses)
    del states
    profile('longest-rxr','rxrce_longest.jsonl',updates=2)
    assert identity==contract()
    write(q/'QUALIFIED.json',dict(passed=True,contract=identity,gates=gates))

def m5_complete(run):
    from scripts.vln.queue_g3_m5 import predecessor_status
    if predecessor_status(run)!='complete':return False
    cfg=json.loads((run/'train/checkpoint-3852/navigation.json').read_text())['config']
    assert cfg==contract()['config'] and 'research' not in cfg
    return True

def main():
    assert os.environ.get('SLURMD_NODENAME')=='worker-2' and os.environ.get('SLURM_JOB_ID')=='4925'
    allocation=os.environ['SLURM_JOB_ID'];own=allocation+'.'+os.environ['SLURM_STEP_ID']
    ROOT.mkdir(exist_ok=True)
    def record(status,**fields):
        write(ROOT/'queue_state.json',dict(status=status,updated_unix=time.time(),pid=os.getpid(),node='worker-2',
            step=own,predecessor=str(TRIPLE),next_run='plain_M5_one_epoch',**fields))
        print(status,json.dumps(fields),flush=True)
    def active():return subprocess.check_output(['squeue','--steps','-h','-j',allocation,'-o','%i'],text=True).split()
    def heartbeat(stop):
        while not stop.wait(15):write(ROOT/'heartbeat.json',dict(unix=time.time(),pid=os.getpid(),step=own))
    from scripts.vln.queue_g3_layers import arm_complete
    def completed():return arm_complete(TRIPLE,'l8_l16_l24')
    def resume():
        if completed():record('predecessor-complete');return
        record('running-predecessor',stage='triple-resume',m5_qualified=(ROOT/'qualification/QUALIFIED.json').exists())
        command=['bash',str(LAYERS/'source-f4d1c2f-l8_l16_l24/train/g1_omega_env.sh'),sys.executable,str(LAYERS/'run_selected_arm.py')]
        env=dict(os.environ,G1_CODE_DIR=str(LAYERS/'source-f4d1c2f-l8_l16_l24'))
        with (ROOT/f'triple-resume-{time.time_ns()}.log').open('x') as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    def launch():
        verify_qualification();run=ROOT/'production/m5_joint_e1'
        identity=contract();run.parent.mkdir(exist_ok=True)
        path=run/'run_identity.json'
        if path.exists():assert json.loads(path.read_text())==identity
        else:
            assert not run.exists() or not any(run.iterdir())
            run.mkdir(exist_ok=True);write(path,identity)
        if not m5_complete(run):
            record('running-m5',run=str(run))
            with (ROOT/f'm5-launch-{time.time_ns()}.log').open('x') as log:
                subprocess.run(['bash',str(OPS/'launch_m5.sh'),str(run)],cwd=CODE,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert m5_complete(run)
        record('complete',run=str(run))
    with (ROOT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        contract() # validate before any pause request
        stop=threading.Event();thread=threading.Thread(target=heartbeat,args=(stop,),daemon=True);thread.start()
        flag=TRIPLE/'train/REQUEST_CHECKPOINT_AND_STOP'
        try:
            if not (ROOT/'qualification/QUALIFIED.json').exists() and not completed():
                q=json.loads((LAYERS/'production/queue_state.json').read_text())
                if any(s not in {own,allocation+'.0',allocation+'.extern',allocation+'.batch'} for s in active()):
                    assert q['status']=='running' and q['arm']=='l8_l16_l24'
                    allowed={own,allocation+'.'+q['step'],allocation+'.0',allocation+'.extern',allocation+'.batch'}
                    assert set(active())<=allowed,'Unexpected compute step; refusing pause'
                    flag.write_text(allocation+'\n');record('waiting-for-checkpoint',previous_step=allocation+'.'+q['step'])
            while not ready_to_qualify(active(),own,allocation):time.sleep(10)
            from qwen_vl.train.recovery import latest_checkpoint,validate_complete
            cp=latest_checkpoint(TRIPLE/'train');assert cp is not None
            recovery=validate_complete(cp)
            write(ROOT/'predecessor_pause.json',dict(checkpoint=str(cp),step=recovery['step'],world_size=recovery['world_size']))
            if flag.exists():flag.rename(ROOT/f'checkpoint-request-consumed-{time.time_ns()}')
            execute_sequence(lambda:qualify(record),resume,completed,launch)
        except BaseException as error:
            record('blocked',error=repr(error));raise
        finally:stop.set();thread.join()

if __name__=='__main__':main()
