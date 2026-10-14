"""Durable worker-2-only sequence: predecessor -> qualification -> selected arms.

Runs GPU steps inside the existing allocation only after the predecessor has
completed successfully. Every launch verifies qualification and source hashes.
The login-side watcher is independent of SSH/VPN lifetime and holds no GPUs.
"""
import argparse,fcntl,hashlib,json,subprocess,time
from pathlib import Path


def source_files(source):
    source=Path(source)
    return {str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('src','scripts','configs','train') for p in sorted((source/folder).rglob('*'))
            if p.is_file() and p.suffix in ('.py','.yaml','.json','.sh')}


def selected_arms(arms):
    arms=list(arms)
    if not arms or len(set(arms))!=len(arms) or any(arm not in ('A0','A1') for arm in arms):
        raise ValueError('Select distinct A0/A1 arms')
    if arms==['A1','A0']:
        raise ValueError('A0 must precede A1 when both arms are selected')
    return arms


def validate_qualification_records(records,arms):
    arms=selected_arms(arms)
    required={'full-suite-initial','teacher-qc','distributed-pair-normalization','full-suite-final'}
    for arm in arms:
        required.update(arm+'-'+gate for gate in (
            'mixed','rxrce_longest','r2rce_longest','threshold','mixed_extreme',
            'recovery-reference','recovery-resumed','recovery-compare','export',
            'r2rce-timing','rxrce-timing'))
        if arm=='A1':required.add('A1-mixed-empty')
    names=[r.get('name') for r in records]
    if (not records or len(set(names))!=len(names) or not required.issubset(names)
            or any(r.get('returncode')!=0 for r in records)):
        raise ValueError('Missing, duplicate, or failed qualification gates: '+str(sorted(required-set(names))))


def check_qualification(source,record,arm=None):
    data=json.loads(Path(record).read_text())
    if data.get('passed') is not True or not data.get('source_files') or data['source_files']!=source_files(source):
        raise ValueError('Missing, failed, or source-mismatched A0/A1 qualification')
    arms=selected_arms(data.get('qualified_arms',[]))
    if arm is not None and arm not in arms:
        raise ValueError(f'{arm} has not passed qualification')
    validate_qualification_records(data.get('records',[]),arms)
    return data


def predecessor_status(root,steps):
    root=Path(root);supervisors=sorted(root.glob('supervisor-*'))
    if not supervisors or not (supervisors[-1]/'resources.json').exists():
        return 'waiting'
    resources=json.loads((supervisors[-1]/'resources.json').read_text())
    if resources.get('status')!='PASS':
        return 'failed'
    from qwen_vl.train.recovery import validate_complete
    final=root/'train'/f'checkpoint-{steps}'
    if not final.exists():
        partials=[p for p in (root/'train').glob('checkpoint-*') if p.name.split('-')[-1].isdigit()]
        if partials:
            latest=max(partials,key=lambda p:int(p.name.split('-')[-1]))
            partial=validate_complete(latest)
            if partial['world_size']!=4 or not 0<partial['step']<steps:
                raise ValueError('Invalid partial predecessor checkpoint')
            return 'waiting'
        raise ValueError('Successful supervisor without a recovery checkpoint')
    record=validate_complete(final)
    if record['step']!=steps or record['world_size']!=4:
        raise ValueError('Predecessor checkpoint has wrong update count/topology')
    state=json.loads((root/'train'/f'checkpoint-{steps}'/'trainer_state.json').read_text())
    if state['global_step']!=steps or state['epoch']<.999:
        raise ValueError('Predecessor did not complete the full epoch')
    return 'complete'


def step_command(allocation,source,name,command):
    source=str(Path(source).resolve())
    return ['srun','--jobid='+str(allocation),'--overlap','--ntasks=1','--gpus=4',
        '--cpus-per-task=60','--nodelist=worker-2','--job-name='+name,
        'env','G1_CODE_DIR='+source,'G1_CUDA_HOME=/mnt/data/vmo-ai-task/anhdh35/cuda-12.9.1',
        'OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','bash',source+'/train/g1_omega_env.sh',*command]


def allocation_ready(allocation):
    result=subprocess.run(['squeue','-h','-j',str(allocation),'-o','%T|%N'],capture_output=True,text=True,check=True)
    if result.stdout.strip()!='RUNNING|worker-2':
        raise RuntimeError('Required worker-2 allocation is not running; no fallback to other workers')


def allocation_idle(allocation):
    result=subprocess.run(['squeue','--steps','-h','-j',str(allocation),'-o','%i|%j'],
                          capture_output=True,text=True,check=True)
    # The interactive shell owns step0; --overlap is necessary to run inside it.
    # Every additional step must finish before another four-GPU process starts.
    shell_steps={str(allocation)+suffix for suffix in ('.0','.extern','.batch')}
    return all(row.split('|',1)[0] in shell_steps for row in result.stdout.splitlines() if row.strip())


def execute_sequence(arms,source,root,predecessor,steps,seal,run_step):
    arms=selected_arms(arms)
    if predecessor_status(predecessor,steps)!='complete':
        raise RuntimeError('Required predecessor incomplete or failed')
    if not seal.exists():
        run_step('address-qualification',['python','-m','scripts.vln.qualify_geoaddress',
                 '--root',str(seal.parent),'--arms',*arms])
    for arm in arms:check_qualification(source,seal,arm=arm)
    previous=Path(predecessor)
    for arm in arms:
        run=root/(arm.lower()+'_joint_e1')
        # Completed runs are never trained twice. Interrupted training retains
        # its output directory and restores full optimizer/RNG/sampler state.
        if predecessor_status(run,steps)!='complete':
            if predecessor_status(previous,steps)!='complete':
                raise RuntimeError('Required predecessor incomplete')
            check_qualification(source,seal,arm=arm)
            run_step(arm.lower()+'-joint-e1',['bash',str(source/'train/launch_a_omega_joint_e1.sh'),
                     arm,str(run),str(seal),str(previous)])
            if predecessor_status(run,steps)!='complete':
                raise RuntimeError(f'{arm} exited without complete epoch checkpoint')
        previous=run


def main():
    p=argparse.ArgumentParser();p.add_argument('--predecessor',required=True);p.add_argument('--steps',type=int,default=3852)
    p.add_argument('--source',required=True);p.add_argument('--qualification');p.add_argument('--check-only',action='store_true')
    p.add_argument('--arm',choices=['A0','A1'],help='Required arm for a production launch check')
    p.add_argument('--arms',nargs='+',choices=['A0','A1'],default=['A0','A1'])
    p.add_argument('--allocation');p.add_argument('--root');a=p.parse_args()
    if a.check_only:
        if not a.arm:p.error('--arm required with --check-only')
        check_qualification(a.source,a.qualification,arm=a.arm)
        if predecessor_status(a.predecessor,a.steps)!='complete':raise RuntimeError('Predecessor not successfully complete')
        print('ADDRESS_LAUNCH_PREREQUISITES_PASS');return
    if not a.allocation or not a.root:p.error('--allocation and --root required for queue')
    arms=selected_arms(a.arms)
    root=Path(a.root).resolve();root.mkdir(parents=True,exist_ok=True)
    source=Path(a.source).resolve();seal=root/'qualification'/'qualification_pass.json'
    state=root/'queue_state.json'
    def record(status,**kw):
        item=dict(status=status,updated_unix=time.time(),allocation=a.allocation,node='worker-2',source=str(source),arms=arms,**kw)
        temp=state.with_suffix('.tmp');temp.write_text(json.dumps(item,indent=2));temp.replace(state)
        print(json.dumps(item),flush=True)
    def run_step(name,command):
        allocation_ready(a.allocation)
        while not allocation_idle(a.allocation):
            record('waiting',stage='allocation-idle',next_stage=name)
            time.sleep(15)
            allocation_ready(a.allocation)
        record('running',stage=name)
        cmd=step_command(a.allocation,source,name,command)
        with (root/(name+'-'+str(time.time_ns())+'.log')).open('x') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{name} failed with exit {result.returncode}; remaining runs are blocked')
    with (root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            while True:
                allocation_ready(a.allocation)
                status=predecessor_status(a.predecessor,a.steps)
                if status=='failed':raise RuntimeError('Predecessor failed; no new runs launched')
                if status=='complete':break
                record('waiting',stage='predecessor',predecessor=str(a.predecessor))
                time.sleep(30)
            execute_sequence(arms,source,root,Path(a.predecessor),a.steps,seal,run_step)
            record('complete',stage='+'.join(arms))
        except BaseException as error:
            record('blocked',error=repr(error))
            raise

if __name__=='__main__':main()
