"""Source-bound G3_M5 qualification and worker-2 queue after successful R1."""
import argparse,fcntl,hashlib,json,subprocess,time
from pathlib import Path

REQUIRED_GATES=('cpu-and-cuda-suite','mixed-empty','mixed-empty-reversed','mixed',
    'rxrce-longest','r2rce-longest','recovery-reference','recovery-resumed',
    'recovery-compare','export','distributed-pairs','r2rce-timing','rxrce-timing',
    'attention16','attention64')


def write_seal(source,path,records,expected):
    names=[r.get('name') for r in records]
    if (len(names)!=len(set(names)) or set(names)!=set(REQUIRED_GATES)
            or any(r.get('returncode')!=0 for r in records)):
        raise ValueError('Incomplete or failed G3_M5 qualification')
    if not expected or source_files(source)!=expected:raise ValueError('Qualification source changed')
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(passed=True,qualified_arm='G3_M5',records=records,source_files=expected),indent=2))
    temp.replace(path)


def check_qualification(source,path):
    data=json.loads(Path(path).read_text());names=[r.get('name') for r in data.get('records',[])]
    if (data.get('passed') is not True or data.get('qualified_arm')!='G3_M5'
            or not data.get('source_files') or data['source_files']!=source_files(source)
            or len(names)!=len(set(names)) or set(names)!=set(REQUIRED_GATES)
            or any(r.get('returncode')!=0 for r in data['records'])):
        raise ValueError('Missing, failed, incomplete or source-mismatched G3_M5 qualification')
    return data

def source_files(source):
    source=Path(source)
    return {str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('src','scripts','configs','train','tests') for p in sorted((source/folder).rglob('*'))
            if p.is_file() and p.suffix in ('.py','.yaml','.json','.sh')}


def predecessor_status(root,steps=3852):
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


def run_allocation_step(allocation,source,name,command,root,record,completed_run=None,steps=3852):
    allocation_ready(allocation)
    while not allocation_idle(allocation):
        record('waiting',stage='allocation-idle',next_stage=name)
        time.sleep(15)
        allocation_ready(allocation)
    # A surviving step can finish while a restarted watcher waits above.
    if completed_run is not None and predecessor_status(completed_run,steps)=='complete':
        record('skipped',stage=name,reason='run already completed while waiting')
        return
    record('running',stage=name)
    cmd=step_command(allocation,source,name,command)
    with (root/(name+'-'+str(time.time_ns())+'.log')).open('x') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{name} failed with exit {result.returncode}; remaining runs are blocked')


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--root')
    p.add_argument('--allocation');p.add_argument('--predecessor',required=True)
    p.add_argument('--seal',required=True);p.add_argument('--check-only',action='store_true')
    a=p.parse_args();source=Path(a.source).resolve()
    check_qualification(source,a.seal)
    if a.check_only:
        if predecessor_status(a.predecessor)!='complete':raise RuntimeError('R1 is not successfully complete')
        print('G3_M5_LAUNCH_CHECK_PASS');return
    if not a.root or not a.allocation:p.error('--root and --allocation required')
    root=Path(a.root).resolve();root.mkdir(parents=True,exist_ok=True)
    def record(status,**fields):
        data=dict(status=status,updated_unix=time.time(),allocation=a.allocation,node='worker-2',source=str(source),**fields)
        path=root/'queue_state.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)
        print(json.dumps(data),flush=True)
    with (root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            while True:
                allocation_ready(a.allocation);status=predecessor_status(a.predecessor)
                if status=='failed':raise RuntimeError('R1 failed; combined run is blocked')
                if status=='complete':break
                record('waiting',stage='R1',predecessor=a.predecessor);time.sleep(30)
            check_qualification(source,a.seal)
            run=root/'g3_m5_joint_e1'
            if predecessor_status(run)!='complete':
                run_allocation_step(a.allocation,source,'g3-m5-joint-e1',
                    ['bash',str(source/'train/launch_g3_m5_omega_joint_e1.sh'),str(run),a.seal,a.predecessor],
                    root,record,completed_run=run)
            if predecessor_status(run)!='complete':raise RuntimeError('G3_M5 exited before completed epoch')
            record('complete',stage='G3_M5')
        except BaseException as error:
            record('blocked',error=repr(error));raise


if __name__=='__main__':main()
