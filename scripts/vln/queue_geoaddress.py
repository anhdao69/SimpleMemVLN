"""Durable worker-2-only sequence: G3 -> qualification -> A0 -> A1.

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


def check_qualification(source,record):
    data=json.loads(Path(record).read_text())
    if data.get('passed') is not True or not data.get('source_files') or data['source_files']!=source_files(source):
        raise ValueError('Missing, failed, or source-mismatched A0/A1 qualification')
    return data


def predecessor_status(root,steps):
    root=Path(root);supervisors=sorted(root.glob('supervisor-*'))
    if not supervisors or not (supervisors[-1]/'resources.json').exists():
        return 'waiting'
    resources=json.loads((supervisors[-1]/'resources.json').read_text())
    if resources.get('status')!='PASS':
        return 'failed'
    from qwen_vl.train.recovery import validate_complete
    record=validate_complete(root/'train'/f'checkpoint-{steps}')
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


def main():
    p=argparse.ArgumentParser();p.add_argument('--predecessor',required=True);p.add_argument('--steps',type=int,default=3852)
    p.add_argument('--source',required=True);p.add_argument('--qualification');p.add_argument('--check-only',action='store_true')
    p.add_argument('--allocation');p.add_argument('--root');a=p.parse_args()
    if a.check_only:
        check_qualification(a.source,a.qualification)
        if predecessor_status(a.predecessor,a.steps)!='complete':raise RuntimeError('Predecessor not successfully complete')
        print('ADDRESS_LAUNCH_PREREQUISITES_PASS');return
    if not a.allocation or not a.root:p.error('--allocation and --root required for queue')
    root=Path(a.root).resolve();root.mkdir(parents=True,exist_ok=True)
    source=Path(a.source).resolve();seal=root/'qualification'/'qualification_pass.json'
    state=root/'queue_state.json'
    def record(status,**kw):
        item=dict(status=status,updated_unix=time.time(),allocation=a.allocation,node='worker-2',source=str(source),**kw)
        temp=state.with_suffix('.tmp');temp.write_text(json.dumps(item,indent=2));temp.replace(state)
        print(json.dumps(item),flush=True)
    def run_step(name,command):
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
                if status=='failed':raise RuntimeError('G3 predecessor failed; no new runs launched')
                if status=='complete':break
                record('waiting',stage='G3',predecessor=str(a.predecessor))
                time.sleep(30)
            if not seal.exists():
                run_step('a01-qualification',['python','-m','scripts.vln.qualify_geoaddress','--root',str(root/'qualification')])
            check_qualification(source,seal)
            previous=Path(a.predecessor)
            for arm in ('A0','A1'):
                run=root/(arm.lower()+'_joint_e1')
                # Completed runs are not trained twice. Interrupted training resumes
                # in the same output directory with optimizer/RNG/sampler state.
                status=predecessor_status(run,a.steps)
                if status!='complete':
                    if predecessor_status(previous,a.steps)!='complete':raise RuntimeError('Required predecessor incomplete')
                    check_qualification(source,seal)
                    run_step(arm.lower()+'-joint-e1',['bash',str(source/'train/launch_a_omega_joint_e1.sh'),arm,str(run),str(seal),str(previous)])
                    if predecessor_status(run,a.steps)!='complete':raise RuntimeError(f'{arm} exited without complete epoch checkpoint')
                previous=run
            record('complete',stage='A0+A1')
        except BaseException as error:
            record('blocked',error=repr(error))
            raise

if __name__=='__main__':main()
