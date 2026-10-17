"""Worker-resident, restartable sequential G3+M5 layer ablation controller."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from scripts.vln.queue_g3_m5 import source_files, predecessor_status

ARMS=('l8','l16','l8_l16_l24')
LAYERS={'l8':[8],'l16':[16],'l8_l16_l24':[8,16,24]}
ARM_GATES=('mixed-empty','mixed-empty-reversed','mixed','rxrce-longest',
           'r2rce-longest','recovery-reference','recovery-resumed','recovery-compare','export')
GATES=('cpu-and-cuda-suite','distributed-pairs')+tuple(f'{arm}/{gate}' for arm in ARMS for gate in ARM_GATES)


def write_seal(source,path,records,expected):
    names=[r.get('name') for r in records]
    if (len(names)!=len(set(names)) or set(names)!=set(GATES)
            or any(r.get('returncode')!=0 for r in records)
            or not expected or expected!=source_files(source)):
        raise ValueError('Incomplete, failed, or changed-source layer qualification')
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(passed=True,arms=LAYERS,records=records,source_files=expected),indent=2))
    temp.replace(path)


def check_seal(source,path):
    data=json.loads(Path(path).read_text())
    names=[r.get('name') for r in data.get('records',[])]
    if (data.get('passed') is not True or data.get('arms')!=LAYERS
            or not data.get('source_files') or data['source_files']!=source_files(source)
            or len(names)!=len(set(names)) or set(names)!=set(GATES)
            or any(r.get('returncode')!=0 for r in data['records'])):
        raise ValueError('Incomplete, failed, or changed-source layer qualification')
    return data


def bind_run(root,arm,source,seal):
    """Never adopt outputs from an unqualified or differently configured run."""
    root,source=Path(root),Path(source).resolve()
    data=check_seal(source,seal)
    identity=dict(arm=arm,layers=LAYERS[arm],source=str(source),source_files=data['source_files'])
    path=root/'run_identity.json'
    if path.exists():
        if json.loads(path.read_text())!=identity:raise ValueError('Run identity differs from qualified source/arm')
        return
    if root.exists() and any(root.iterdir()):raise ValueError('Refusing existing unbound run directory')
    root.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(identity,indent=2));temp.replace(path)


def arm_complete(root,arm):
    if predecessor_status(root)!='complete':return False
    config=json.loads((Path(root)/'train/checkpoint-3852/navigation.json').read_text())['config']
    research=config.get('research',{})
    if (research.get('run_id')!='G3_M5' or research.get('student_layers')!=LAYERS[arm]
            or 'student_layer' in research):
        raise ValueError('Completed checkpoint has wrong supervised layer identity')
    return True


def run_sequence(launch,complete,record):
    """No scheduler dependency races: this worker process owns every child."""
    arm=None
    try:
        for arm in ARMS:
            if complete(arm):
                record(status='skipped',arm=arm);continue
            record(status='running',arm=arm)
            launch(arm)
            if not complete(arm):raise RuntimeError(f'{arm} exited without complete epoch checkpoint')
            record(status='arm-complete',arm=arm)
        record(status='complete',arm=None)
    except BaseException as error:
        record(status='blocked',arm=arm,error=repr(error));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--seal',required=True)
    p.add_argument('--sources',nargs=3,required=True,help='Separate source worktrees in L8,L16,multi order')
    a=p.parse_args()
    if os.environ.get('SLURMD_NODENAME')!='worker-2' or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Controller must live inside the worker-2 Slurm allocation')
    root=Path(a.root).resolve();root.mkdir(parents=True,exist_ok=True)
    sources={arm:Path(source).resolve() for arm,source in zip(ARMS,a.sources,strict=True)}
    if len(set(sources.values()))!=3:raise ValueError('Independent source worktrees required')
    for source in sources.values():check_seal(source,a.seal)
    def record(**fields):
        data=dict(updated_unix=time.time(),pid=os.getpid(),node='worker-2',
                  allocation=os.environ['SLURM_JOB_ID'],sources={k:str(v) for k,v in sources.items()},**fields)
        path=root/'queue_state.json';temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(data,indent=2));temp.replace(path)
        print(json.dumps(data),flush=True)
    def complete(arm):return arm_complete(root/arm,arm)
    def launch(arm):
        source=sources[arm];check_seal(source,a.seal)
        command=['bash',str(source/'train/launch_g3_layers_joint_e1.sh'),str(root/arm),arm,a.seal]
        with (root/f'{arm}-{time.time_ns()}.log').open('x') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=source)
    def heartbeat(stop):
        while not stop.wait(15):
            path=root/'heartbeat.json';temp=path.with_suffix('.tmp')
            temp.write_text(json.dumps(dict(unix=time.time(),pid=os.getpid(),node='worker-2')));temp.replace(path)
    with (root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for arm,source in sources.items():bind_run(root/arm,arm,source,a.seal)
        stop=threading.Event();thread=threading.Thread(target=heartbeat,args=(stop,),daemon=True);thread.start()
        try:run_sequence(launch,complete,record)
        finally:stop.set();thread.join()


if __name__=='__main__':main()
