"""Worker-resident, restartable G3+A1 continuation followed by qualified G1-future.

Only a complete epoch permits the next stage. An early stop or failed child
blocks the sequence; existing complete stages are never trained twice.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def source_files(root):
    root=Path(root)
    return {str(p.relative_to(root)):file_hash(p)
            for directory in ('src','scripts','configs','train')
            for p in sorted((root/directory).rglob('*'))
            if p.is_file() and p.suffix in ('.py','.yaml','.json','.sh')}


def check_source(root,expected):
    if not expected or source_files(root)!=expected:
        raise ValueError('Qualified source changed')


def stage_complete(stage):
    from qwen_vl.train.recovery import validate_complete
    train=Path(stage['run'])/'train';steps=stage['steps']
    checkpoint=train/f'checkpoint-{steps}';final=train/'final'
    required=[checkpoint/'RECOVERY_COMPLETE.json',final/'export_info.json',
              final/'pytorch_model.bin',final/'reload_reference.json',train/'trainer_state.json']
    if not all(p.is_file() and p.stat().st_size for p in required):return False
    marker=validate_complete(checkpoint)
    if marker['step']!=steps or marker['world_size']!=4 or not marker['epoch_boundary']:
        raise ValueError('Final checkpoint is not the expected complete four-rank epoch')
    navigation=json.loads((checkpoint/'navigation.json').read_text())
    if navigation!=stage['navigation']:raise ValueError('Completed run contract differs')
    for path in (checkpoint/'trainer_state.json',train/'trainer_state.json',final/'export_info.json'):
        state=json.loads(path.read_text())
        if state['global_step']!=steps or state['max_steps']!=steps or state['epoch']<1:
            return False
    if not math.isfinite(json.loads((final/'reload_reference.json').read_text())['loss_sum']):
        raise ValueError('Nonfinite final forward reference')
    return True


def run_sequence(names,launch,complete,record):
    name=None
    try:
        for name in names:
            if complete(name):record(status='already-complete',stage=name);continue
            record(status='running',stage=name)
            launch(name)
            if not complete(name):raise RuntimeError(f'{name} exited without a complete epoch')
            record(status='stage-complete',stage=name)
        record(status='complete',stage=None)
    except BaseException as error:
        record(status='blocked',stage=name,error=repr(error));raise


def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def launch_after_existing(lock_path,complete,launch):
    """A restarted controller waits for an orphan's per-run launcher lock.

    Both production launchers hold this same lock until their supervisor exits.
    The outer queue lock excludes competing controllers during this handoff.
    Never hold the launcher lock while starting the launcher that acquires it.
    """
    lock_path=Path(lock_path);lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if complete():return
    launch()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    if os.environ.get('SLURMD_NODENAME')!='worker-2' or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Queue must run within worker-2 Slurm allocation')
    config=json.loads(Path(a.config).read_text());root=Path(config['root']);root.mkdir(parents=True,exist_ok=True)
    stages=config['stages'];names=[s['name'] for s in stages]
    if names!=['g3a1','g1_future']:raise ValueError('Incorrect training order')
    by_name={s['name']:s for s in stages}
    def validate(stage):
        check_source(stage['source'],stage['source_files'])
        if not stage.get('pinned_files'):raise ValueError('Missing pinned input files')
        for path,digest in stage['pinned_files'].items():
            if file_hash(path)!=digest:raise ValueError(f'Pinned input changed: {path}')
    def record(**fields):
        data=dict(unix=time.time(),pid=os.getpid(),allocation=os.environ['SLURM_JOB_ID'],**fields)
        atomic_json(root/'queue_state.json',data);print(json.dumps(data),flush=True)
    def launch(name):
        stage=by_name[name];validate(stage)
        environment=dict(os.environ,**stage.get('environment',{}))
        def start():
            with (root/f'{name}-{time.time_ns()}.log').open('x') as log:
                subprocess.run(stage['command'],cwd=stage['source'],env=environment,
                               stdout=log,stderr=subprocess.STDOUT,check=True)
        launch_after_existing(Path(stage['run'])/'launcher.lock',lambda:stage_complete(stage),start)
    def heartbeat(stop):
        while not stop.wait(15):atomic_json(root/'heartbeat.json',dict(unix=time.time(),pid=os.getpid()))
    with (root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=root/'queue_identity.json'
        if identity.exists() and json.loads(identity.read_text())!=config:
            raise ValueError('Queue configuration differs from previous invocation')
        for stage in stages:validate(stage)
        atomic_json(identity,config)
        stop=threading.Event();thread=threading.Thread(target=heartbeat,args=(stop,),daemon=True);thread.start()
        try:run_sequence(names,launch,lambda name:stage_complete(by_name[name]),record)
        finally:stop.set();thread.join()


if __name__=='__main__':main()
