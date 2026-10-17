"""User-directed queue override: preserve stopped L16 and run the triple arm."""
import fcntl,json,os,subprocess,threading,time
from pathlib import Path
from scripts.vln.queue_g3_layers import check_seal,bind_run,arm_complete

root=Path('/mnt/data/vmo-ai-task/anhdh35/g3-m5-layers-worker2-20261017')
production=root/'production';arm='l8_l16_l24'
sources={a:root/f'source-f4d1c2f-{a}' for a in ('l8','l16',arm)}
seal=root/'qualification-f4d1c2f/qualification_pass.json'
assert os.environ.get('SLURMD_NODENAME')=='worker-2'
assert os.environ.get('SLURM_JOB_ID')=='4925'
assert json.loads((root/'publication-l8-epoch1-20261017/UPLOAD_VERIFIED.json').read_text())['passed']
assert (production/'l16/USER_STOPPED.json').exists()

def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)

def record(status,**fields):
    data=dict(updated_unix=time.time(),pid=os.getpid(),node='worker-2',
        allocation=os.environ['SLURM_JOB_ID'],step=os.environ.get('SLURM_STEP_ID'),
        sources={k:str(v) for k,v in sources.items()},status=status,arm=arm,
        requested_sequence=[arm],omitted={'l16':'stopped by explicit user request; checkpoint preserved'},**fields)
    write(production/'queue_state.json',data);print(json.dumps(data),flush=True)

def heartbeat(stop):
    while not stop.wait(15):
        write(production/'heartbeat.json',dict(unix=time.time(),pid=os.getpid(),node='worker-2'))

with (production/'queue.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name,source in sources.items():
        check_seal(source,seal);bind_run(production/name,name,source,seal)
    assert arm_complete(production/'l8','l8')
    old=production/'queue_state.json'
    if old.exists():
        history=production/'queue-history';history.mkdir(exist_ok=True)
        (history/f'before-user-override-{time.time_ns()}.json').write_text(old.read_text())
    stop=threading.Event();thread=threading.Thread(target=heartbeat,args=(stop,),daemon=True);thread.start()
    try:
        if not arm_complete(production/arm,arm):
            record('running')
            command=['bash',str(sources[arm]/'train/launch_g3_layers_joint_e1.sh'),str(production/arm),arm,str(seal)]
            with (production/f'{arm}-user-override-{time.time_ns()}.log').open('x') as log:
                subprocess.run(command,cwd=sources[arm],stdout=log,stderr=subprocess.STDOUT,check=True)
            assert arm_complete(production/arm,arm),'Training exited without complete epoch checkpoint'
        record('complete',completed_arms=['l8',arm])
    except BaseException as error:
        record('blocked',error=repr(error));raise
    finally:
        stop.set();thread.join()
