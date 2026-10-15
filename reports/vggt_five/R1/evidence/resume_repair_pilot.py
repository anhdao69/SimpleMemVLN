"""Campaign recovery audit, then leave the full R1 epoch running independently."""
import json
import os
import subprocess
import time
from pathlib import Path

from scripts.vln.queue_vggt_r1 import check_qualification
from qwen_vl.train.recovery import completed_checkpoints, validate_complete

B = Path('/mnt/data/vmo-ai-task/anhdh35/r1-omega-worker2-20261015')
SOURCE = B/'source-bcb14ea'
ROOT = B/'attempt3'
RUN = B/'attempt2/r1_joint_e1'
TRAIN = RUN/'train'
FLAG = TRAIN/'REQUEST_CHECKPOINT_AND_STOP'
STATE = B/'repair_resume_state.json'


def record(status, **fields):
    result = dict(status=status, unix=time.time(), **fields)
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2)); temporary.replace(STATE)
    print(json.dumps(result), flush=True)


def progress():
    result = []
    for rank in range(4):
        path = TRAIN/f'profile_rank{rank}.jsonl'
        updates = []
        for line in path.read_text().splitlines() if path.exists() else []:
            try: updates.append(json.loads(line)['update'])
            except (ValueError, KeyError): pass
        result.append(max(updates, default=0))
    return result


def launch(label):
    command = ['bash',str(SOURCE/'train/g1_omega_env.sh'),'python','-m','scripts.vln.queue_vggt_r1',
               '--source',str(SOURCE),'--root',str(ROOT),'--allocation','4925']
    with (ROOT/(label+'.log')).open('x') as log:
        process = subprocess.Popen(command, env=dict(os.environ,G1_CODE_DIR=str(SOURCE)),
                                   stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    return process


def main():
    record('waiting-for-repaired-qualification')
    deadline = time.monotonic()+1800
    seal = ROOT/'qualification/qualification_pass.json'
    while not seal.exists():
        if time.monotonic()>deadline: raise TimeoutError('Qualification did not complete')
        log = B/'repair_precursors.log'
        if log.exists() and 'Traceback' in log.read_text(): raise RuntimeError('Repair qualification failed')
        time.sleep(5)
    check_qualification(SOURCE,seal)
    checkpoint10 = validate_complete(TRAIN/'checkpoint-10')
    assert checkpoint10['step'] == 10 and checkpoint10['world_size'] == 4
    assert completed_checkpoints(TRAIN)[-1][0] == 10
    assert not FLAG.exists()
    ROOT.mkdir(exist_ok=True)
    link = ROOT/'r1_joint_e1'
    if not link.exists(): link.symlink_to(RUN, target_is_directory=True)
    assert link.is_symlink() and link.resolve() == RUN
    first = launch('queue-repair-resume')
    record('resuming-checkpoint10',queue_pid=first.pid)
    deadline = time.monotonic()+1800
    while min(progress()) < 20:
        if first.poll() is not None: raise RuntimeError('Queue exited before update20')
        if time.monotonic()>deadline: raise TimeoutError('Resume did not pass update19')
        time.sleep(5)
    FLAG.write_text('4925\n')
    record('requesting-post-repair-checkpoint',rank_updates=progress())
    first.wait(timeout=900)
    # An intentionally incomplete epoch leaves the queue blocked after the
    # successful checkpoint stop. Validate the save/supervisor before restart.
    checkpoint = completed_checkpoints(TRAIN)[-1]
    step, path, saved = checkpoint
    assert 20 <= step < 3852 and saved['world_size'] == 4
    trainer = json.loads((path/'trainer_state.json').read_text())
    assert trainer['max_steps'] == 3852 and trainer['num_train_epochs'] == 1
    assert trainer['global_step'] == step
    latest = sorted(RUN.glob('supervisor-*'))[-1]
    resources = json.loads((latest/'resources.json').read_text())
    assert resources['status'] == 'PASS'
    assert FLAG.read_text().strip() == '4925'
    FLAG.rename(TRAIN/f'REQUEST_CHECKPOINT_AND_STOP.repair-{step}.handled')
    proof = dict(checkpoint=str(path),inventory=saved,trainer_state=trainer,
                 resources=resources,queue_stop_returncode=first.returncode)
    (B/'repair_checkpoint.json').write_text(json.dumps(proof,indent=2))
    second = launch('queue-production-resumed')
    record('full-training-restarted',checkpoint=step,queue_pid=second.pid)
    deadline = time.monotonic()+1200
    while min(progress()) < step+2:
        if second.poll() is not None: raise RuntimeError('Production queue exited after checkpoint recovery')
        if time.monotonic()>deadline: raise TimeoutError('Production did not advance after checkpoint recovery')
        time.sleep(5)
    record('passed-full-training-running',checkpoint=step,queue_pid=second.pid,rank_updates=progress())


if __name__ == '__main__':
    try: main()
    except BaseException as error:
        record('blocked',error=repr(error))
        raise
