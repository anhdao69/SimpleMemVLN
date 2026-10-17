"""Deployment recipe: CPU-only waiting step, then exec the qualified worker queue.

Lives outside the training source tree. No GPU allocation/use occurs in Python
until the successful seal exists and other compute steps have finished.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def gate_status(seal_exists, records, active_steps, qualification_step, own_step, allocation):
    if any(record.get('returncode')!=0 for record in records):
        raise RuntimeError('Qualification failed; training remains blocked')
    allowed={own_step,qualification_step,f'{allocation}.0',f'{allocation}.extern',f'{allocation}.batch'}
    if seal_exists:
        # Wait for the qualifying step to exit as well as any unrelated step.
        return 'ready' if not (set(active_steps)-allowed) and qualification_step not in active_steps else 'waiting-for-idle'
    if qualification_step not in active_steps:
        raise RuntimeError('Qualification exited without a successful seal')
    return 'waiting-for-qualification'


def main():
    root=Path(sys.argv[1]).resolve();qualification_step=sys.argv[2]
    if os.environ.get('SLURMD_NODENAME')!='worker-2':raise RuntimeError('worker-2 only')
    allocation=os.environ['SLURM_JOB_ID'];own_step=f"{allocation}.{os.environ['SLURM_STEP_ID']}"
    qualification=root/'qualification-f4d1c2f';seal=qualification/'qualification_pass.json'
    while True:
        path=qualification/'qualification.json'
        records=json.loads(path.read_text()) if path.exists() else []
        active=subprocess.check_output(['squeue','--steps','-h','-j',allocation,'-o','%i'],text=True,timeout=30).split()
        try:status=gate_status(seal.exists(),records,active,qualification_step,own_step,allocation)
        except RuntimeError as error:
            status='blocked';detail=str(error)
        else:detail=None
        state=dict(status=status,error=detail,unix=time.time(),step=own_step,qualification_step=qualification_step)
        target=root/'launch_gate_state.json';temp=target.with_suffix('.tmp')
        temp.write_text(json.dumps(state,indent=2));temp.replace(target)
        print(json.dumps(state),flush=True)
        if status=='blocked':raise RuntimeError(detail)
        if status=='ready':break
        time.sleep(15)
    sources=[str(root/f'source-f4d1c2f-{arm}') for arm in ('l8','l16','l8_l16_l24')]
    command=[sys.executable,'-m','scripts.vln.queue_g3_layers','--root',str(root/'production'),
             '--seal',str(seal),'--sources',*sources]
    os.execv(sys.executable,command)


if __name__=='__main__':main()
