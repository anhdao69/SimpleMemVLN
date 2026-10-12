import sys,subprocess,json,time
from pathlib import Path
from scripts.vln import submit_episodic as s
original=subprocess.run
C=Path('/mnt/data/vmo-ai-task/anhdh35/checkpoints/SimpleMemVLN-c2-h2-speed-20261011')
def held(command,*args,**kwargs):
    if isinstance(command,list) and command and command[0]=='sbatch':
        command=[command[0],'--hold','--dependency=afterany:4876',*command[1:]]
        result=original(command,*args,**kwargs)
        with (C/'submission_ledger.jsonl').open('a') as f:
            f.write(json.dumps(dict(event='held_submission',command=command,job_id=result.stdout.strip().split(';')[0],time=time.time()))+'\n')
        return result
    return original(command,*args,**kwargs)
subprocess.run=held
s.main()
