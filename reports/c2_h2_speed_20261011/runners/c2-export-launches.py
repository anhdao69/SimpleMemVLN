from pathlib import Path
import json,shlex
C=Path('/mnt/data/vmo-ai-task/anhdh35/checkpoints/SimpleMemVLN-c2-h2-speed-20261011')
rows=[]
for p in sorted(C.glob('runs/*/*/launch_manifest.json')):
 d=json.loads(p.read_text());a=p.parent;s=d['resources']
 if d['attempt']=='P1-V0-s429-H2-a1':continue
 command=['sbatch','--parsable','--job-name='+d['job_name'],'--comment='+d['comment'],'--partition='+d['partition'],'--nodes=1','--ntasks=1',f'--gres=gpu:nvidia_h100_80gb_hbm3:{s["gpus"]}',f'--cpus-per-task={s["cpus"]}',f'--mem={s["mem"]}',f'--time={s["time"]}','--signal=B:USR1@1800',f'--output={a}/slurm-%j.log',d['source_dir']+'/train/slurm/episodic_candidate.sbatch',str(a)]
 rows.append(dict(run=d['run_id'],attempt=d['attempt'],attempt_dir=str(a),command=command,launch=d,status=json.loads((a/'status.json').read_text())))
(C/'recovery_commands.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
