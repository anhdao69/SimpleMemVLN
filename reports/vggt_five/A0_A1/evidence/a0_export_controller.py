"""One-off A0-only handoff. A1 is never launched by this controller."""
import json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
R=Path('/mnt/data/vmo-ai-task/anhdh35');root=R/'a01-omega-worker2-20261014-retry'
source=root/'source-0c22a7b';qualification=root/'qualification-attempt-1'
g3=R/'SimpleMemVLN-c2_vggt_omega_g3-runs/g3_omega_joint_e1_20261013'
sys.path[:0]=[str(source),str(source/'src')]
from scripts.vln.queue_geoaddress import source_files,predecessor_status,allocation_ready,step_command
required={'full-suite-initial','A0-mixed','A0-rxrce_longest','A0-r2rce_longest','A0-threshold','A0-mixed_extreme','A0-recovery-reference','A0-recovery-resumed','A0-recovery-compare','A0-export'}
base=['srun','--jobid=4925','--overlap','--ntasks=1','--gpus=4','--cpus-per-task=60','--nodelist=worker-2']
passed=False

def state(stage,**kw):
    data=dict(stage=stage,time=time.time(),qualified_arms=['A0'] if passed else [],deferred_arms=['A1'],**kw)
    (root/'a0_state.json').write_text(json.dumps(data,indent=2));print(json.dumps(data),flush=True)

def stop_qualification():
    rows=subprocess.check_output(['squeue','--steps','-h','-j','4925','-o','%i|%j'],text=True)
    owned=[row.split('|')[0] for row in rows.splitlines() if row.endswith('|a0-export-final')]
    for sid in owned:
        subprocess.run(['scancel',sid],check=True)
        end=time.time()+180
        while sid+'|' in subprocess.check_output(['squeue','--steps','-h','-j','4925','-o','%i|%j'],text=True):
            if time.time()>end:raise RuntimeError('Owned export step failed to stop; no concurrent launch')
            time.sleep(2)

try:
    required.remove('A0-export')
    records=json.loads((qualification/'qualification.json').read_text())
    inherited=[r for r in records if r['name'] in required]
    if {r['name'] for r in inherited}!=required or any(r['returncode'] for r in inherited):
        raise RuntimeError('Incomplete preceding A0 training/recovery evidence')
    initial=json.loads((qualification/'source_start.json').read_text());current=source_files(source)
    changes={k for k in set(initial)|set(current) if initial.get(k)!=current.get(k)}
    if changes!={'scripts/vln/export_vggt_g1.py'}:
        raise RuntimeError('Changes exceed isolated exporter fix: '+repr(changes))
    for manifest in (qualification/'experiments').glob('A0-*/source_manifest.json'):
        if json.loads(manifest.read_text())!=initial:raise RuntimeError('Inherited training source mismatch')
    subprocess.run(['scancel','4925.37'],check=True)
    while '4925.37|' in subprocess.check_output(['squeue','--steps','-h','-j','4925','-o','%i|%j'],text=True):time.sleep(2)
    commands=[('export-regression',['python','-m','pytest','tests/vln/test_vggt_export.py','tests/vln/test_geoaddress_finalize.py','-q']),
      ('A0-export',['python','-m','scripts.vln.check_vggt_export','--source',str(qualification/'experiments/A0-recovery-resumed/train/final'),
       '--output',str(root/'A0-serving-export-final'),'--manifest',str(R/'SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl'),
       '--model-path',str(R/'cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a')])]
    for name,command in commands:
        state(name)
        with (root/(name+'-final.log')).open('x') as log:
            p=subprocess.Popen(step_command('4925',source,'a0-export-final',command),stdout=log,stderr=subprocess.STDOUT)
            try:rc=p.wait(timeout=900)
            except subprocess.TimeoutExpired:stop_qualification();raise
        if rc:raise RuntimeError(name+' failed: '+str(rc))
        inherited.append(dict(name=name,command=command,returncode=rc))
    if source_files(source)!=current:raise RuntimeError('Export qualification source changed')
    seal=root/'a0_qualification_pass.json'
    seal.write_text(json.dumps(dict(passed=True,qualified_arms=['A0'],deferred_arms=['A1'],source_files=current,records=inherited,
        inherited_source_files=initial,verified_changes=sorted(changes),
        scope='A0-only: verified identical training source; only exporter changed and retested on H100. Full prior regression, stress and recovery evidence retained. A1 NOT qualified.'),indent=2))
    passed=True;state('A0-qualified')
except BaseException as error:
    traceback.print_exc();state('A0-blocked',error=repr(error))
finally:
    stop_qualification()
    existing=set(g3.glob('supervisor-*'))
    with (root/'g3-resume-a0-final.log').open('a') as log:
        resume=subprocess.Popen(base+['--job-name=g3_omega_resume','bash',str(g3/'source/train/launch_g3_omega_joint_e1.sh'),str(g3)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (root/'g3_resume_submitted.json').write_text(json.dumps({'pid':resume.pid,'time':time.time(),'controller':'A0-only'}))
    state('G3-resume-submitted',pid=resume.pid)
if not passed:sys.exit(1)
end=time.time()+120
while not set(g3.glob('supervisor-*'))-existing:
    if time.time()>end:raise RuntimeError('G3 supervisor not observed; A0 stays blocked')
    time.sleep(2)
state('A0-queued-after-G3')
while True:
    allocation_ready('4925');status=predecessor_status(g3,3852)
    if status=='failed':state('A0-blocked-G3-failed');sys.exit(1)
    if status=='complete':break
    time.sleep(15)
data=json.loads(seal.read_text())
if data.get('qualified_arms')!=['A0'] or data['source_files']!=source_files(source):raise RuntimeError('A0 source/seal mismatch')
run=root/'a0_joint_e1';state('A0-training',run=str(run))
command=step_command('4925',source,'a0-joint-e1',['bash',str(source/'train/launch_a_omega_joint_e1.sh'),'A0',str(run),str(seal),str(g3)])
with (root/'a0-production.log').open('a') as log:
    result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
if result.returncode or predecessor_status(run,3852)!='complete':state('A0-training-failed',returncode=result.returncode);sys.exit(1)
state('A0-complete')
