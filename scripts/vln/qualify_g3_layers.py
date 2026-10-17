"""All-arm qualification in one owned worker-2 step; immutable source gates."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from scripts.vln.queue_g3_m5 import source_files
from scripts.vln.queue_g3_layers import ARMS, LAYERS, write_seal
from scripts.vln.qualify_g3_m5 import validate_empty_exposures


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    if os.environ.get('SLURMD_NODENAME')!='worker-2':raise RuntimeError('worker-2 only')
    root=Path(a.root).resolve();root.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve().parents[2];expected=source_files(source)
    identity=root/'source_manifest.json'
    if identity.exists() and json.loads(identity.read_text())!=expected:raise ValueError('Qualification source changed')
    identity.write_text(json.dumps(expected,indent=2))
    progress=root/'qualification.json';records=json.loads(progress.read_text()) if progress.exists() else []
    base=Path('/mnt/data/vmo-ai-task/anhdh35');fixtures=base/'g1-omega-20261013/fixtures'
    def task(name,command,timeout=3600):
        if any(r['name']==name and r['returncode']==0 for r in records):return
        print('START',name,flush=True);start=time.time()
        logpath=root/(name.replace('/','-')+f'-{time.time_ns()}.log')
        with logpath.open('x') as log:
            proc=subprocess.Popen([sys.executable,*command],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=45)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                code=124
        records[:]=[r for r in records if r['name']!=name]
        records.append(dict(name=name,command=command,returncode=code,seconds=time.time()-start,log=str(logpath)))
        temp=progress.with_suffix('.tmp');temp.write_text(json.dumps(records,indent=2));temp.replace(progress)
        print('DONE',name,code,flush=True)
        if code:raise RuntimeError(f'{name} failed: {code}; no training is authorized by this failed qualification')
    def profile(arm,name,fixture,updates=3,extra=()):
        task(f'{arm}/{name}',['-m','scripts.vln.profile_vggt_g1','--arm','G3_M5','--root',str(root/arm),
            '--name',name,'--fixture',str(fixture),'--updates',str(updates),'--cpu-adam',
            '--threads','4','--workers','8','--teacher-batch','8','--attention-batch','32',
            '--vision-batch','8','--student-layers',*[str(i) for i in LAYERS[arm]],*extra])
    task('cpu-and-cuda-suite',['-m','pytest','tests/vln','-q'])
    os.environ['G3_REDUCTION_HEADS']='3'
    task('distributed-pairs',['-m','torch.distributed.run','--standalone','--nproc_per_node=4','-m','scripts.vln.check_vggt_g3_reduction'])
    empty=base/'r1-omega-worker2-20261015/stall-update19-fixture.jsonl'
    episodes=[json.loads(row) for row in empty.read_text().splitlines()]
    from qwen_vl.research.vggt_g3 import sample_pairs
    from qwen_vl.research.vggt_sampling import collection_trajectory_key
    supported=[e for e in episodes if sample_pairs(collection_trajectory_key(e),len(e['steps']))]
    unsupported=[e for e in episodes if not sample_pairs(collection_trajectory_key(e),len(e['steps']))]
    assert len(supported)>=3 and unsupported
    reverse=root/'mixed_empty_first_slot.jsonl'
    reverse.write_text(''.join(json.dumps(e)+'\n' for e in supported[:3]+[unsupported[0]]*5))
    # Qualify the largest memory footprint first, while the production sequence
    # remains the requested L8 -> L16 -> all-three.
    for arm in reversed(ARMS):
        profile(arm,'mixed-empty',empty)
        profile(arm,'mixed-empty-reversed',reverse)
        for name,slot in [('mixed-empty',1),('mixed-empty-reversed',0)]:
            ranks=[[json.loads(line) for line in (root/arm/'experiments'/name/'train'/f'exposures_rank{r}.jsonl').read_text().splitlines()] for r in range(4)]
            validate_empty_exposures(ranks,slot)
        profile(arm,'mixed',fixtures/'mixed_tail.jsonl',updates=6)
        # Three complete updates = six longest episodes per GPU, including
        # optimizer steps and repeated arena reuse, with full BPTT.
        profile(arm,'rxrce-longest',fixtures/'rxrce_longest.jsonl',updates=3)
        profile(arm,'r2rce-longest',fixtures/'r2rce_longest.jsonl',updates=3)
        profile(arm,'recovery-reference',fixtures/'mixed_tail.jsonl',extra=['--save'])
        ref=root/arm/'experiments/recovery-reference/train';res=root/arm/'experiments/recovery-resumed/train'
        profile(arm,'recovery-resumed',fixtures/'mixed_tail.jsonl',extra=['--save','--resume',str(ref/'checkpoint-2'),'--first-update','3'])
        task(f'{arm}/recovery-compare',['-m','scripts.vln.check_vggt_recovery','--reference',str(ref),'--resumed',str(res),'--out',str(root/arm/'recovery.json')])
        task(f'{arm}/export',['-m','scripts.vln.check_g3_m5_export','--source',str(res/'final'),'--output',str(root/arm/'serving-export'),
            '--manifest',str(base/'SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl'),'--model-path',os.environ['VLN_MODEL_PATH']])
    write_seal(source,root/'qualification_pass.json',records,expected)
    print('G3_M5_LAYERS_QUALIFICATION_PASS',flush=True)


if __name__=='__main__':main()
