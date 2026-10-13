"""Sequential, fail-closed G3 runtime/recovery qualification in an owned4GPU step."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root);fixtures=Path('/mnt/data/vmo-ai-task/anhdh35/g1-omega-20261013/fixtures')
    gate=root/'experiments/longest-rxr/result.json'
    deadline=time.monotonic()+1800
    while not gate.exists():
        if time.monotonic()>deadline:raise TimeoutError('Longest test did not complete')
        time.sleep(5)
    result=json.loads(gate.read_text());assert result['passed'] and result['check']['peak_allocated_gib']<72
    records=[]
    def task(name,command):
        print('START',name,flush=True)
        with (root/(name+'.log')).open('x') as log:
            code=subprocess.run([sys.executable,*command],stdout=log,stderr=subprocess.STDOUT).returncode
        records.append(dict(name=name,command=command,returncode=code))
        (root/'qualification.json').write_text(json.dumps(records,indent=2))
        print('DONE',name,code,flush=True)
        if code:raise SystemExit(code)
    def profile(name,fixture,updates=3,extra=()):
        task(name,['-m','scripts.vln.profile_vggt_g1','--arm','G3','--root',str(root),
            '--name',name,'--fixture',str(fixture),'--updates',str(updates),'--cpu-adam',
            '--threads','4','--workers','8','--teacher-batch','8','--attention-batch','32',
            '--vision-batch','8',*extra])
    profile('mixed-teacher4',fixtures/'mixed_tail.jsonl',extra=['--teacher-batch','4'])
    profile('mixed-empty',root/'mixed_empty.jsonl')
    profile('r2r-representative',fixtures/'r2rce_representative.jsonl',updates=8)
    profile('rxr-representative',fixtures/'rxrce_representative.jsonl',updates=8)
    profile('recovery-reference',fixtures/'mixed_tail.jsonl',extra=['--save'])
    ref=root/'experiments/recovery-reference/train';res=root/'experiments/recovery-resumed/train'
    profile('recovery-resumed',fixtures/'mixed_tail.jsonl',extra=['--save','--resume',str(ref/'checkpoint-2'),'--first-update','3'])
    task('recovery-compare',['scripts/vln/check_vggt_recovery.py','--reference',str(ref),'--resumed',str(res),'--out',str(root/'recovery.json')])
    task('export-check',['scripts/vln/check_vggt_export.py','--source',str(res/'final'),
        '--output',str(root/'serving-export'),'--manifest',
        '/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl',
        '--model-path',os.environ['VLN_MODEL_PATH']])
    task('distributed-pairs',['-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                             '-m','scripts.vln.check_vggt_g3_reduction'])
    task('full-suite',['-m','pytest','tests/vln','-q'])
    print('G3_QUALIFICATION_PASS',flush=True)


if __name__=='__main__':main()
