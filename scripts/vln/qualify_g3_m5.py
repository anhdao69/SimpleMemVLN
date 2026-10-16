"""One owned four-H100 step: measure, stress, recover, export, seal."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from scripts.vln.queue_g3_m5 import source_files,write_seal


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root).resolve();root.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve().parents[2];expected=source_files(source);records=[]
    base=Path('/mnt/data/vmo-ai-task/anhdh35');fixtures=base/'g1-omega-20261013/fixtures'
    def task(name,command,timeout=2400):
        print('START',name,flush=True);start=time.time()
        with (root/(name+'.log')).open('x') as log:
            proc=subprocess.Popen([sys.executable,*command],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=45)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                code=124
        records.append(dict(name=name,command=command,returncode=code,seconds=time.time()-start))
        (root/'qualification.json').write_text(json.dumps(records,indent=2))
        print('DONE',name,code,flush=True)
        if code:raise RuntimeError(f'{name} failed: {code}')
    def profile(name,fixture,updates=3,attention=32,extra=()):
        task(name,['-m','scripts.vln.profile_vggt_g1','--arm','G3_M5','--root',str(root),
            '--name',name,'--fixture',str(fixture),'--updates',str(updates),'--cpu-adam',
            '--threads','4','--workers','8','--teacher-batch','8','--attention-batch',str(attention),
            '--vision-batch','8',*extra])
    task('cpu-and-cuda-suite',['-m','pytest','tests/vln','-q'],timeout=1800)
    # Exact original failing R1 batch, in both accumulation-slot orders.
    empty=base/'r1-omega-worker2-20261015/stall-update19-fixture.jsonl'
    rows=empty.read_text().splitlines();assert len(rows)==8
    reverse=root/'mixed_empty_reversed.jsonl';reverse.write_text('\n'.join(rows[4:]+rows[:4])+'\n')
    profile('mixed-empty',empty)
    profile('mixed-empty-reversed',reverse)
    profile('mixed',fixtures/'mixed_tail.jsonl')
    profile('attention16',fixtures/'mixed_tail.jsonl',attention=16)
    profile('attention64',fixtures/'mixed_tail.jsonl',attention=64)
    # Keep qualified32 unless alternate repeat measurements justify changing it.
    profile('rxrce-longest',fixtures/'rxrce_longest.jsonl',updates=1)
    profile('r2rce-longest',fixtures/'r2rce_longest.jsonl',updates=1)
    profile('recovery-reference',fixtures/'mixed_tail.jsonl',extra=['--save'])
    ref=root/'experiments/recovery-reference/train';res=root/'experiments/recovery-resumed/train'
    profile('recovery-resumed',fixtures/'mixed_tail.jsonl',extra=['--save','--resume',str(ref/'checkpoint-2'),'--first-update','3'])
    task('recovery-compare',['-m','scripts.vln.check_vggt_recovery','--reference',str(ref),'--resumed',str(res),'--out',str(root/'recovery.json')])
    task('export',['-m','scripts.vln.check_g3_m5_export','--source',str(res/'final'),'--output',str(root/'serving-export'),
        '--manifest',str(base/'SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl'),'--model-path',os.environ['VLN_MODEL_PATH']])
    task('distributed-pairs',['-m','torch.distributed.run','--standalone','--nproc_per_node=4','-m','scripts.vln.check_vggt_g3_reduction'])
    profile('r2rce-timing',fixtures/'r2rce_representative.jsonl',updates=8)
    profile('rxrce-timing',fixtures/'rxrce_representative.jsonl',updates=8)
    write_seal(source,root/'qualification_pass.json',records,expected)
    print('G3_M5_QUALIFICATION_PASS',flush=True)


if __name__=='__main__':main()
