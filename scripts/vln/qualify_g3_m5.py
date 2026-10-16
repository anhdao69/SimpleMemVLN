"""One owned four-H100 step: measure, stress, recover, export, seal."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from scripts.vln.queue_g3_m5 import source_files,write_seal


def validate_empty_exposures(ranks,slot):
    """Require mixed support at a real accumulation slot, after length sorting."""
    if len(ranks)!=4 or slot not in (0,1):raise ValueError('Four ranks and slot0/1 required')
    updates={e['update'] for e in ranks[0]}
    if not updates:raise ValueError('No distributed exposures')
    for update in updates:
        counts=[]
        for rank in ranks:
            batch=[e for e in rank if e['update']==update]
            if len(batch)!=2:raise ValueError('Expected two complete episodes per rank/update')
            counts.append(batch[slot]['aux_pair_count'])
        if not min(counts)==0<max(counts):raise ValueError(f'No mixed support at slot{slot}: {counts}')


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
    # Original failing R1 batch: mixed support in the second slot. Five
    # complete short episodes + three supported ones place empty support in
    # the first slot even after the real sampler's descending length sort.
    empty=base/'r1-omega-worker2-20261015/stall-update19-fixture.jsonl'
    rows=empty.read_text().splitlines();assert len(rows)==8
    episodes=[json.loads(row) for row in rows]
    from qwen_vl.research.vggt_g3 import sample_pairs
    from qwen_vl.research.vggt_sampling import collection_trajectory_key
    supported=[e for e in episodes if sample_pairs(collection_trajectory_key(e),len(e['steps']))]
    unsupported=[e for e in episodes if not sample_pairs(collection_trajectory_key(e),len(e['steps']))]
    assert supported and unsupported
    first_slot=supported[:3]+[unsupported[0]]*5
    reverse=root/'mixed_empty_first_slot.jsonl';reverse.write_text(''.join(json.dumps(e)+'\n' for e in first_slot))
    profile('mixed-empty',empty)
    profile('mixed-empty-reversed',reverse)
    for name,slot in [('mixed-empty',1),('mixed-empty-reversed',0)]:
        ranks=[[json.loads(line) for line in (root/'experiments'/name/'train'/f'exposures_rank{r}.jsonl').read_text().splitlines()] for r in range(4)]
        validate_empty_exposures(ranks,slot)
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
