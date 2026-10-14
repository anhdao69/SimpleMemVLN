"""Sequential complete-episode A0/A1 qualification; no production launch on failure."""
import argparse,json,os,subprocess,sys
from pathlib import Path


def verify_source(source,expected):
    from scripts.vln.queue_geoaddress import source_files
    if not expected or source_files(source)!=expected:
        raise ValueError("Qualification source changed during testing")


def write_qualification_seal(source,path,records,expected):
    from scripts.vln.queue_geoaddress import source_files
    if not records or any(r['returncode']!=0 for r in records):
        raise ValueError('Cannot seal failed/incomplete qualification')
    verify_source(source,expected)
    Path(path).write_text(json.dumps(dict(passed=True,records=records,source_files=expected),indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root);root.mkdir(parents=True,exist_ok=True);fixtures=Path('/mnt/data/vmo-ai-task/anhdh35/g1-omega-20261013/fixtures');records=[]
    from scripts.vln.queue_geoaddress import source_files
    source=Path.cwd();initial=source_files(source)
    (root/'source_start.json').write_text(json.dumps(initial,indent=2))
    def task(name,cmd):
        verify_source(source,initial)
        print('START',name,flush=True)
        with (root/(name+'.log')).open('x') as log:
            status=subprocess.run([sys.executable,*cmd],stdout=log,stderr=subprocess.STDOUT).returncode
        records.append(dict(name=name,command=cmd,returncode=status));(root/'qualification.json').write_text(json.dumps(records,indent=2))
        print('DONE',name,status,flush=True)
        if status:raise SystemExit(status)
        verify_source(source,initial)
    def profile(arm,name,fixture,updates=3,extra=()):
        label=arm+'-'+name
        task(label,['-m','scripts.vln.profile_vggt_g1','--arm',arm,'--root',str(root),'--name',label,'--fixture',str(fixture),
            '--updates',str(updates),'--cpu-adam','--threads','4','--workers','8','--teacher-batch','8',
            '--attention-batch','32','--vision-batch','8',*extra])
        if json.loads((root/'experiments'/label/'source_manifest.json').read_text())!=initial:
            raise ValueError('Profile used different source: '+label)
        result=json.loads((root/'experiments'/label/'result.json').read_text())
        assert result['passed'] and result['check']['peak_allocated_gib']<=72,label
    task('full-suite-initial',['-m','pytest','tests/vln','-q'])
    task('teacher-qc',['-m','scripts.vln.check_geoaddress_teacher','--manifest',
        '/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl','--out',str(root/'teacher-qc')])
    for arm in ('A0','A1'):
        profile(arm,'mixed',fixtures/'mixed_tail.jsonl')
        if arm=='A1':profile(arm,'mixed-empty',Path('/mnt/data/vmo-ai-task/anhdh35/g3-omega-20261013/mixed_empty.jsonl'))
        for name in ('rxrce_longest','r2rce_longest','threshold','mixed_extreme'):
            profile(arm,name,fixtures/(name+'.jsonl'))
        profile(arm,'recovery-reference',fixtures/'mixed_tail.jsonl',extra=['--save'])
        ref=root/'experiments'/f'{arm}-recovery-reference/train';res=root/'experiments'/f'{arm}-recovery-resumed/train'
        profile(arm,'recovery-resumed',fixtures/'mixed_tail.jsonl',extra=['--save','--resume',str(ref/'checkpoint-2'),'--first-update','3'])
        task(arm+'-recovery-compare',['-m','scripts.vln.check_geoaddress_recovery','--reference',str(ref),'--resumed',str(res),'--out',str(root/(arm+'-recovery.json'))])
        task(arm+'-export',['-m','scripts.vln.check_vggt_export','--source',str(res/'final'),'--output',str(root/(arm+'-serving-export')),
            '--manifest','/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl','--model-path',os.environ['VLN_MODEL_PATH']])
        for dataset in ('r2rce','rxrce'):
            profile(arm,dataset+'-timing',fixtures/(dataset+'_representative.jsonl'),updates=8)
    task('distributed-pair-normalization',['-m','torch.distributed.run','--standalone','--nproc_per_node=4','-m','scripts.vln.check_vggt_g3_reduction'])
    task('full-suite-final',['-m','pytest','tests/vln','-q'])
    write_qualification_seal(Path.cwd(),root/'qualification_pass.json',records,initial)
    print('GEOADDRESS_QUALIFICATION_PASS',flush=True)

if __name__=='__main__':main()
