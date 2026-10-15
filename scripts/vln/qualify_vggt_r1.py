"""Real complete-episode R1 qualification under the selected production profile."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import yaml
from scripts.vln.queue_vggt_r1 import source_files,write_seal,check_baseline


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root);root.mkdir(parents=True,exist_ok=True);source=Path.cwd();records=[]
    expected=source_files(source);(root/'source_start.json').write_text(json.dumps(expected,indent=2))
    cfg=yaml.safe_load(Path('configs/vggt/r1_omega_h4.yaml').read_text());check_baseline(cfg)
    base=Path('/mnt/data/vmo-ai-task/anhdh35');fixtures=base/'g1-omega-20261013/fixtures'
    manifest=base/'SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl'
    check_baseline(cfg,manifest)
    def task(name,args):
        if source_files(source)!=expected:raise ValueError('Source changed during qualification')
        print('START',name,flush=True)
        with (root/(name+'.log')).open('x') as log:
            rc=subprocess.run([sys.executable,*args],stdout=log,stderr=subprocess.STDOUT).returncode
        records.append(dict(name=name,command=args,returncode=rc));(root/'qualification.json').write_text(json.dumps(records,indent=2))
        print('DONE',name,rc,flush=True)
        if rc:raise SystemExit(rc)
    def profile(name,fixture,updates=3,extra=()):
        task(name,['-m','scripts.vln.profile_vggt_g1','--arm','R1','--root',str(root),'--name',name,
            '--fixture',str(fixture),'--updates',str(updates),'--cpu-adam','--threads','4',
            '--workers',str(cfg['training']['dataloader_num_workers']),'--teacher-batch',str(cfg['research']['teacher_batch_size']),
            '--attention-batch',str(cfg['runtime']['window_attention_batch_steps']),
            '--vision-batch',str(cfg['runtime']['vision_microbatch_images']),*extra])
        folder=root/'experiments'/name;result=json.loads((folder/'result.json').read_text())
        if json.loads((folder/'source_manifest.json').read_text())!=expected:raise ValueError('Profile source mismatch')
        if not result['passed'] or result['check']['peak_allocated_gib']>72:raise ValueError('R1 resource qualification failed: '+name)
        if '--save' in extra:
            from qwen_vl.train.recovery import validate_complete
            saved=validate_complete(folder/'train'/f'checkpoint-{updates}')
            if saved['world_size']!=4 or saved['step']!=updates:
                raise ValueError('R1 full checkpoint topology/update mismatch: '+name)
    task('full-suite-initial',['-m','pytest','tests/vln','-q'])
    task('baseline-load',['-m','scripts.vln.check_text_export','--checkpoint',cfg['research']['baseline']['artifact'],
        '--model-path',os.environ['VLN_MODEL_PATH'],'--manifest',str(manifest),'--out',str(root/'baseline-load.json')])
    task('baseline-serving',['-m','scripts.vln.check_vggt_export','--baseline-serving',
        '--source',cfg['research']['baseline']['artifact'],'--output',str(root/'baseline-serving'),
        '--manifest',str(manifest),'--model-path',os.environ['VLN_MODEL_PATH']])
    profile('mixed',fixtures/'mixed_tail.jsonl')
    profile('mixed-empty',base/'g3-omega-20261013/mixed_empty.jsonl')
    for name in ('rxrce_longest','r2rce_longest','threshold','mixed_extreme'):
        profile(name,fixtures/(name+'.jsonl'),extra=['--save'] if name.endswith('_longest') else [])
    profile('recovery-reference',fixtures/'mixed_tail.jsonl',extra=['--save'])
    ref=root/'experiments/recovery-reference/train';res=root/'experiments/recovery-resumed/train'
    profile('recovery-resumed',fixtures/'mixed_tail.jsonl',extra=['--save','--resume',str(ref/'checkpoint-2'),'--first-update','3'])
    task('recovery-compare',['-m','scripts.vln.check_vggt_recovery','--reference',str(ref),'--resumed',str(res),'--out',str(root/'recovery.json')])
    task('export',['-m','scripts.vln.check_vggt_export','--source',str(res/'final'),'--output',str(root/'serving-export'),
        '--manifest',str(manifest),'--model-path',os.environ['VLN_MODEL_PATH']])
    for name in ('r2rce','rxrce'):profile(name+'-timing',fixtures/(name+'_representative.jsonl'),updates=8)
    profile('joint-timing',manifest,updates=12)
    task('distributed-pair-normalization',['-m','torch.distributed.run','--standalone','--nproc_per_node=4','-m','scripts.vln.check_vggt_g3_reduction'])
    task('full-suite-final',['-m','pytest','tests/vln','-q'])
    write_seal(source,root/'qualification_pass.json',records,expected)
    print('R1_QUALIFICATION_PASS',flush=True)


if __name__=='__main__':main()
