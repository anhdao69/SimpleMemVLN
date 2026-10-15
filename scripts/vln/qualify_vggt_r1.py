"""Real complete-episode R1 qualification under the selected production profile."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
import yaml
from scripts.vln.queue_vggt_r1 import source_files,write_seal,check_baseline


# An audit-only repair does not invalidate expensive unchanged training runs.
# Never carry serving/full-suite results: those must execute the revised audit.
REUSABLE_TRAINING = {'mixed', 'mixed-empty', 'rxrce_longest', 'r2rce_longest',
                    'threshold', 'mixed_extreme', 'recovery-reference',
                    'recovery-resumed', 'recovery-compare'}
AUDIT_ONLY_CHANGES = {'scripts/vln/check_vggt_export.py', 'scripts/vln/qualify_vggt_r1.py'}


def reusable_training_records(root, expected):
    root = Path(root).resolve()
    manifest = root/'source_start.json'
    old = json.loads(manifest.read_text())
    changed = sorted(k for k in set(old) | set(expected) if old.get(k) != expected.get(k))
    if not old or not expected or set(changed) - AUDIT_ONLY_CHANGES:
        raise ValueError('Cannot carry evidence across training runtime changes')
    records = json.loads((root/'qualification.json').read_text())
    names = [r['name'] for r in records]
    if len(names) != len(set(names)):
        raise ValueError('Prior qualification contains duplicate records')
    reusable = {}
    for record in records:
        name = record['name']
        if name not in REUSABLE_TRAINING or record.get('returncode') != 0:
            continue
        log = root/(name+'.log')
        reusable[name] = dict(record, carried_evidence=dict(
            root=str(root), source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            changed_audit_files=changed, log_sha256=hashlib.sha256(log.read_bytes()).hexdigest()))
    # Resume/compare evidence belongs to one uninterrupted/restarted pair.
    if 'recovery-reference' not in reusable:
        reusable.pop('recovery-resumed', None)
    if not {'recovery-reference', 'recovery-resumed'} <= reusable.keys():
        reusable.pop('recovery-compare', None)
    return reusable


def check_carried_command(name, previous, requested):
    option = '--out' if name == 'recovery-compare' else '--root'
    def normalized(command):
        result = list(command)
        if result.count(option) != 1:
            raise ValueError('Invalid carried command output option')
        result[result.index(option)+1] = '<relocated-output>'
        return result
    if normalized(previous) != normalized(requested):
        raise ValueError('Carried training command changed: '+name)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--reuse-training-from',help='Carry passed training gates only after audit-only source equivalence check')
    a=p.parse_args()
    root=Path(a.root);root.mkdir(parents=True,exist_ok=True);source=Path.cwd();records=[]
    expected=source_files(source)
    prior=Path(a.reuse_training_from).resolve() if a.reuse_training_from else None
    if prior and root.resolve()==prior:raise ValueError('Use a new qualification root; preserve prior evidence')
    reusable=reusable_training_records(prior,expected) if prior else {}
    prior_source=json.loads((prior/'source_start.json').read_text()) if prior else None
    with (root/'source_start.json').open('x') as handle:json.dump(expected,handle,indent=2)
    cfg=yaml.safe_load(Path('configs/vggt/r1_omega_h4.yaml').read_text());check_baseline(cfg)
    base=Path('/mnt/data/vmo-ai-task/anhdh35');fixtures=base/'g1-omega-20261013/fixtures'
    manifest=base/'SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl'
    check_baseline(cfg,manifest)
    def task(name,args):
        if source_files(source)!=expected:raise ValueError('Source changed during qualification')
        if name in reusable:
            record=reusable[name]
            check_carried_command(name,record['command'],args)
            if name=='recovery-compare':
                result=json.loads((prior/'recovery.json').read_text())
                if result.get('passed') is not True:raise ValueError('Prior recovery comparison failed')
            records.append(record)
            (root/'qualification.json').write_text(json.dumps(records,indent=2))
            print('CARRIED_UNCHANGED_TRAINING',name,record['carried_evidence'],flush=True)
            return
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
        folder=(prior if name in reusable else root)/'experiments'/name;result=json.loads((folder/'result.json').read_text())
        if json.loads((folder/'source_manifest.json').read_text())!=(prior_source if name in reusable else expected):raise ValueError('Profile source mismatch')
        if result.get('fixture_sha256')!=hashlib.sha256(Path(fixture).read_bytes()).hexdigest():raise ValueError('Profile fixture changed: '+name)
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
    ref=(prior if 'recovery-reference' in reusable else root)/'experiments/recovery-reference/train'
    res=(prior if 'recovery-resumed' in reusable else root)/'experiments/recovery-resumed/train'
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
