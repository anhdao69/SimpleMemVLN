"""Worker-2-only R1 qualification -> one joint epoch; durable and fail-closed."""
import argparse,fcntl,hashlib,json,time
from pathlib import Path
from scripts.vln.queue_geoaddress import source_files,run_allocation_step,predecessor_status

REQUIRED_GATES=('full-suite-initial','baseline-load','mixed','mixed-empty','rxrce_longest',
    'r2rce_longest','threshold','mixed_extreme','recovery-reference','recovery-resumed',
    'recovery-compare','export','r2rce-timing','rxrce-timing','joint-timing',
    'distributed-pair-normalization','full-suite-final')


def validate_records(records):
    names=[r.get('name') for r in records]
    if len(names)!=len(set(names)) or set(names)!=set(REQUIRED_GATES) or any(r.get('returncode')!=0 for r in records):
        raise ValueError('R1 qualification gates are incomplete, duplicate or failed')


def write_seal(source,path,records,expected):
    validate_records(records)
    if not expected or source_files(source)!=expected:raise ValueError('R1 qualification source changed')
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(passed=True,qualified_arm='R1',records=records,source_files=expected),indent=2))
    temp.replace(path)


def check_qualification(source,path):
    data=json.loads(Path(path).read_text())
    if data.get('passed') is not True or data.get('qualified_arm')!='R1' or not data.get('source_files') or data['source_files']!=source_files(source):
        raise ValueError('Missing, failed or source-mismatched R1 qualification')
    validate_records(data.get('records',[]));return data


def check_baseline(config):
    record=config['research']['baseline'];root=Path(record['artifact'])
    for name,key in (('navigation.json','navigation_sha256'),('pytorch_model.bin','weights_sha256')):
        digest=hashlib.sha256()
        with (root/name).open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
        if digest.hexdigest()!=record[key]:raise ValueError('R0 baseline artifact changed: '+name)
    if json.loads((root/'navigation.json').read_text())!=record['metadata']:
        raise ValueError('R0 baseline metadata differs from registry')
    manifest=Path(record['manifest'])
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=record['manifest_sha256']:
        raise ValueError('R1 joint R2R + RxR15 manifest changed')


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--root')
    p.add_argument('--allocation');p.add_argument('--check-only',action='store_true');p.add_argument('--seal')
    a=p.parse_args();source=Path(a.source).resolve()
    import yaml
    config=yaml.safe_load((source/'configs/vggt/r1_omega_h4.yaml').read_text())
    if a.check_only:
        check_qualification(source,a.seal);check_baseline(config);print('R1_LAUNCH_CHECK_PASS');return
    if not a.root or not a.allocation:p.error('--root and --allocation required')
    root=Path(a.root).resolve();root.mkdir(parents=True,exist_ok=True)
    def record(status,**fields):
        data=dict(status=status,updated_unix=time.time(),allocation=a.allocation,node='worker-2',source=str(source),**fields)
        path=root/'queue_state.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)
        print(json.dumps(data),flush=True)
    def step(name,command,completed=None):
        run_allocation_step(a.allocation,source,name,command,root,record,completed_run=completed)
    with (root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            check_baseline(config)
            seal=root/'qualification/qualification_pass.json'
            if not seal.exists():step('r1-qualification',['python','-m','scripts.vln.qualify_vggt_r1','--root',str(root/'qualification')])
            check_qualification(source,seal)
            run=root/'r1_joint_e1'
            if predecessor_status(run,3852)!='complete':
                step('r1-joint-e1',['bash',str(source/'train/launch_r1_omega_joint_e1.sh'),str(run),str(seal)],run)
            if predecessor_status(run,3852)!='complete':raise RuntimeError('R1 exited without completed epoch')
            record('complete',stage='R1')
        except BaseException as error:
            record('blocked',error=repr(error));raise


if __name__=='__main__':main()
