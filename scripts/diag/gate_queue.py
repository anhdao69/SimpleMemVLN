"""Persistent sequential five-model gates after completed preparation and teacher.

Only standard-library imports while waiting. Episode IDs are frozen and saved
before any measurement. A failed prerequisite/model stops the queue, preserving
all partial artifacts. No automatic retries or altered gate thresholds.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8<<20),b''):digest.update(part)
    return digest.hexdigest()


def write_json(path,data):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def read_status(path):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def select_ids(panel):
    ids=[str(x['episode_id']) for x in panel['episodes']]
    if len(ids)<2 or len(set(ids))!=len(ids):raise ValueError('Need distinct panel episode IDs')
    return sorted(ids)[:2]


def prerequisites_ready(teacher,preparation,complete):
    for name,record in [('teacher',teacher),('preparation',preparation)]:
        if record and record.get('status') in ('failed','error','blocked_preparation_failed'):
            raise RuntimeError(f'{name} failed; gate queue stopped')
    if not teacher or teacher.get('status')!='complete':return False
    if teacher.get('returncode')!=0:raise RuntimeError('Teacher did not exit successfully')
    if not preparation or preparation.get('status')!='complete' or not complete:return False
    if preparation.get('episode_count')!=240 or complete.get('episodes')!=240:
        raise ValueError('Completed preparation/teacher must cover all 240 panel episodes')
    return True


def selected_uids(panel,manifest,ids):
    frozen={str(x['episode_id']):x for x in panel['episodes']}
    actual={str(x['episode_id']):x for x in manifest}
    if len(actual)!=len(manifest):raise ValueError('Duplicate manifest IDs')
    selected=[]
    for eid in ids:
        if eid not in actual or eid not in frozen:raise ValueError('Selected episode absent')
        a,b=actual[eid],frozen[eid]
        if (str(a['instruction_id'])!=str(b['instruction_id']) or a['scene_id']!=b['scene_id']
                or len(a['steps'])!=b['gt_action_count']):
            raise ValueError(f'Selected episode identity/length differs: {eid}')
        selected.append(a['episode_uid'])
    if len(set(selected))!=len(selected):raise ValueError('Duplicate selected UIDs')
    return selected


def run(args):
    source=Path(args.source).resolve();assets=Path(args.assets).resolve()
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    panel=json.loads(Path(args.panel).read_text());panel_hash=sha256(args.panel)
    ids=select_ids(panel)
    write_json(output/'selection.json',dict(episode_ids=ids,panel=str(Path(args.panel).resolve()),
        panel_sha256=panel_hash,rule='first two distinct episode IDs in lexicographic order',selected_unix=time.time()))
    status=dict(status='waiting_for_teacher',pid=os.getpid(),episode_ids=ids,models_completed=[],queued_unix=time.time())
    def save():status['updated_unix']=time.time();write_json(output/'queue_status.json',status)
    save()
    try:
        teacher_file=assets/'teacher_cache_process.json'
        preparation_file=assets/'expert_panel_render/PREPARATION.json'
        complete_file=assets/'teacher_cache/COMPLETE.json'
        while not prerequisites_ready(read_status(teacher_file),read_status(preparation_file),read_status(complete_file)):
            save();time.sleep(15)
        # The teacher supervisor marks complete only after its child exits.
        # Record all authoritative prerequisites and verify their file contracts.
        if sha256(args.panel)!=panel_hash:raise RuntimeError('Frozen panel changed while queued')
        preparation=read_status(preparation_file);complete=read_status(complete_file)
        manifest_path=Path(args.manifest).resolve()
        if sha256(manifest_path)!=preparation['manifest_sha256']:raise RuntimeError('Prepared manifest hash mismatch')
        if sha256(assets/'teacher_cache/index.jsonl')!=complete['index_sha256']:
            raise RuntimeError('Completed teacher index hash mismatch')
        manifest=[json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
        uids=selected_uids(panel,manifest,ids)
        write_json(output/'prerequisites.json',dict(teacher=read_status(teacher_file),
            preparation=preparation,complete=complete,manifest=str(manifest_path),manifest_sha256=sha256(manifest_path)))
        models=json.loads(Path(args.models).read_text())
        if {x['model'] for x in models}!={'M5_E1','G3_M5','TRIPLE','G3','G1'} or len(models)!=5:
            raise ValueError('Expected exactly the five preregistered models')
        env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(source/'src'),str(source)]),
            PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',
            TOKENIZERS_PARALLELISM='false',CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        commands=[]
        for model in models:
            cmd=[args.python,'-u','-m','scripts.diag.instrument_gate','--model',model['model'],
                 '--checkpoint',model['checkpoint'],'--weights-sha256',model['weights_sha256'],
                 '--source',str(source),'--model-path',args.model_path,'--manifest',str(manifest_path),
                 '--input-kind','panel','--output',str(output/model['model'])]
            for uid in uids:cmd.extend(['--episode-uid',uid])
            commands.append(dict(model=model['model'],command=cmd,cwd=str(source)))
        write_json(output/'commands.json',commands)
        status.update(status='running',selected_episode_uids=uids);save()
        for item in commands:
            name=item['model'];status.update(current_model=name,model_started_unix=time.time());save()
            files=['scripts/diag/instrument_gate.py','scripts/diag/state.py','scripts/diag/attention.py',
                   'src/qwen_vl/train/vln_runtime.py','src/qwen_vl/stream/window_attention.py',
                   'src/qwen_vl/models/committed_write.py']
            write_json(output/f'{name}_source_hashes.json',{p:sha256(source/p) for p in files})
            print('START',name,flush=True)
            with (output/f'{name}.log').open('w') as stream:
                child=subprocess.Popen(item['command'],cwd=source,env=env,stdout=stream,stderr=subprocess.STDOUT)
                status['child_pid']=child.pid;save();code=child.wait()
            status['last_exit_code']=code;save()
            result=read_status(output/name/'result.json')
            if (code!=0 or not result or result.get('status')!='PASS'
                    or result.get('panel_episodes_completed')!=2 or result.get('selected_episode_ids')!=ids):
                raise RuntimeError(f'{name} required gates failed or incomplete; no later models launched')
            status['models_completed'].append(name);save()
            print('PASS',name,flush=True)
        status.update(status='complete',finished_unix=time.time());save();print('COMPLETE',flush=True)
    except BaseException as exc:
        status.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());save()
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','assets','panel','manifest','output','model-path'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--models',default=str(Path(__file__).with_name('gate_models.json')))
    run(parser.parse_args())

if __name__=='__main__':main()
