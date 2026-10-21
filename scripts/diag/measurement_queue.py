"""Wait for all five panel gates, then run full-panel captures sequentially."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from scripts.diag.gate_queue import read_status,write_json,sha256


def prerequisite(status):
    if not status:return False
    if status.get('status')=='failed':raise RuntimeError('Required gate queue failed; measurements not launched')
    if status.get('status')=='complete':
        if (len(status.get('models_completed',[]))!=5 or
            set(status.get('models_completed',[]))!={'M5_E1','G3_M5','TRIPLE','G3','G1'}):
            raise ValueError('Gate queue completion lacks all five models')
        return True
    pid=status.get('pid')
    if pid and not Path(f'/proc/{pid}').exists():raise RuntimeError('Required gate supervisor ended before completion')
    return False



def validate_all_gates(models,results,panel_uids,manifest_hash):
    from scripts.diag.measure_panel import validate_gate_evidence
    required={'M5_E1','G3_M5','TRIPLE','G3','G1'}
    if len(models)!=5 or {m['model'] for m in models}!=required or set(results)!=required:
        raise ValueError('Exactly all five model gate reports required before measurements')
    for model in models:
        name=model['model']
        validate_gate_evidence(results[name],name,model['weights_sha256'],panel_uids,manifest_hash)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','assets','output','model-path'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();source=args.source.resolve();assets=args.assets.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    status=dict(status='waiting_for_all_panel_gates',pid=os.getpid(),models_completed=[])
    def save():status['updated_unix']=time.time();write_json(args.output/'queue_status.json',status)
    save()
    try:
        gates=assets/'instrument_gates/panel_allhooks'
        while not prerequisite(read_status(gates/'queue_status.json')):save();time.sleep(15)
        projection=assets/'projection_seed0_4096'
        if not (projection/'manifest.json').exists():raise RuntimeError('Fixed projection generation incomplete')
        models=json.loads((source/'scripts/diag/gate_models.json').read_text())
        manifest=assets/'expert_panel_render/episodes.jsonl'
        episodes=[json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        results={model['model']:read_status(gates/model['model']/'result.json') for model in models}
        validate_all_gates(models,results,{e['episode_uid'] for e in episodes},sha256(manifest))
        status['gate_result_hashes']={model['model']:sha256(gates/model['model']/'result.json') for model in models}
        status['projection_manifest_sha256']=sha256(projection/'manifest.json');save()
        commands=[]
        for model in models:
            name=model['model'];replay='G1_retry_safe_path' if name=='G1' else name
            cmd=[sys.executable,'-u','-m','scripts.diag.measure_panel','--model',name,
                '--checkpoint',model['checkpoint'],'--model-path',str(args.model_path),
                '--manifest',str(assets/'expert_panel_render/episodes.jsonl'),
                '--panel',str(source/'reports/diagnostics/g3m5_failure_20261019/panel_ids.json'),
                '--teacher-cache',str(assets/'teacher_cache'),'--g2-result',str(assets/'g2_replay'/replay/'result.json'),
                '--instrument-result',str(gates/name/'result.json'),
                '--projection-cache',str(projection),'--output',str(args.output/name)]
            if name in ('G3_M5','TRIPLE'):
                recipe='g3_m5_omega_h4.yaml' if name=='G3_M5' else 'g3_m5_l8_l16_l24_omega_h4.yaml'
                cmd+=['--research-config',str(source/'configs/vggt'/recipe)]
            commands.append(dict(model=name,command=cmd))
        write_json(args.output/'commands.json',commands)
        env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(source/'src'),str(source))),OMP_NUM_THREADS='4',
            OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
        for item in commands:
            name=item['model'];status.update(status='running',current_model=name);save()
            write_json(args.output/f'{name}_code_hashes.json',
                {str(path.relative_to(source)):sha256(path) for path in sorted((source/'scripts/diag').glob('*.py'))})
            print('START',name,flush=True)
            with (args.output/f'{name}.log').open('x') as log:
                child=subprocess.Popen(item['command'],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
                status['child_pid']=child.pid;save();code=child.wait()
            result=read_status(args.output/name/'COMPLETE.json')
            if code or not result or result.get('episodes')!=240:
                raise RuntimeError(f'{name} measurements failed or incomplete; no later model launched')
            status['models_completed'].append(name);save();print('COMPLETE_MODEL',name,flush=True)
        status.update(status='complete',finished_unix=time.time());save()
    except BaseException as exc:
        status.update(status='failed',error=str(exc),traceback=traceback.format_exc());save();raise


if __name__=='__main__':main()
