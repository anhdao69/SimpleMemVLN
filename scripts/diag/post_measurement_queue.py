"""Supervise remaining ablations, probes and closed-loop experiments.

Requires all full-panel captures first. Ridge fits run on CPU alongside one
ablation model on GPU. Closed-loop workers run only after both finish and every
two-episode launcher smoke passes. Any child failure stops this queue's other
children; unrelated evaluation processes are never touched.
"""
import argparse
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
import traceback
from scripts.diag.gate_queue import read_status,write_json,sha256
from scripts.diag.measurement_queue import prerequisite



MODEL_ORDER=('M5_E1','G3_M5','TRIPLE','G3','G1')


def ordered_models(models):
    lookup={row['model']:row for row in models}
    if len(models)!=5 or len(lookup)!=5 or set(lookup)!=set(MODEL_ORDER):
        raise ValueError('Exactly the five distinct registered models required')
    return [lookup[name] for name in MODEL_ORDER]


def validate_completion(record,**expected):
    if not record or any(record.get(key)!=value for key,value in expected.items()):
        raise ValueError('Missing or inconsistent stage completion evidence')


def owned_groups(children):
    # All children were launched start_new_session=True. An exited group leader
    # can leave a policy grandchild alive, so do not filter by leader.poll().
    return [child.pid for child in children.values()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','assets','output','model-path'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--simulator-python',default='/storage/anhdh35/miniconda3/envs/spatialstack_dagger/bin/python')
    args=parser.parse_args();source=args.source.resolve();assets=args.assets.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir()
    panel=source/'reports/diagnostics/g3m5_failure_20261019/panel_ids.json'
    de_panel=panel.with_name('de_panel_ids.json');models=ordered_models(json.loads((source/'scripts/diag/gate_models.json').read_text()))
    status=dict(status='waiting_for_panel_measurements',pid=os.getpid(),children={},stages_completed=[])
    commands=[];children={}
    def save():status['updated_unix']=time.time();write_json(out/'queue_status.json',status)
    env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(source/'src'),str(source))),OMP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',
        HABITAT_SIM_LOG='quiet',MAGNUM_LOG='quiet')
    def launch(label,command,environment=None):
        commands.append(dict(label=label,command=command,cwd=str(source)))
        write_json(out/'commands.json',commands)
        log=(out/'logs'/f'{label}.log').open('x')
        try:child=subprocess.Popen(command,cwd=source,env=environment or env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        finally:log.close()
        children[label]=child;status['children'][label]=dict(pid=child.pid,status='running');save()
        print('START',label,flush=True);return label
    def wait(labels,watch=()):
        while True:
            for label in [*labels,*watch]:
                code=children[label].poll()
                if code is not None:
                    status['children'][label].update(status='complete' if code==0 else 'failed',exit_code=code)
                    if code:raise RuntimeError(f'{label} exited {code}')
            save()
            if all(children[label].poll() is not None for label in labels):return
            time.sleep(10)
    def command(module,*arguments):return [sys.executable,'-u','-m','scripts.diag.'+module,*map(str,arguments)]
    save()
    try:
        measurements=assets/'panel_measurements';gates=assets/'instrument_gates/panel_allhooks'
        while not prerequisite(read_status(measurements/'queue_status.json')):save();time.sleep(15)
        for model in models:
            name=model['model'];complete=read_status(measurements/name/'COMPLETE.json')
            validate_completion(complete,episodes=240,weights_sha256=model['weights_sha256'],
                                journal_sha256=sha256(measurements/name/'episodes.jsonl'))
        status.update(status='running',stage='probes_and_ablation');save()
        fit=command('fit_probes','--panel',panel,'--teacher-cache',assets/'teacher_cache','--output',out/'probes','--device','cpu')
        for model in models:fit+=['--capture',f"{model['model']}={measurements/model['model']}"]
        cpu_env=dict(env,OMP_NUM_THREADS='16',OPENBLAS_NUM_THREADS='16',MKL_NUM_THREADS='16')
        fit_label=launch('fit_probes',fit,cpu_env)
        for model in models[:3]:
            name=model['model']
            db=command('run_bank_ablation','--model',name,'--checkpoint',model['checkpoint'],
                '--model-path',args.model_path,'--manifest',assets/'expert_panel_render/episodes.jsonl',
                '--panel',panel,'--baseline-root',measurements/name,'--instrument-result',gates/name/'result.json',
                '--measurement-queue-status',measurements/'queue_status.json','--gate-queue-status',gates/'queue_status.json',
                '--projection-cache',assets/'projection_seed0_4096','--output',out/'bank_ablation'/name)
            label=launch('bank_'+name,db);wait([label],watch=[fit_label])
            complete=read_status(out/'bank_ablation'/name/'COMPLETE.json')
            if not complete or complete.get('episodes')!=240 or not complete.get('adapter_gate_passed'):
                raise ValueError('Ablation output incomplete or adapter gate missing')
        wait([fit_label])
        validate_completion(read_status(out/'probes/COMPLETE.json'),status='complete',
                            models=list(MODEL_ORDER),lag_groups=['A','B','C'])
        status['stages_completed'].append('probes_and_ablation');save()
        native=command('summarize_captures','--panel',panel,'--output',out/'capture_summary')
        for model in models:native+=['--capture',f"{model['model']}={measurements/model['model']}"]
        wait([launch('summarize_captures',native)])
        validate_completion(read_status(out/'capture_summary/COMPLETE.json'),models=list(MODEL_ORDER),
            episodes_per_model=240,summary_sha256=sha256(out/'capture_summary/summary.json'))
        apply=command('apply_ablation_probes','--probe-root',out/'probes','--teacher-cache',assets/'teacher_cache',
            '--panel',panel,'--output',out/'ablation_probe_results',
            '--ablation-root',out/'bank_ablation','--baseline-root',measurements)
        wait([launch('apply_ablation_probes',apply)])
        validate_completion(read_status(out/'ablation_probe_results/COMPLETE.json'),
            status='complete',refitted=False,models=list(MODEL_ORDER[:3]),
            summary_sha256=sha256(out/'ablation_probe_results/summary.json'))
        status['stages_completed'].append('offline_summaries');status['stage']='closed_loop';save()
        dataset='/storage/anhdh35/JanusVLN/data/datasets/r2r/val_unseen/val_unseen.json.gz'
        for model in models[:2]:
            name=model['model'];original='m5_e1_epoch1_20261019' if name=='M5_E1' else 'g3_m5_epoch1_20261017'
            frozen=assets/'remote_evaluation'/original
            base=[args.simulator_python,'-u','-m','scripts.diag.selector_eval',
                '--source',str(frozen/'source'),'--baseline-contract',str(frozen/'full/evaluation_contract.json'),
                '--baseline-journal',str(frozen/'full/episodes.jsonl'),'--checkpoint',model['checkpoint'],
                '--model-path',str(args.model_path),'--dataset',dataset,
                '--habitat-config',str(frozen/'source/configs/eval/vln_r2r.yaml'),'--panel',str(de_panel)]
            for condition in ('random','score'):
                smoke=out/'closed_loop'/name/f'smoke_{condition}'
                wait([launch(f'{name}_{condition}_smoke',base+['--condition',condition,'--smoke','--output',str(smoke)])])
                result=read_status(smoke/'summary.json')
                if not result or not result.get('complete') or result.get('episodes')!=2 or result.get('failures')!=0:
                    raise ValueError('Two-episode launcher smoke failed')
                workers=[]
                for shard in range(4):
                    destination=out/'closed_loop'/name/condition/f'shard-{shard}'
                    workers.append(launch(f'{name}_{condition}_{shard}',base+['--condition',condition,
                        '--shards','4','--shard',str(shard),'--output',str(destination)]))
                wait(workers)
        de=command('selector_results','--panel',de_panel,'--output',out/'selector_results')
        for name,original in [('M5_E1','m5_e1_epoch1_20261019'),('G3_M5','g3_m5_epoch1_20261017')]:
            de+=['--baseline',f'{name}={assets/"remote_evaluation"/original/"full/episodes.jsonl"}',
                 '--runs',f'{name}={out/"closed_loop"/name}']
        wait([launch('selector_results',de)])
        status['stages_completed'].append('closed_loop');status.update(status='complete',finished_unix=time.time());save()
    except BaseException as exc:
        groups=owned_groups(children)
        for group in groups:
            try:os.killpg(group,signal.SIGTERM)
            except ProcessLookupError:pass
        for child in children.values():
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait()
        # Catch policy descendants even when their direct supervisor already exited.
        for group in groups:
            try:os.killpg(group,signal.SIGKILL)
            except ProcessLookupError:pass
        status.update(status='failed',error=str(exc),traceback=traceback.format_exc());save();raise


if __name__=='__main__':main()
