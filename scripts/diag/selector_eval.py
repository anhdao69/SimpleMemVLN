"""D-e diagnostic-only launcher; imports each run's verified frozen evaluator.

Run with simulator Python -m scripts.diag.selector_eval. Worker outputs must be
new separate directories. --condition random selects frozen600, score selects
frozen60 determinism episodes; --smoke explicitly limits either to two episodes.
No qualified evaluation directory or stored navigation configuration is modified.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys

FLAGS={'diagnostic':True,'qualified':False}


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def verify_source(source,contract):
    root=Path(source).resolve()/'src/qwen_vl'
    for name,expected in contract['source_sha256'].items():
        p=root/name
        if not p.is_file() or sha256(p)!=expected:raise ValueError(f'Frozen source hash mismatch: {p}')
    return root


def import_frozen(source,contract):
    root=verify_source(source,contract)
    for name,module in list(sys.modules.items()):
        if name=='qwen_vl' or name.startswith('qwen_vl.'):
            path=getattr(module,'__file__',None)
            if path and not Path(path).resolve().is_relative_to(root):
                raise ValueError('Different qwen_vl source already imported; launch an isolated process')
    sys.path.insert(0,str(root.parent))
    module=importlib.import_module('qwen_vl.eval.habitat_r2r')
    if Path(module.__file__).resolve()!=root/'eval/habitat_r2r.py':raise ValueError('Frozen evaluator import mismatch')
    return module


def panel_ids(panel,condition):
    ids=[str(e['episode_id']) for e in panel['episodes']]
    repeat=list(map(str,panel['metadata']['determinism_episode_ids']))
    if len(ids)!=600 or len(set(ids))!=600 or len(repeat)!=60 or len(set(repeat))!=60 or not set(repeat)<=set(ids):
        raise ValueError('Expected frozen600 with unique60 determinism subset')
    if condition not in ('random','score'):raise ValueError('Unknown condition')
    return ids if condition=='random' else repeat


def baseline_records(path,wanted):
    rows={};seen=set()
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():continue
            row=json.loads(line);eid=str(row['episode_id'])
            if eid in seen:raise ValueError('Duplicate baseline episode')
            seen.add(eid)
            if eid in wanted:
                rows[eid]={k:row[k] for k in ('episode_id','episode_uid','scene_id','instruction','metrics','failed')}
    if set(rows)!=set(wanted):raise ValueError('Baseline coverage missing requested episodes')
    return rows


def validate_baseline(episodes,baseline):
    ids=[str(e.episode_id) for e in episodes]
    if len(ids)!=len(set(ids)) or set(ids)!=set(baseline):raise ValueError('Dataset/baseline IDs differ')
    for ep in episodes:
        row=baseline[str(ep.episode_id)];uid=f'val_unseen:{ep.scene_id}:{ep.episode_id}'
        if row['episode_uid']!=uid or row['scene_id']!=ep.scene_id or row['instruction']!=ep.instruction.instruction_text.strip():
            raise ValueError(f'Original baseline UID/scene/instruction differs: {ep.episode_id}')


def process_class(frozen):
    class DiagnosticProcess(frozen.ModelProcess):
        def __init__(self,args):
            root=Path(__file__).resolve().parents[2]
            command=[args.model_python,'-P','-u','-m','scripts.diag.selector_server',
                     '--checkpoint',str(args.checkpoint),'--model-path',str(args.model_path),
                     '--source',str(args.source),'--baseline-contract',str(args.baseline_contract),
                     '--condition',args.condition]
            env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(args.source.resolve()/'src'),str(root))),PYTHONDONTWRITEBYTECODE='1')
            self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1,env=env)
            self.selector=selectors.DefaultSelector();self.selector.register(self.process.stdout,selectors.EVENT_READ);self.timeout=600
            try:
                self.runtime=self.read()
                if self.runtime.get('status')!='ready' or not self.runtime.get('diagnostic') or self.runtime.get('qualified') is not False:
                    raise frozen.ModelProcessError('Diagnostic model failed to initialize')
            except BaseException:self.close();raise
    return DiagnosticProcess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','baseline-contract','baseline-journal','checkpoint','model-path','dataset','habitat-config','panel','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--condition',choices=['random','score'],required=True)
    p.add_argument('--model-python',default='/storage/anhdh35/SimpleMemVLN/.venv/bin/python')
    p.add_argument('--scenes-dir',type=Path)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--shards',type=int,default=1);p.add_argument('--shard',type=int,default=0)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--preflight-only',action='store_true',help='Verify inputs and baseline identities without loading a model or simulator')
    args=p.parse_args()
    if not 0<=args.shard<args.shards or (args.smoke and args.shards!=1):raise ValueError('Invalid shard or smoke partition')
    contract=json.loads(args.baseline_contract.read_text());frozen=import_frozen(args.source,contract)
    for path,key in ((args.dataset,'dataset_sha256'),(args.habitat_config,'habitat_config_sha256'),(args.checkpoint/'navigation.json','navigation_sha256')):
        if sha256(path)!=contract[key]:raise ValueError(f'Baseline input hash mismatch: {path}')
    weight=args.checkpoint/('model.safetensors' if (args.checkpoint/'model.safetensors').exists() else 'pytorch_model.bin')
    if sha256(weight)!=contract['weights_sha256']:raise ValueError('Checkpoint weights mismatch')
    if frozen.base_asset_hashes(args.model_path)!=contract['base_assets_sha256']:raise ValueError('Base/tokenizer asset hashes differ')
    if contract['seed']!=42 or contract['max_steps']!=500 or contract['split']!='val_unseen':raise ValueError('Unexpected original evaluation protocol')
    import habitat
    import habitat_sim
    from habitat.config.default import get_config
    from habitat.datasets import make_dataset
    if habitat.__version__!='0.2.4' or habitat_sim.__version__!='0.2.4':raise ValueError('Habitat0.2.4 required')
    config=get_config(str(args.habitat_config.resolve()))
    scenes=(args.scenes_dir or Path(contract['scenes_dir'])).resolve()
    if str(scenes)!=contract['scenes_dir']:raise ValueError('Preserve original absolute scene path for selector UID')
    with habitat.config.read_write(config):
        config.habitat.dataset.split='val_unseen';config.habitat.dataset.scenes_dir=str(scenes)
        config.habitat.dataset.data_path=str(args.dataset.resolve());config.habitat.seed=42
        config.habitat.simulator.habitat_sim_v0.gpu_device_id=args.gpu
        config.habitat.environment.iterator_options.shuffle=False
    sim=config.habitat.simulator;cam=sim.agents.main_agent.sim_sensors.rgb_sensor
    if (cam.width,cam.height,cam.hfov,sim.forward_step_size,sim.turn_angle)!=(640,480,79,.25,15):raise ValueError('Camera/action contract differs')
    if config.habitat.environment.max_episode_steps!=500 or config.habitat.task.measurements.success.success_distance!=3. or config.habitat.dataset.type!='R2RVLN-v1':raise ValueError('Task protocol differs')
    manifest=json.loads((args.panel.parent/'PANEL_MANIFEST.json').read_text())
    if sha256(args.panel)!=manifest['panel_file_sha256']['de_panel_ids.json']:
        raise ValueError('Frozen D-e panel hash mismatch')
    panel=json.loads(args.panel.read_text());all_ids=panel_ids(panel,'random');wanted=panel_ids(panel,args.condition)
    dataset=make_dataset(config.habitat.dataset.type,config=config.habitat.dataset)
    all_episodes=frozen.select_episodes(dataset.episodes,all_ids)
    baseline=baseline_records(args.baseline_journal,set(all_ids));validate_baseline(all_episodes,baseline)
    selected=frozen.select_episodes(all_episodes,wanted)
    if args.smoke:selected=selected[:2]
    selected=selected[args.shard::args.shards]
    if not selected:raise ValueError('Empty worker partition')
    dataset.episodes=selected
    args.output.mkdir(parents=True,exist_ok=False)
    evidence=dict(FLAGS,condition=args.condition,smoke=args.smoke,shard=args.shard,shards=args.shards,
                  episode_ids=[str(e.episode_id) for e in selected],seed=42,max_steps=500,scenes_dir=str(scenes),
                  baseline_identity_verified_for=600,original_contract=contract,
                  input_sha256={name:sha256(getattr(args,name)) for name in ('baseline_contract','baseline_journal','panel','dataset','habitat_config')},
                  launcher_sha256=sha256(__file__),server_sha256=sha256(Path(__file__).with_name('selector_server.py')))
    frozen.atomic_json(args.output/'evaluation_contract.json',evidence)
    if args.preflight_only:
        frozen.atomic_json(args.output/'preflight.json',dict(FLAGS,passed=True,baseline_episodes_verified=600,
            model_loaded=False,simulator_created=False,condition=args.condition))
        print(json.dumps(dict(preflight='passed',baseline_episodes_verified=600)),flush=True)
        return
    model=process_class(frozen)(args);records=[]
    try:
        frozen.atomic_json(args.output/'runtime.json',model.runtime)
        with habitat.Env(config=config,dataset=dataset) as env:
            for ep in selected:
                row=frozen.rollout_episode(env,model,ep,'val_unseen',max_steps=500)
                row.update(FLAGS,condition=args.condition,smoke=args.smoke);records.append(row)
                with (args.output/'episodes.jsonl').open('a') as stream:
                    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
                frozen.atomic_json(args.output/'summary.json',dict(frozen.summarize(records),**FLAGS,
                    condition=args.condition,smoke=args.smoke,scheduled_episodes=len(selected),complete=len(records)==len(selected)))
                print(json.dumps({'episode_id':str(ep.episode_id),'completed':len(records),'total':len(selected),'failed':row['failed']}),flush=True)
                if row.get('fatal'):raise RuntimeError('Fatal rollout; stop diagnostic worker')
    finally:model.close()


if __name__=='__main__':main()
