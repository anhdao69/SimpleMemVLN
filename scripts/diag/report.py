"""Build a NEW review stage from completed actual D-a..f artifacts.

Usage: python -m scripts.diag.report --post-root ROOT --panel panel_ids.json
 --prerequisites report_prerequisites.json --output NEW_STAGE

Prerequisite schema (assembled after runs, never inferred from missing evidence):
{complete:true, provenance:{branch,scientific_commit,diagnostic_commit,checkpoints,
 teacher,dataset,environment,baseline_results:{MODEL:{episodes,successes,sr,spl?}}}, gates:{MODEL:{g1,g2,g3,dc,df?,db?}},
 scope_amendments:[...], limitations:[...], source_files:[{path,sha256}],
 command_files:[".../commands.json"]}. Gate values have passed:true and retain
 their numeric evidence. All five models need g1/g2/g3/dc, two probe models df,
 three bank models db. Failed rendering is separately recorded under provenance
 and authorized scope_amendments; it is never relabeled passed. source_files
 must pin the original spec, both scope amendments, panel and all evidence used.
 Commands use queue records {label,command:[argv...],cwd}; environment is retained
 in prerequisite provenance. Original optional trained heads may be unavailable.
"""
import argparse
import csv
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import numpy as np
from scripts.diag.cache_teacher import sha256
from scripts.diag.statistics import episode_bootstrap,paired_bootstrap
from scripts.diag.summarize_captures import Collector,MODELS
from scripts.diag.report_decisions import decide,valid,DRIFT_KEYS

METRICS=('cosine','cosine_Q1','cosine_Q2','cosine_Q3','cosine_Q4','cosine_Global','top1','mrr')
CHUNK_BYTES=64*1024*1024

def read(path):return json.loads(Path(path).read_text())
def check_file(path,digest):
    if sha256(path)!=digest:raise ValueError(f'Hash mismatch: {path}')

def write_verified_chunked_csv(source,part_root,filename):
    """Split a large CSV as deterministic gzip byte chunks and verify reconstruction."""
    part_root=Path(part_root);part_root.mkdir(parents=True,exist_ok=True)
    for old in part_root.glob(filename+'.part-*.gz'):old.unlink()
    part_records=[]
    with Path(source).open('rb') as raw:
        suffix=0
        while True:
            chunk=raw.read(CHUNK_BYTES)
            if not chunk:break
            part_path=part_root/f'{filename}.part-{suffix:03d}.gz'
            with part_path.open('wb') as target:
                with gzip.GzipFile(filename='',fileobj=target,mode='wb',compresslevel=3,mtime=0) as gz:gz.write(chunk)
            part_records.append(dict(path=part_path.name,bytes=part_path.stat().st_size,sha256=sha256(part_path)))
            suffix+=1
    with Path(source).open(newline='') as table:row_count=sum(1 for _ in csv.reader(table))-1
    reconstructed=hashlib.sha256();reconstructed_bytes=0
    for part in part_records:
        with gzip.open(part_root/part['path'],'rb') as decoded:
            while block:=decoded.read(8*1024*1024):
                reconstructed.update(block);reconstructed_bytes+=len(block)
    if reconstructed.hexdigest()!=sha256(source) or reconstructed_bytes!=Path(source).stat().st_size:
        raise ValueError(f'Chunked CSV round-trip verification failed: {source}')
    return row_count,part_records
def rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def stage_csv_exports(sources,exports,bundle_root):
    """Reuse verified split-gzip CSVs and copy only inputs absent from that bundle."""
    bundle_root=Path(bundle_root);manifest_path=bundle_root/'manifest.json'
    manifest=read(manifest_path)
    if manifest.get('status')!='verified':raise ValueError('Completed CSV bundle is not verified')
    packed={str(Path(item['source']).resolve()):item for item in manifest['sources']}
    auxiliary_path=bundle_root/'native_recall_pairs.manifest.json'
    auxiliary=read(auxiliary_path) if auxiliary_path.is_file() else None
    if auxiliary:
        if auxiliary.get('status')!='verified':raise ValueError('Native recall pair bundle is not verified')
        packed.update({str(Path(item['source']).resolve()):dict(item,_manifest_path=auxiliary_path)
                       for item in auxiliary['sources']})
    source_map={str(Path(source).resolve()):Path(source) for source in sources}
    inventory=[];used=set()
    for key,item in list(packed.items()):
        if key not in source_map:raise ValueError(f'Packaged CSV is absent from report inputs: {key}')
        source=source_map[key]
        if sha256(source)!=item['source_sha256']:raise ValueError(f'Packaged source CSV changed: {source}')
        parts=[]
        item_manifest=Path(item.get('_manifest_path',manifest_path))
        for part in item['parts']:
            path=item_manifest.parent/part['path']
            if sha256(path)!=part['sha256']:raise ValueError(f'Packaged CSV part changed: {path}')
            parts.append(dict(path=str(path.relative_to(bundle_root)),bytes=part['bytes'],sha256=part['sha256']))
        inventory.append(dict(kind='chunked_gzip_csv',path=f"completed_exports/{item['relative_source']}",
            source=str(source),source_sha256=item['source_sha256'],source_bytes=item['source_bytes'],
            rows=item['csv_records_excluding_header'],bundle_manifest=str(item_manifest.relative_to(bundle_root)),
            bundle_manifest_sha256=sha256(item_manifest),parts=parts,
            reconstruction=f"gzip -cd {item['relative_source']}.part-*.gz > {Path(item['relative_source']).name}"))
        used.add(key)
    exports=Path(exports)
    for key,source in source_map.items():
        if key in used:continue
        if source.stat().st_size>CHUNK_BYTES:
            if source.name=='native_recall_pairs.csv':
                relative='native_recall_pairs.csv';part_root=bundle_root
                row_count,part_records=write_verified_chunked_csv(source,part_root,relative)
                source_record=dict(source=str(source),relative_source=relative,source_bytes=source.stat().st_size,
                    source_sha256=sha256(source),csv_records_excluding_header=row_count,parts=part_records)
                sidecar=bundle_root/'native_recall_pairs.manifest.json'
                dedicated={'status':'verified','diagnostic_scope':'D-a native self-recall pair-level export',
                    'encoding':'CSV bytes split into <=64 MiB chunks; deterministic gzip; concatenate decompressed parts',
                    'sources':[source_record]}
                sidecar.write_text(json.dumps(dedicated,indent=2)+'\n')
                manifest_inventory=[dict(path=str((part_root/part['path']).relative_to(bundle_root)),bytes=part['bytes'],sha256=part['sha256']) for part in part_records]
                inventory.append(dict(kind='chunked_gzip_csv',path=f'completed_exports/{relative}',source=str(source),
                    source_sha256=source_record['source_sha256'],source_bytes=source_record['source_bytes'],rows=row_count,
                    bundle_manifest='completed_exports/native_recall_pairs.manifest.json',bundle_manifest_sha256=sha256(sidecar),
                    parts=manifest_inventory,reconstruction=f'gzip -cd {relative}.part-*.gz > {relative}'))
            else:
                bundle_id=hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:16]
                relative_dir=Path('additional_large')/bundle_id
                part_root=exports/relative_dir
                row_count,part_records=write_verified_chunked_csv(source,part_root,source.name)
                source_record=dict(source=str(source),relative_source=source.name,source_bytes=source.stat().st_size,
                    source_sha256=sha256(source),csv_records_excluding_header=row_count,parts=part_records)
                sidecar=part_root/'manifest.json'
                sidecar.write_text(json.dumps(dict(status='verified',encoding='deterministic gzip byte chunks',sources=[source_record]),indent=2)+'\n')
                report_relative=Path('exports')/relative_dir
                inventory.append(dict(kind='chunked_gzip_csv',path=str(report_relative/source.name),source=str(source),
                    source_sha256=source_record['source_sha256'],source_bytes=source_record['source_bytes'],rows=row_count,
                    bundle_manifest=str(report_relative/'manifest.json'),bundle_manifest_sha256=sha256(sidecar),
                    parts=[dict(path=str(report_relative/part['path']),bytes=part['bytes'],sha256=part['sha256']) for part in part_records],
                    reconstruction=f"gzip -cd {report_relative/source.name}.part-*.gz > {report_relative/source.name}"))
            used.add(key);continue
        destination=exports/'additional'/f'{len(inventory):03d}_{source.name}'
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
        inventory.append(dict(kind='copied_csv',path=str(destination.relative_to(exports.parent)),
            source=str(source),sha256=sha256(destination),bytes=destination.stat().st_size))
    return inventory

def unique(items,**filters):
    found=[r for r in items if all(r.get(k)==v for k,v in filters.items())]
    if len(found)>1:raise ValueError(f'Ambiguous report estimand: {filters}')
    return found[0] if found else None

def validate_prerequisites(path,panel):
    p=read(path)
    if p.get('complete') is not True:raise ValueError('Report prerequisites incomplete')
    for key in ('branch','scientific_commit','diagnostic_commit','checkpoints','teacher','dataset','environment'):
        if not p['provenance'].get(key):raise ValueError(f'Missing provenance {key}')
    from scripts.diag.report_tables import validate_baselines
    validate_baselines(p['provenance'].get('baseline_results'))
    if not p.get('source_files') or not p.get('command_files'):raise ValueError('Pinned sources and reproduction commands required')
    pinned={str(Path(r['path']).resolve()):r['sha256'] for r in p['source_files']}
    for file,digest in pinned.items():check_file(file,digest)
    if not {'TASK_SPEC.md','SCOPE_UPDATE.md','KL_SCOPE_UPDATE.md'}<={Path(f).name for f in pinned}:
        raise ValueError('Original spec and both scope amendments must be pinned')
    if str(panel.resolve()) not in pinned:raise ValueError('Panel must be pinned in prerequisite sources')
    for model in MODELS:
        names=['g1','g2','g3','dc']+(['df'] if model in ('G3_M5','TRIPLE') else [])+(['db'] if model in ('M5_E1','G3_M5','TRIPLE') else [])
        for name in names:
            if p['gates'].get(model,{}).get(name,{}).get('passed') is not True:raise ValueError(f'Missing/pending/failed gate {model}/{name}')
    for name in p['command_files']:
        if str(Path(name).resolve()) not in pinned:raise ValueError('Command file must be hash pinned')
        for record in read(name):
            if not isinstance(record['command'],list) or not record['command'] or not record['cwd']:raise ValueError('Invalid exact reproduction command')
    return p

def validate_artifacts(root,panel):
    """No output directory is created until all mandatory inputs are validated."""
    native=read(root/'capture_summary/summary.json');nd=read(root/'capture_summary/COMPLETE.json')
    check_file(root/'capture_summary/summary.json',nd['summary_sha256'])
    if nd['episodes_per_model']!=240 or set(nd['models'])!=set(MODELS) or native['panel_sha256']!=sha256(panel):raise ValueError('Native panel completion mismatch')
    for name,digest in native['output_csv_sha256'].items():check_file(root/'capture_summary'/name,digest)
    pp=read(root/'probes/provenance.json');pd=read(root/'probes/COMPLETE.json')
    if pd['status']!='complete' or set(pd['models'])!=set(MODELS) or pp['panel_sha256']!=sha256(panel):raise ValueError('Probe completion mismatch')
    for model in MODELS:
        if pp['captures'][model]['complete_sha256']!=native['capture_complete_sha256'][model]:raise ValueError('Native/probe baseline captures differ')
    ad=read(root/'ablation_probe_results/COMPLETE.json');ab=read(root/'ablation_probe_results/summary.json')
    check_file(root/'ablation_probe_results/summary.json',ad['summary_sha256'])
    if ad['status']!='complete' or ad['refitted'] or set(ad['models'])!={'M5_E1','G3_M5','TRIPLE'}:raise ValueError('Ablation incomplete/refitted')
    ap=read(root/'ablation_probe_results/provenance.json')
    if ap['probe_provenance_sha256']!=sha256(root/'probes/provenance.json'):raise ValueError('Ablation probe identity differs')
    de=read(root/'selector_results/results.json')
    if de.get('diagnostic') is not True or de.get('qualified') is not False or set(de['models'])!={'M5_E1','G3_M5'}:raise ValueError('Selector diagnostic identity incomplete')
    for model,record in de['models'].items():
        if record['n_episodes']!=600 or record['n_determinism']!=60 or record['determinism']['n_episodes']!=60:raise ValueError('Selector fixed denominators incomplete')
        for worker in record['workers']+record['determinism_workers']:
            check_file(Path(worker['root'])/'episodes.jsonl',worker['journal_sha256'])
            check_file(Path(worker['root'])/'evaluation_contract.json',worker['contract_sha256'])
    if len(ab['coverage'])!=720 or any(sum(r['model']==m for r in ab['coverage'])!=240 for m in ('M5_E1','G3_M5','TRIPLE')):raise ValueError('Ablation panel coverage incomplete')
    if ap['panel_sha256']!=sha256(panel):raise ValueError('Ablation panel differs')
    return native,ab,de

def aggregate_probes(root,panel):
    from scripts.diag.recall import sampled_pairs,lag_group,lag_bucket
    panel_rows={str(r['episode_id']):r for r in read(panel)['episodes']};c=Collector();identities={};files=[]
    for model in MODELS:
        for group in ('A','B','C'):
            families=[('control',24)]+[('state',l) for l in ((8,16,22,24,26) if model=='TRIPLE' else (22,24,26))]
            control=None
            expected={(eid,t,s) for eid,e in panel_rows.items() for t,s in sampled_pairs(e['gt_action_count']) if lag_group(t-s)==group}
            for feature,layer in families:
                directory=root/model/group/f'{feature}_layer_{layer}';done=read(directory/'COMPLETE.json')
                check_file(directory/'pairs.csv',done['pair_metrics_sha256']);table=rows(directory/'pairs.csv');files.append(directory/'pairs.csv')
                if done['folds']!=2 or done['pairs']!=len(table):raise ValueError('Incomplete probe family')
                folds={i:read(directory/f'fold_{i}.json') for i in range(2)}
                for fold in folds.values():
                    check_file(directory/fold['model_file'],fold['model_sha256'])
                    if set(fold['test_scenes'])&set(fold['train_scenes']):raise ValueError('Probe scene leakage')
                index={}
                for r in table:
                    eid=str(r['episode_id']);t,s=int(r['anchor']),int(r['source']);key=(eid,t,s)
                    if key in index or key not in expected:raise ValueError('Duplicate/unregistered probe pair')
                    if Path(r['scene_id']).stem!=panel_rows[eid]['scene'] or int(r['lag'])!=t-s or r['lag_group']!=group or r['lag_bucket']!=str(lag_bucket(t-s)):raise ValueError('Probe pair metadata mismatch')
                    if r['feature']!=feature or int(r['layer'])!=layer:raise ValueError('Probe feature/layer mismatch')
                    if int(r['n_candidate_frames'])!=panel_rows[eid]['gt_action_count']:raise ValueError('Retrieval did not rank all episode frames')
                    fold=folds[int(r['fold'])]
                    if r['scene_id'] not in fold['test_scenes'] or r['scene_id'] in fold['train_scenes']:raise ValueError('Probe row not held out')
                    identity=(r['episode_uid'],r['scene_id'])
                    if eid in identities and identities[eid]!=identity:raise ValueError('Cross-model probe UID/scene mismatch')
                    identities[eid]=identity;index[key]=r
                if set(index)!=expected:raise ValueError('Probe pair coverage incomplete')
                for key,r in index.items():
                    episode=dict(episode_id=r['episode_id'],scene_id=r['scene_id'])
                    for metric in METRICS:
                        for label in (group,r['lag_bucket']):
                            dims=dict(diagnostic='D-a2',model=model,feature=feature,layer=layer,lag=label,metric=metric)
                            counts={'pairs':1,'candidate_frame_visits':int(r['n_candidate_frames'])}
                            c.add(dims,episode,float(r[metric]),counts=counts)
                            if feature=='state':
                                v=control[key]
                                if r['fold']!=v['fold']:raise ValueError('State/control folds differ')
                                c.add(dict(dims,feature='memory_specific'),episode,float(r[metric])-float(v[metric]),counts=counts)
                if feature=='control':control=index
    comparisons=[]
    for left,right in [('M5_E1','G3_M5'),('M5_E1','TRIPLE'),('G1','G3')]:comparisons.extend(c.contrasts(left,right))
    return c,comparisons,files

def pooled_bank_contrast(path):
    """Equal layers inside episode, then paired episode contrast; never pool episodes."""
    by=defaultdict(lambda:defaultdict(dict))
    for row in rows(path):
        if row.get('diagnostic')=='D-c' and row.get('model') in ('M5_E1','G3_M5') and row.get('category')=='total_bank' and row.get('role')=='combined' and row.get('metric')=='mass' and row.get('scope')=='query_head_mean':
            by[row['model']][row['episode_id']][int(row['layer'])]=row
    paired=[]
    if set(by)!= {'M5_E1','G3_M5'}:return None
    if set(by['M5_E1'])!=set(by['G3_M5']):raise ValueError('R4 episode pairing differs')
    for eid in by['M5_E1']:
        values=[];scene=None
        for model in ('M5_E1','G3_M5'):
            layers=by[model][eid]
            if set(layers)!=set(range(3,32,4)):return None
            if any(not r['value'] for r in layers.values()):return None
            scenes={r['scene'] for r in layers.values()}
            if len(scenes)!=1 or (scene is not None and scene not in scenes):raise ValueError('R4 scene mismatch')
            scene=next(iter(scenes));values.append(float(np.mean([float(r['value']) for r in layers.values()])))
        paired.append(dict(episode_id=eid,scene=scene,left=values[0],right=values[1]))
    result=paired_bootstrap(paired);result['definition']='G3_M5-M5_E1; equal eight layers per episode; combined33 total bank mass';return result

def decision_evidence(native,metric2,ab,de,r4):
    contrasts=native['paired_contrasts'];e={}
    for group,key in [('B','b'),('C','c')]:
        e['r1_'+key]=unique(contrasts,diagnostic='D-a',model='G3_M5 - M5_E1',layer=24,region='Global',metric='forgetting',lag=group)
        e['r1_probe_'+key]=unique(metric2,model='G3_M5 - M5_E1',feature='state',layer=24,metric='cosine',lag=group)
    e['r2_target']=unique(ab['contrasts'],model='G3_M5',contrast='targeted-control',group='C',feature='state',metric='cosine')
    e['r2_attention']=unique(contrasts,diagnostic='D-c',model='G3_M5 - M5_E1',layer='3-23',role='combined',category='bank_lag_8_32',metric='mass_per_token')
    e['r3_margins']={str(l):unique(contrasts,diagnostic='D-d',model='G3_M5 - M5_E1',layer=l,metric='normalized_cutoff_margin',scope='admission_mean') for l in (2,6,10,14,18,22)}
    e['r3_interaction']=de['interaction'];e['r4_mass']=r4
    e['r5_overlap']={f'{m}:{l}':unique(native['summaries'],diagnostic='D-f',model=m,layer=l,metric='selected_overlap',scope='episode_mean',length_cohort='all') for m in ('G3_M5','TRIPLE') for l in range(3,32,4)}
    return e

def aggregate_bank_kl(root,panel):
    ids=[str(e['episode_id']) for e in read(panel)['episodes']];summaries=[];all_rows=[]
    for model in ('M5_E1','G3_M5','TRIPLE'):
        directory=root/model;done=read(directory/'COMPLETE.json')
        if done['episodes']!=240 or done.get('parameter_versions_unchanged') is not True or not done['adapter_gate_passed']:raise ValueError('Bank ablation incomplete')
        check_file(directory/'episodes.jsonl',done['journal_sha256'])
        journal=[json.loads(line) for line in (directory/'episodes.jsonl').read_text().splitlines() if line.strip()]
        if [str(r['episode_id']) for r in journal]!=ids:raise ValueError('Bank KL panel mismatch')
        records=[];decisions=0
        for row in journal:
            name='global_vocabulary_kl.json';path=directory/row['folder']/name;check_file(path,row['files_sha256'][name]);kl=read(path)
            if len(kl['per_decision'])!=kl['n_decisions'] or kl['n_decisions']<=0 or kl['units']!='nats':raise ValueError('KL decision denominator invalid')
            if not np.isfinite(kl['per_decision']).all() or not np.isclose(np.mean(kl['per_decision']),kl['mean'],rtol=1e-12,atol=1e-12):raise ValueError('KL mean differs from decisions')
            records.append(dict(episode_id=str(row['episode_id']),scene=row['scene_id'],value=kl['mean']))
            all_rows.append(dict(model=model,episode_id=str(row['episode_id']),scene_id=row['scene_id'],value=kl['mean'],n_decisions=kl['n_decisions']))
            decisions+=kl['n_decisions']
        summaries.append(dict(model=model,**episode_bootstrap(records),counts={'decisions':decisions},units='nats'))
    return dict(summaries=summaries,episodes=all_rows,definition='KL(original full vocabulary || bank-off) at teacher-forced response decision positions')


def format_metric(m):
    if m is not None:
        for value in [m.get('estimate'),*(m.get('ci95') or [])]:
            if value is not None and not np.isfinite(float(value)):raise ValueError('Nonfinite report metric')
    if not valid(m):return 'unavailable (undefined or missing)'
    counts=m.get('counts',{})
    extra=', '.join(f'{k}={v}' for k,v in counts.items())
    if 'n_pairs' in m:extra+=f", pairs={m['n_pairs']}"
    if m.get('n_undefined'):extra+=f", undefined episodes={m['n_undefined']}"
    return f"{m['estimate']:.6g} [{m['ci95'][0]:.6g}, {m['ci95'][1]:.6g}]; n={m['n_episodes']} episodes"+(f'; {extra}' if extra else '')

def recommended(decisions):
    supported={k for k,v in decisions.items() if v['status']=='Supported'};items=[]
    if supported&{'R1','R2'}:items.append('Test visibility-aware G3: supervise lags 4–28 for R4 and down-weight pairs whose source tokens are visible in lower banks.')
    if 'R3' in supported:items.append('Protect selection in a controlled follow-up: recall-gap selection, a frozen scorer, or blocked G3 gradients into scorer layers.')
    if supported=={'R4'}:items.append('Keep G3 with Window8 rather than combining these redundant memories.')
    if not supported:items.append('Run a second seed of M5_E1 and G3_M5 before redesign; the closed-loop regression remains plausibly sensitive to seed and rollout noise.')
    if supported=={'R5'}:items.append('Investigate the measured train/serve selection mismatch before any redesign; this task does not train a correction.')
    return ' '.join(items)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--post-root',type=Path,required=True);p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--prerequisites',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('Report stage must be new; interim report is never overwritten')
    prereq=validate_prerequisites(args.prerequisites,args.panel);native,ab,de=validate_artifacts(args.post_root,args.panel)
    import matplotlib  # dependency preflight before creating a report stage
    bank_kl=aggregate_bank_kl(args.post_root/'bank_ablation',args.panel)
    collector,comparisons,pair_files=aggregate_probes(args.post_root/'probes',args.panel)
    evidence=decision_evidence(native,comparisons,ab,de,pooled_bank_contrast(args.post_root/'capture_summary/episode_metrics.csv'))
    decisions=decide(evidence)
    required=[evidence[k] for k in ('r1_b','r1_c','r1_probe_b','r1_probe_c','r2_attention','r3_interaction','r4_mass')]+list(evidence['r3_margins'].values())+list(evidence['r5_overlap'].values())
    # Targeted estimates may legitimately be undefined when preregistered matching is unavailable.
    target_unavailable=all(row['matched_pairs']==0 for row in ab['coverage'] if row['model']=='G3_M5')
    full=all(valid(m) for m in required) and (evidence['r2_target'] is not None or target_unavailable) and prereq['complete']
    from scripts.diag.export_recall_pairs import export_recall_pairs
    pair_csv=args.post_root/'capture_summary/native_recall_pairs.csv'
    capture_roots={model:Path(path) for model,path in native['capture_roots'].items()}
    capture_base=capture_roots[MODELS[0]].parent
    if set(capture_roots)!=set(MODELS) or any(path!=capture_base/model for model,path in capture_roots.items()):
        raise ValueError('Native capture roots are not one verified model-panel tree')
    pair_export=export_recall_pairs(capture_base,args.panel,pair_csv)
    for model in MODELS:
        complete_path=capture_roots[model]/'COMPLETE.json'
        digest=sha256(complete_path)
        if digest!=native['capture_complete_sha256'][model]:
            raise ValueError(f'Native D-a pair export capture manifest mismatch for {model}')
        pair_export['models'][model]['capture_manifest_sha256']=digest
    pair_manifest=read(pair_export['manifest_path'])
    for model in MODELS:
        pair_manifest['models'][model]['capture_manifest_sha256']=pair_export['models'][model]['capture_manifest_sha256']
    Path(pair_export['manifest_path']).write_text(json.dumps(pair_manifest,indent=2,sort_keys=True)+'\n')
    pair_export['manifest_sha256']=sha256(pair_export['manifest_path'])
    result=dict(diagnostic=True,qualified=False,full_completion=full,decisions=decisions,decision_evidence=evidence,
        D_a_metric1=native,D_a_metric2=dict(summaries=collector.summaries(),paired_contrasts=comparisons),D_b=ab,D_b_decision_token_KL=bank_kl,D_e=de,
        D_a_metric1_pair_export={**{k:v for k,v in pair_export.items() if k!='models'},
            'models':{k:{x:v[x] for x in ('episodes','layers','rows','capture_manifest_sha256')} for k,v in pair_export['models'].items()}},
        provenance=prereq,scope=dict(active='D-a through D-f',D_g='removed by user',optional_heads='See prerequisite provenance; no substitute head readouts',
            KL='full-vocabulary teacher-forced decision-token KL in nats',rendering='Original failure retained; user authorized specified data despite mismatch'),
        report_environment=dict(python=__import__('sys').version,numpy=np.__version__,matplotlib=matplotlib.__version__,torch=__import__('torch').__version__,execution='CPU-only report aggregation'),
        decision_interpretation=__import__('scripts.diag.report_decisions',fromlist=['__doc__']).__doc__,
        targeted_matching_unavailable=target_unavailable,
        uncertainty=dict(unit='episode',within_scene=True,resamples=2000,CI='95% percentile',seed=20261019),recommendation=recommended(decisions))
    args.output.mkdir(parents=True,exist_ok=False);exports=args.output/'exports';exports.mkdir()
    collector.export_episodes(exports/'metric2_episode_metrics.csv')
    sources=list((args.post_root/'capture_summary').glob('*.csv'))+list((args.post_root/'selector_results').glob('*.csv'))+list((args.post_root/'ablation_probe_results').rglob('*.csv'))+pair_files+[pair_csv]
    inventory=stage_csv_exports(sources,exports,
        Path(__file__).resolve().parents[2]/'reports/diagnostics/g3m5_failure_20261019/completed_exports')
    pair_manifest_copy=exports/'native_recall_pairs.capture_manifest.json';shutil.copyfile(pair_export['manifest_path'],pair_manifest_copy)
    inventory.append(dict(kind='json_manifest',path=str(pair_manifest_copy.relative_to(args.output)),
        source=str(Path(pair_export['manifest_path']).resolve()),sha256=sha256(pair_manifest_copy),bytes=pair_manifest_copy.stat().st_size))
    for source in [args.panel,args.prerequisites,*map(Path,prereq['command_files'])]:
        name=f'{len(inventory):03d}_{source.name}';destination=exports/name;shutil.copyfile(source,destination)
        inventory.append(dict(path=str(destination.relative_to(args.output)),source=str(source.resolve()),sha256=sha256(destination)))
    result['exports']=inventory;result['input_artifacts']={str(p.resolve()):sha256(p) for p in [args.panel,args.prerequisites,args.post_root/'capture_summary/summary.json',args.post_root/'probes/provenance.json',args.post_root/'ablation_probe_results/summary.json',args.post_root/'selector_results/results.json',Path(__file__).resolve(),Path(__file__).with_name('export_recall_pairs.py')]}
    from scripts.diag.report_figures import build_figures
    result['figures']=build_figures(result,args.output,args.post_root/'capture_summary/selection_steps.csv')
    (args.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    write_report(args.output,result)
    (args.output/'COMPLETE.json').write_text(json.dumps(dict(full_completion=full,results_sha256=sha256(args.output/'results.json'),report_sha256=sha256(args.output/'REPORT.md'),stage='requires scientific review'),indent=2)+'\n')


def write_report(output,r):
    from scripts.diag.report_tables import baseline_table,diagnostic_tables
    lines=['# G3 + M5 diagnostic report','', '## 1. Executive summary','', '| Hypothesis | Decision | Primary evidence (estimate [95% CI]) |','|---|---|---|']
    for key,value in r['decisions'].items():
        entries=[]
        for name in value['evidence_keys']:
            obj=r['decision_evidence'].get(name)
            if name in ('r3_margins','r5_overlap'):entries.extend(f'{name}/{k}: {format_metric(v)}' for k,v in obj.items())
            else:entries.append(f'{name}: {format_metric(obj)}')
        lines.append(f"| {key} | {value['status']} | {'; '.join(entries)} |")
    lines+=['','Original full-run results to explain (archived observations):']+baseline_table(r['provenance']['provenance']['baseline_results'])
    supported=[k for k,v in r['decisions'].items() if v['status']=='Supported']
    answer=('The measurements support '+', '.join(supported)+' as candidate mechanisms, while teacher-forced evidence alone does not establish the cause or size of the closed-loop regression.' if supported else 'The registered diagnostics do not identify a supported mechanism for the closed-loop regression; inconclusive evidence is not evidence of absence.')
    lines+=['',answer,'',r['recommendation'],'','D-g was removed by user instruction; no objective-interference finding is made.',f"Study completion: **{r['full_completion']}**. Scientific decisions can remain Inconclusive after a completed study.",'','## 2. Provenance','', 'All hashes, checkpoint identities, teacher/data identity and runtime versions are recorded verbatim in `results.json#/provenance` and the exported prerequisite file.','', '```json',json.dumps(r['provenance']['provenance'],indent=2),'```','','## 3. Parity gates','', '```json',json.dumps(r['provenance']['gates'],indent=2),'```','', 'The rendering mismatch remains a failed parity result. The user authorized proceeding with the specified dataset; this does not relax any model gate. Scope amendments:','',json.dumps(r['provenance'].get('scope_amendments',[]),indent=2),'','## 4. Diagnostic results']
    definitions={
        'D-a':'Metric 1 is native token-normalized self-recall and forgetting, primary Global region/layer 24. Metric 2 is a frozen-state Gaussian projection plus lag embedding, scene-held-out nested-CV ridge. Raw state cosine, current-view control and paired memory-specific differences are separate. Retrieval ranks all frames of the same episode. Anchors are every second step and lags are the registered fixed subset; lag zero has no fitted probe.',
        'D-b':'Only lower attention banks are removed; R4 and upper-layer banks remain. Target/control pairs and token counts were frozen before outcomes. Original held-out probe folds are reused without fitting. Undefined matched pairs retain explicit coverage. KL uses approved full-vocabulary decision-token distributions in nats. Optional original-head readouts are unavailable unless documented in provenance.',
        'D-c':'FP32 probability mass uses the exact causal visible keys. Anchor, visual32 and combined33 query roles are separate; primary combined33 bank8–32 per-token mass combines category mass/counts before dividing. R4 pools all eight layers equally within each episode.',
        'D-d':'Candidate CV and normalized margins are admission means. Final-plan summaries remain separate. Cross-model Jaccard and same-token score Spearman use matched layouts; empty-empty banks are excluded from behavior means. Spatial maps average episode-normalized occupancy.',
        'D-e':'Unqualified diagnostic selector swap on the fixed 600 episodes per model; original-selector repeats on 60. Failures contribute zero SR/SPL. Partial finite OS/NE have explicit coverage. Δ is random minus score; the interaction is G3_M5 Δ minus M5_E1 Δ. SR fractions multiply by 100 to obtain percentage points.',
        'D-f':'Training-pair probe insertion uses verified dummy-head invariance when original heads are unavailable. Empty-bank overlap is undefined, never a zero or a success. Layer and step results, length tertiles, logit RMS/max and argmax flips are retained; combined-model results are descriptive.'}
    for diagnostic,definition in definitions.items():
        lines+=['',f'### {diagnostic}','',definition,'',f'![{diagnostic} diagnostic figure](figures/{diagnostic}.png)','',f'All plotted values, CIs and denominators are in `results.json#/figures/{diagnostic}`; compact derived tables are in `exports/`, and full captured CSVs are losslessly packaged under `completed_exports/`.']
        if diagnostic=='D-a':
            pair_export=r['D_a_metric1_pair_export']
            lines+=['',f"The native Metric 1 pair-level export contains {pair_export['rows']:,} rows across the five models (per-model episodes, layers and row counts are in `results.json#/D_a_metric1_pair_export/models`). It is available as `completed_exports/native_recall_pairs.csv` via the split-gzip parts listed in `results.json#/exports`; reconstruct with `gzip -cd native_recall_pairs.csv.part-*.gz > native_recall_pairs.csv`. Its capture-file hashes and final CSV SHA-256 are in `completed_exports/native_recall_pairs.manifest.json`."]
        lines+=diagnostic_tables(diagnostic,r)
    lines+=['','## 5. Synthesis','',answer,'','Teacher-forced state, ablation, attention and selection findings are mechanistic associations under expert trajectories. Only D-e directly changes closed-loop behavior, and its selector intervention does not isolate every effect of training. No diagnostic alone reconstructs training dynamics or quantifies the causal share of the original regression.','','## 6. Limitations','', 'Expert teacher forcing differs from closed-loop visitation. A linear probe has a capacity ceiling; low decodability does not prove absence of information. The checkpoints use a single training seed. Dummy-head conclusions rely on the measured invariance gate. The documented rendering mismatch limits training-image equivalence. Multiple exploratory layer/lag/region comparisons are not additional preregistered hypotheses.','',*map(str,r['provenance'].get('limitations',[])),'','## 7. Reproduction','', 'Exact recorded queue commands (run from the recorded working directories with the provenance environment):','']
    for file in r['provenance']['command_files']:
        for record in read(file):lines+=['```bash',f'cd {shlex.quote(record["cwd"])}',shlex.join(record['command']),'```','']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
