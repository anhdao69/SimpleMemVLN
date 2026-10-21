"""Apply original scene-held-out D-a probes to D-b; never fit or load a navigation model."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scripts.diag import probes
from scripts.diag.cache_teacher import sha256
from scripts.diag.statistics import episode_bootstrap

MODELS=('M5_E1','G3_M5','TRIPLE')
KEYS=('model','contrast','group','feature','metric')


def heldout_fold(folds,scene):
    matches=[r for r in folds if scene in r['test_scenes']]
    if len(matches)!=1 or scene in matches[0]['train_scenes']:
        raise ValueError('Exactly one held-out fold for full original scene ID required')
    return matches[0]


def feature_vector(frame,lag):
    from scripts.diag.recall import lag_embedding
    frame=np.asarray(frame,dtype=np.float32)
    if frame.ndim!=1 or not np.isfinite(frame).all():raise ValueError('Finite feature vector required')
    return np.concatenate([frame,lag_embedding([lag]).numpy()[0]])


def aggregate(rows,episodes):
    grouped={}
    for row in rows:
        key=tuple(row[k] for k in KEYS)
        grouped.setdefault(key,{}).setdefault(str(row['episode_id']),[]).append(float(row['value']))
    ep_rows=[];summaries=[]
    for key,by_episode in sorted(grouped.items()):
        entries=[]
        for episode in episodes:
            eid=str(episode['episode_id']);values=by_episode.get(eid,[])
            if not np.isfinite(values).all():raise ValueError('Nonfinite pair metric')
            item=dict(zip(KEYS,key),episode_id=eid,scene_id=episode['scene_id'],
                      value=float(np.mean(values)) if values else None,n_pairs=len(values))
            ep_rows.append(item);entries.append(dict(episode_id=eid,scene=item['scene_id'],value=item['value']))
        summaries.append(dict(zip(KEYS,key),**episode_bootstrap(entries),n_pairs=sum(map(len,by_episode.values()))))
    return ep_rows,summaries


def difference_in_differences(rows):
    groups={}
    for row in rows:
        if row['contrast']!='B0-baseline':continue
        if row['model'] not in ('G3_M5','M5_E1') or row['group'] not in ('A','C'):continue
        key=(row['episode_id'],row['scene_id'],row['contrast'],row['feature'],row['metric'])
        groups.setdefault(key,{})[(row['model'],row['group'])]=row['value']
    result=[]
    for key,values in sorted(groups.items()):
        required=[(m,g) for m in ('G3_M5','M5_E1') for g in ('A','C')]
        value=None
        if all(values.get(k) is not None for k in required):
            value=values['G3_M5','C']-values['G3_M5','A']-values['M5_E1','C']+values['M5_E1','A']
        eid,scene,contrast,feature,metric=key
        result.append(dict(episode_id=eid,scene_id=scene,contrast=contrast,feature=feature,metric=metric,value=value))
    return result


def read_json(path):return json.loads(Path(path).read_text())
def read_journal(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def write_csv(path,rows):
    with Path(path).open('x',newline='') as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def load_folds(directory,width):
    import torch
    done=read_json(directory/'COMPLETE.json')
    if done['folds']!=2 or done['pair_metrics_sha256']!=sha256(directory/'pairs.csv'):raise ValueError('Probe fit incomplete or changed')
    records=[]
    for fold in range(2):
        meta=read_json(directory/f'fold_{fold}.json');path=directory/f'fold_{fold}.pt'
        if meta['model_sha256']!=sha256(path) or meta['feature_dimensions']!=width+16 or meta['target_shape']!=[5,2048]:raise ValueError('Probe identity/dimensions differ')
        record=torch.load(path,map_location='cpu',weights_only=False)
        if record['fold']!=fold or record['test_scenes']!=meta['test_scenes'] or record['train_scenes']!=meta['train_scenes']:raise ValueError('Fold scene provenance differs')
        if set(record['train_scenes'])&set(record['test_scenes']):raise ValueError('Training/test scene leakage')
        records.append(record)
    return records


def metrics(fold,feature,lag,source,targets,prediction=None):
    if prediction is None:prediction=probes.predict(fold['model'],feature_vector(feature,lag)[None])[0]
    if not np.isfinite(prediction).all():raise ValueError('Nonfinite probe prediction')
    cos=probes.region_cosines(prediction[None],targets[source][None])[0]
    retrieval=probes.retrieval_metrics(prediction,targets,source)
    return dict(cosine=float(cos.mean()),**{f'cosine_{r}':float(v) for r,v in zip(('Q1','Q2','Q3','Q4','Global'),cos)},
                top1=float(retrieval['top1']),mrr=float(retrieval['mrr']))


def validate_ablation_identity(provenance,baseline,identity,complete_hash):
    if (provenance['baseline_complete_sha256']!=complete_hash or
        provenance['projection_manifest_sha256']!=baseline['state_projection']['cache_manifest_sha256'] or
        provenance['weights_sha256']!=baseline['weights_sha256'] or
        provenance['manifest_sha256']!=identity['manifest_sha256']):
        raise ValueError('Ablation baseline/projection/teacher identity differs')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-root','ablation-root','probe-root','teacher-cache','panel','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    import torch
    from scripts.diag.fit_probes import validate_inputs,load_tensor,pair_metadata,MODELS as ALL_MODELS
    panel=read_json(args.panel)
    captures={m:args.baseline_root/m for m in ALL_MODELS}
    journals,teachers,identity=validate_inputs(captures,panel,args.teacher_cache)
    fit_provenance=read_json(args.probe_root/'provenance.json');fit_done=read_json(args.probe_root/'COMPLETE.json')
    if fit_done.get('status')!='complete' or set(fit_done['models'])!=set(ALL_MODELS):raise ValueError('All original fitted models required')
    if fit_provenance['panel_sha256']!=sha256(args.panel) or fit_provenance['teacher_identity_sha256']!=sha256(args.teacher_cache/'identity.json'):raise ValueError('Probe panel/teacher identity differs')
    for model in ALL_MODELS:
        if fit_provenance['captures'][model]['complete_sha256']!=sha256(captures[model]/'COMPLETE.json'):raise ValueError('Probe was fitted on different baseline')
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'provenance.json',dict(probe_provenance_sha256=sha256(args.probe_root/'provenance.json'),
        teacher_identity_sha256=sha256(args.teacher_cache/'identity.json'),panel_sha256=sha256(args.panel),
        refitted=False,scalers='original training-fold statistics',retrieval_candidates='all original episode frames',
        bootstrap='2000 draws, seed20261019; paired episode means within fixed scenes',
        targeted_anchors='original selected anchors including odd anchors; no stride filter',lag0='excluded: no registered A/B/C probe'))
    all_episode_rows=[];all_summaries=[];coverage=[]
    for model in MODELS:
        base=captures[model];root=args.ablation_root/model
        provenance=read_json(root/'provenance.json');done=read_json(root/'COMPLETE.json');bp=read_json(base/'provenance.json')
        if done['episodes']!=240 or done['journal_sha256']!=sha256(root/'episodes.jsonl') or not done['adapter_gate_passed']:raise ValueError('Ablation incomplete or ungated')
        validate_ablation_identity(provenance,bp,identity,sha256(base/'COMPLETE.json'))
        ablations=read_journal(root/'episodes.jsonl')
        if len(ablations)!=240 or [r['episode_uid'] for r in ablations]!=[r['episode_uid'] for r in journals[model]]:raise ValueError('Ablation panel identity differs')
        folds={(g,f):load_folds(args.probe_root/model/g/f'{f}_layer_24',w) for g in ('A','B','C') for f,w in [('state',4096),('control',20480)]}
        destination=args.output/model;destination.mkdir();contrasts=[];model_episode_rows=[];native_rows=[]
        with (destination/'pairs.csv').open('x',newline='') as stream:
            writer=None
            def emit(common,condition,feature,values,fold):
                nonlocal writer
                row=dict(common,condition=condition,feature=feature,fold=fold,**values)
                if writer is None:writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                writer.writerow(row)
            def compare(common,left,right,label):
                for feature in ('state','control','memory_specific'):
                    for metric in left['state']:
                        lv=left[feature][metric] if feature!='memory_specific' else left['state'][metric]-left['control'][metric]
                        rv=right[feature][metric] if feature!='memory_specific' else right['state'][metric]-right['control'][metric]
                        contrasts.append(dict(model=model,episode_id=common['episode_id'],scene_id=common['scene_id'],
                            contrast=label,group=common['group'],feature=feature,metric=metric,value=rv-lv))
            for baseline,ablation in zip(journals[model],ablations):
                if baseline['scene_id']!=ablation['scene_id']:raise ValueError('Full scene ID changed')
                eid=str(baseline['episode_id']);scene=baseline['scene_id'];folder=root/ablation['folder'];bf=base/baseline['folder']
                teacher=teachers[eid];targets=load_tensor(args.teacher_cache/teacher['file'],teacher['sha256'],teacher['shape'])
                def array(where,row,name,width):return load_tensor(where/name,row['files_sha256'][name],(baseline['frames'],width))
                features={}
                for f,w,stem in [('state',4096,'projected_state'),('control',20480,'current_view')]:
                    features['baseline',f]=array(bf,baseline,f'{stem}_layer_24.pt',w)
                    features['B0',f]=array(folder,ablation,f'global_bank_off/{stem}_layer_24.pt',w)
                for group in ('A','B','C'):
                    pairs=pair_metadata([baseline],group)
                    predictions={}
                    for condition in ('baseline','B0'):
                        for feature in ('state','control'):
                            fold=heldout_fold(folds[group,feature],scene)
                            pieces=[]
                            for start in range(0,len(pairs),64):
                                batch=pairs[start:start+64]
                                x=np.stack([feature_vector(features[condition,feature][r['anchor']],r['lag']) for r in batch])
                                pieces.append(probes.predict(fold['model'],x))
                            predictions[condition,feature]=np.concatenate(pieces) if pieces else []
                    for pair_number,pair in enumerate(pairs):
                        common=dict(model=model,episode_id=eid,scene_id=scene,anchor=pair['anchor'],source=pair['source'],lag=pair['lag'],group=group,pair_index=-1,n_candidate_frames=len(targets))
                        results={}
                        for condition in ('baseline','B0'):
                            results[condition]={}
                            for f in ('state','control'):
                                fold=heldout_fold(folds[group,f],scene)
                                values=metrics(fold,features[condition,f][pair['anchor']],pair['lag'],pair['source'],targets,predictions[condition,f][pair_number])
                                results[condition][f]=values;emit(common,condition,f,values,fold['fold'])
                        compare(common,results['baseline'],results['B0'],'B0-baseline')
                native_tables=[]
                for where,row,name in [(bf,baseline,'recall_layer_24.csv'),(folder,ablation,'global_bank_off/recall_layer_24.csv')]:
                    if sha256(where/name)!=row['files_sha256'][name]:raise ValueError('Global native metric hash differs')
                    with (where/name).open() as f:table=list(csv.DictReader(f))
                    indexed={(r['anchor'],r['source'],r['region']):r for r in table}
                    if len(indexed)!=len(table):raise ValueError('Duplicate global native pair')
                    native_tables.append(indexed)
                if native_tables[0].keys()!=native_tables[1].keys():raise ValueError('Global native pair sets differ')
                for key,left in native_tables[0].items():
                    right=native_tables[1][key]
                    if left['lag_group'] not in ('A','B','C'):continue
                    contrasts.append(dict(model=model,episode_id=eid,scene_id=scene,contrast='B0-baseline',
                        group=left['lag_group'],feature='native',metric='epsilon_'+left['region'],
                        value=float(right['epsilon'])-float(left['epsilon'])))
                    native_rows.append(dict(model=model,episode_id=eid,scene_id=scene,pair_index=-1,
                        condition='B0',anchor=left['anchor'],source=left['source'],lag=left['lag'],region=left['region'],
                        epsilon=right['epsilon'],baseline_epsilon=left['epsilon'],delta=float(right['epsilon'])-float(left['epsilon'])))
                path=folder/'targeted_features.pt'
                if sha256(path)!=ablation['files_sha256']['targeted_features.pt']:raise ValueError('Targeted feature hash differs')
                targeted=torch.load(path,map_location='cpu',weights_only=True)
                if targeted['episode_uid']!=baseline['episode_uid'] or targeted['scene_id']!=scene or targeted['lag_embedding_appended']:raise ValueError('Targeted feature identity differs')
                paired={}
                for item in targeted['conditions']:
                    if not 8<=item['lag']<=32 or item['anchor']-item['source']!=item['lag']:raise ValueError('Targeted pair not registered group C')
                    common=dict(model=model,episode_id=eid,scene_id=scene,anchor=item['anchor'],source=item['source'],lag=item['lag'],group='C',pair_index=item['pair_index'],n_candidate_frames=len(targets))
                    values={}
                    for f,key in [('state','projected_state'),('control','current_view')]:
                        fold=heldout_fold(folds['C',f],scene)
                        values[f]=metrics(fold,targeted[key][item['feature_index']].numpy(),item['lag'],item['source'],targets)
                        emit(common,item['condition'],f,values[f],fold['fold'])
                    pair=paired.setdefault(item['pair_index'],dict(common=common))
                    if pair['common']!=common:raise ValueError('Target/control pair identities differ')
                    if item['condition'] in pair:raise ValueError('Duplicate targeted condition')
                    pair[item['condition']]=values
                for pair in paired.values():
                    if not {'targeted','control'}<=set(pair):raise ValueError('Missing paired intervention condition')
                    compare(pair['common'],pair['control'],pair['targeted'],'targeted-control')
                epsilon=folder/'targeted_epsilon.csv'
                if paired:
                    if sha256(epsilon)!=ablation['files_sha256']['targeted_epsilon.csv']:raise ValueError('Native metric hash differs')
                    with epsilon.open() as f:erows=list(csv.DictReader(f))
                    eg={}
                    for r in erows:
                        native_rows.append(dict(model=model,episode_id=eid,scene_id=scene,**r))
                        key=(r['pair_index'],r['region'])
                        if r['condition'] in eg.setdefault(key,{}):raise ValueError('Duplicate native pair metric')
                        eg[key][r['condition']]=float(r['epsilon'])
                    for (pair_index,region),values in eg.items():
                        if set(values)!= {'targeted','control'}:raise ValueError('Incomplete native metric match')
                        contrasts.append(dict(model=model,episode_id=eid,scene_id=scene,contrast='targeted-control',group='C',feature='native',metric='epsilon_'+region,value=values['targeted']-values['control']))
                episode_values,_=aggregate(contrasts,[baseline]);model_episode_rows.extend(episode_values);contrasts=[]
                coverage.append(dict(model=model,episode_id=eid,selected_pairs=ablation['selected_pairs'],matched_pairs=ablation['matched_pairs'],unavailable=ablation['unavailable']))
        write_csv(destination/'episode_contrasts.csv',model_episode_rows);write_csv(destination/'native_pairs.csv',native_rows)
        # Fill absent target/control matches explicitly for every episode/category.
        groups={}
        for row in model_episode_rows:groups.setdefault(tuple(row[k] for k in KEYS),{})[row['episode_id']]=row
        for key,by_id in sorted(groups.items()):
            entries=[];count=0
            for episode in journals[model]:
                eid=str(episode['episode_id']);row=by_id.get(eid)
                if row is None:row=dict(zip(KEYS,key),episode_id=eid,scene_id=episode['scene_id'],value=None,n_pairs=0)
                all_episode_rows.append(row);count+=row['n_pairs']
                entries.append(dict(episode_id=eid,scene=row['scene_id'],value=row['value']))
            all_summaries.append(dict(zip(KEYS,key),**episode_bootstrap(entries),n_pairs=count))
        del folds
    did=difference_in_differences(all_episode_rows);write_csv(args.output/'difference_in_differences_episodes.csv',did)
    did_groups={}
    for row in did:did_groups.setdefault((row['contrast'],row['feature'],row['metric']),[]).append(dict(episode_id=row['episode_id'],scene=row['scene_id'],value=row['value']))
    did_summary=[dict(contrast=k[0],feature=k[1],metric=k[2],**episode_bootstrap(rows)) for k,rows in sorted(did_groups.items())]
    write_json(args.output/'summary.json',dict(contrasts=all_summaries,difference_in_differences=did_summary,
        did_definition='(G3_M5_C-G3_M5_A)-(M5_E1_C-M5_E1_A), using B0-baseline changes',coverage=coverage))
    write_json(args.output/'COMPLETE.json',dict(models=list(MODELS),refitted=False,status='complete',summary_sha256=sha256(args.output/'summary.json')))

if __name__=='__main__':main()
