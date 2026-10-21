"""D-a Metric2 ridge fitting on complete immutable capture artifacts.

Run python -m scripts.diag.fit_probes. Requires all five complete240 captures and
complete teacher cache before fitting. One model/lag group/feature family at a
 time; pair matrices are disk-backed and held-out predictions emitted in chunks.
Original current-view control is layer24 for ALL state layers, as TASK_SPEC says.
Only diagnostic ridge weights are fitted. No navigation model is loaded.
"""
import argparse
import csv
import json
from pathlib import Path
import tempfile
import numpy as np
from scripts.diag import probes
from scripts.diag.cache_teacher import sha256

MODELS=('M5_E1','G3_M5','TRIPLE','G3','G1')


def pair_metadata(episodes,group):
    from scripts.diag.recall import sampled_pairs,lag_group,lag_bucket
    if group not in ('A','B','C'):raise ValueError('Unknown registered lag group')
    rows=[]
    for episode in episodes:
        for anchor,source in sampled_pairs(episode['frames']):
            lag=anchor-source
            if lag_group(lag)!=group:continue
            rows.append(dict(episode_id=str(episode['episode_id']),episode_uid=episode['episode_uid'],
                scene_id=episode['scene_id'],anchor=anchor,source=source,lag=lag,
                lag_group=group,lag_bucket=lag_bucket(lag)))
    return rows


def pair_features(features,pairs):
    from scripts.diag.recall import lag_embedding
    features=np.asarray(features,dtype=np.float32)
    if features.ndim!=2 or not np.isfinite(features).all():raise ValueError('Finite frame-feature matrix required')
    indices=[p['anchor'] for p in pairs]
    return np.concatenate((features[indices],lag_embedding([p['lag'] for p in pairs]).numpy()),axis=1)


def cross_fit_chunks(x,y,scenes,lambdas=probes.LAMBDAS,device='cpu',chunk_size=64):
    """Yield (fitted fold record, held-out indices, predictions); no full N×target array."""
    scenes=np.asarray(scenes,dtype=str)
    if len(scenes)!=len(x) or len(y)!=len(x) or chunk_size<1:raise ValueError('Aligned probe inputs and positive chunk size required')
    for fold,test_scenes in enumerate(probes.scene_folds(scenes,2)):
        test=np.flatnonzero(np.isin(scenes,test_scenes));train=np.flatnonzero(~np.isin(scenes,test_scenes))
        selection=probes.choose_lambda(x[train],y[train],scenes[train],lambdas,device)
        fitted=probes.fit_ridge(x[train],y[train],selection['lambda'],device)
        record=dict(fold=fold,model=fitted,selection=selection,test_scenes=test_scenes,
                    train_scenes=sorted(set(scenes[train])),n_train=len(train),n_test=len(test))
        for start in range(0,len(test),chunk_size):
            idx=test[start:start+chunk_size]
            yield record,idx,probes.predict(fitted,x[idx])


def paired_metrics(state,control):
    if any(str(state[k])!=str(control[k]) for k in ('episode_id','anchor','source')):
        raise ValueError('Memory-specific metric requires exact paired samples')
    return {f'memory_specific_{k}':float(state[k])-float(control[k]) for k in ('cosine','top1','mrr')}


def load_tensor(path,expected_sha,shape=None):
    import torch
    if sha256(path)!=expected_sha:raise ValueError(f'Input artifact hash mismatch: {path}')
    value=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(value,torch.Tensor) or (shape is not None and tuple(value.shape)!=tuple(shape)):
        raise ValueError(f'Unexpected tensor shape/type: {path}')
    if not torch.isfinite(value).all():raise ValueError(f'Nonfinite input: {path}')
    return value.float().numpy()


def validate_inputs(captures,panel,teacher_cache):
    """Read/hash gates only. Return ordered journals and teacher index."""
    ids=[str(e['episode_id']) for e in panel['episodes']]
    if len(ids)!=240 or len(set(ids))!=240 or set(captures)!=set(MODELS):raise ValueError('All five frozen240 model captures required')
    teacher_cache=Path(teacher_cache);identity=json.loads((teacher_cache/'identity.json').read_text())
    done=json.loads((teacher_cache/'COMPLETE.json').read_text())
    if done['episodes']!=240 or done['index_sha256']!=sha256(teacher_cache/'index.jsonl'):raise ValueError('Teacher cache incomplete/changed')
    teachers=[json.loads(line) for line in (teacher_cache/'index.jsonl').read_text().splitlines() if line.strip()]
    if len(teachers)!=240 or [str(r['episode_id']) for r in teachers]!=ids:raise ValueError('Teacher panel order/identity differs')
    index={str(r['episode_id']):r for r in teachers};journals={}
    for model,root in captures.items():
        root=Path(root);provenance=json.loads((root/'provenance.json').read_text());complete=json.loads((root/'COMPLETE.json').read_text())
        if complete['episodes']!=240 or complete['journal_sha256']!=sha256(root/'episodes.jsonl'):raise ValueError(f'Capture incomplete/changed: {model}')
        if provenance['model']!=model or provenance['teacher_identity_sha256']!=sha256(teacher_cache/'identity.json'):
            raise ValueError('Capture model/teacher identity differs')
        if provenance['panel_sha256']!=identity['panel_sha256'] or provenance['manifest_sha256']!=identity['manifest_sha256']:
            raise ValueError('Capture and teacher panel/manifest differ')
        rows=[json.loads(line) for line in (root/'episodes.jsonl').read_text().splitlines() if line.strip()]
        if len(rows)!=240 or [str(r['episode_id']) for r in rows]!=ids:raise ValueError('Capture panel order/identity differs')
        expected_layers=(8,16,22,24,26) if model=='TRIPLE' else (22,24,26)
        for row,entry in zip(rows,panel['episodes']):
            teacher=index[str(row['episode_id'])]
            if (row['episode_uid']!=teacher['episode_uid'] or row['frames']!=entry['gt_action_count'] or
                Path(row['scene_id']).stem!=entry['scene'] or not set(expected_layers)<=set(row['layers'])):
                raise ValueError('Capture episode identity/length/layers differ')
        journals[model]=rows
    return journals,index,identity


def write_metrics(directory,x,y,pairs,teacher_index,teacher_root,feature,layer,device,chunk_size):
    """Persist reusable folds once; stream pair metrics without accumulating predictions."""
    import torch
    directory.mkdir(parents=True,exist_ok=False);saved=set();last_id=None;targets=None
    with (directory/'pairs.csv').open('x',newline='') as stream:
        writer=None
        for record,indices,predictions in cross_fit_chunks(x,y,[p['scene_id'] for p in pairs],device=device,chunk_size=chunk_size):
            fold=record['fold']
            if fold not in saved:
                torch.save(record,directory/f'fold_{fold}.pt')
                details={k:v for k,v in record.items() if k!='model'}
                details.update(feature=feature,layer=layer,feature_dimensions=x.shape[1],target_shape=[5,2048],
                               model_file=f'fold_{fold}.pt',model_sha256=sha256(directory/f'fold_{fold}.pt'),
                               lambda_grid=list(probes.LAMBDAS),scalers_in_model_file=True)
                (directory/f'fold_{fold}.json').write_text(json.dumps(details,indent=2)+'\n');saved.add(fold)
            for idx,prediction in zip(indices,predictions):
                pair=pairs[int(idx)];eid=pair['episode_id']
                if eid!=last_id:
                    entry=teacher_index[eid];targets=load_tensor(teacher_root/entry['file'],entry['sha256'],entry['shape']);last_id=eid
                result=probes.retrieval_metrics(prediction,targets,pair['source'])
                cosines=probes.region_cosines(prediction[None],targets[pair['source']][None])[0]
                row=dict(pair,feature=feature,layer=layer,fold=fold,cosine=float(cosines.mean()),
                         **{f'cosine_{r}':float(v) for r,v in zip(('Q1','Q2','Q3','Q4','Global'),cosines)},
                         top1=result['top1'],mrr=result['mrr'],rank=result['rank'],ties=result['ties'],
                         n_candidate_frames=result['n_candidate_frames'])
                if writer is None:writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                writer.writerow(row)
    (directory/'COMPLETE.json').write_text(json.dumps(dict(pairs=len(pairs),folds=len(saved),
        pair_metrics_sha256=sha256(directory/'pairs.csv')),indent=2)+'\n')


def paired_file(state_path,control_path,destination):
    def key(row):return row['episode_id'],row['anchor'],row['source']
    with Path(control_path).open() as f:control_rows=list(csv.DictReader(f))
    control={key(r):r for r in control_rows}
    if len(control)!=len(control_rows):raise ValueError('Duplicate control pair')
    with Path(state_path).open() as source,Path(destination).open('x',newline='') as out:
        writer=None;seen=set()
        for row in csv.DictReader(source):
            k=key(row)
            if k in seen or k not in control:raise ValueError('Missing or duplicated paired control')
            seen.add(k);paired=dict(row,control_layer=24,**paired_metrics(row,control[k]))
            if writer is None:writer=csv.DictWriter(out,fieldnames=list(paired));writer.writeheader()
            writer.writerow(paired)
        if seen!=set(control):raise ValueError('Paired comparison coverage differs')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',action='append',required=True,help='MODEL=/absolute/capture/root; exactly five models')
    p.add_argument('--panel',type=Path,required=True);p.add_argument('--teacher-cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--chunk-size',type=int,default=64)
    args=p.parse_args();captures={}
    for item in args.capture:
        model,path=item.split('=',1)
        if model in captures:raise ValueError('Duplicate model capture')
        captures[model]=Path(path)
    panel=json.loads(args.panel.read_text());journals,teachers,identity=validate_inputs(captures,panel,args.teacher_cache)
    if identity['panel_sha256']!=sha256(args.panel):raise ValueError('Frozen panel hash differs')
    args.output.mkdir(parents=True,exist_ok=False)
    provenance=dict(panel_sha256=sha256(args.panel),teacher_identity_sha256=sha256(args.teacher_cache/'identity.json'),
        captures={model:{'root':str(root.resolve()),'complete_sha256':sha256(root/'COMPLETE.json')} for model,root in captures.items()},
        lambda_grid=list(probes.LAMBDAS),lag_groups=['A','B','C'],anchor_stride=2,
        current_view_control_layer=24,target_cache_dtype='float16',ridge_compute_dtype='float64',
        navigation_model_loaded=False,normalization='training-fold statistics only',
        inference_unit='episode; pair rows must be episode-aggregated before bootstrap',
        code_sha256={'runner':sha256(__file__),'probes':sha256(probes.__file__)})
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    for model in MODELS:
        root=captures[model];episodes=journals[model];by_id={str(r['episode_id']):r for r in episodes}
        layers=(8,16,22,24,26) if model=='TRIPLE' else (22,24,26)
        for group in ('A','B','C'):
            pairs=pair_metadata(episodes,group);group_dir=args.output/model/group;group_dir.mkdir(parents=True)
            with tempfile.TemporaryDirectory(prefix='matrices_',dir=group_dir) as scratch:
                scratch=Path(scratch);y=np.lib.format.open_memmap(scratch/'targets.npy',mode='w+',dtype=np.float32,shape=(len(pairs),5,2048))
                grouped={}
                for idx,pair in enumerate(pairs):grouped.setdefault(pair['episode_id'],[]).append(idx)
                for eid,indices in grouped.items():
                    entry=teachers[eid];targets=load_tensor(args.teacher_cache/entry['file'],entry['sha256'],entry['shape'])
                    y[indices]=targets[[pairs[i]['source'] for i in indices]]
                y.flush()
                for feature,layer in [('control',24)]+[('state',l) for l in layers]:
                    width=20480 if feature=='control' else 4096
                    x=np.lib.format.open_memmap(scratch/'features.npy',mode='w+',dtype=np.float32,shape=(len(pairs),width+16))
                    for eid,indices in grouped.items():
                        row=by_id[eid];name=f"{'current_view' if feature=='control' else 'projected_state'}_layer_{layer}.pt"
                        frames=load_tensor(root/row['folder']/name,row['files_sha256'][name],(row['frames'],width))
                        x[indices]=pair_features(frames,[pairs[i] for i in indices])
                    x.flush();directory=group_dir/(f'state_layer_{layer}' if feature=='state' else 'control_layer_24')
                    write_metrics(directory,x,y,pairs,teachers,args.teacher_cache,feature,layer,args.device,args.chunk_size)
                    if feature=='state':paired_file(directory/'pairs.csv',group_dir/'control_layer_24/pairs.csv',directory/'memory_specific.csv')
                    del x
                del y
    (args.output/'COMPLETE.json').write_text(json.dumps(dict(models=list(MODELS),lag_groups=['A','B','C'],status='complete'),indent=2)+'\n')


if __name__=='__main__':main()
