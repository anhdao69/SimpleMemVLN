"""Summarize actual complete D-a1/D-c/D-d/D-f captures, never fabricate inputs.

Every inferential value is an episode mean with a within-scene episode bootstrap
(2000 draws, seed20261019). Tokens/pairs/queries are denominator metadata, not
bootstrap units. D-c averages per-query/head ratios after combining bank8–32
mass/counts; anchor, visual32 and combined33 are distinct estimands. D-d final
plan summaries are separate from admission/step means. D-f empty banks are
undefined for overlap, not zero or warmup successes. Combined-model D-f remains
descriptive; no R1–R5 verdict is produced by this partial diagnostic summary.
"""
import argparse
from collections import defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import numpy as np
from scripts.diag.statistics import episode_bootstrap,paired_bootstrap,finite
from scripts.diag.cache_teacher import sha256

MODELS=('M5_E1','G3_M5','TRIPLE','G3','G1')
REGION_TOKENS={'Q1':70,'Q2':70,'Q3':80,'Q4':80,'Global':300}


class Collector:
    def __init__(self):self.groups={}
    def add(self,meta,episode,value,weight=1,counts=None):
        if weight<0 or not math.isfinite(weight):raise ValueError('Invalid aggregation weight')
        key=tuple(sorted(meta.items()));eid=str(episode['episode_id']);scene=episode['scene_id']
        record=self.groups.setdefault(key,{}).setdefault(eid,dict(episode_id=eid,scene=scene,total=0.,weight=0.,counts={}))
        if record['scene']!=scene:raise ValueError('Episode scene identity changed')
        value=finite(value)
        if value is not None and weight:
            record['total']+=value*weight;record['weight']+=weight
        for name,n in (counts or {}).items():record['counts'][name]=record['counts'].get(name,0)+int(n)
    @staticmethod
    def observations(records):
        return [dict(episode_id=r['episode_id'],scene=r['scene'],value=r['total']/r['weight'] if r['weight'] else None) for r in records.values()]
    def summaries(self):
        out=[]
        for key,records in sorted(self.groups.items(),key=lambda pair:str(pair[0])):
            result=episode_bootstrap(self.observations(records));counts=defaultdict(int)
            for r in records.values():
                for name,n in r['counts'].items():counts[name]+=n
            out.append(dict(dict(key),**result,counts=dict(counts)))
        return out
    def contrasts(self,left,right,diagnostics=None):
        out=[]
        for key,records in self.groups.items():
            meta=dict(key)
            if meta.get('model')!=left or (diagnostics and meta.get('diagnostic') not in diagnostics):continue
            other=dict(meta,model=right);match=self.groups.get(tuple(sorted(other.items())))
            if match is None:continue  # e.g. only TRIPLE has layers8/16
            if set(records)!=set(match):raise ValueError('Paired model episode coverage differs')
            rows=[];counts=defaultdict(int)
            for eid,a in records.items():
                b=match[eid]
                if a['scene']!=b['scene']:raise ValueError('Paired model scene differs')
                rows.append(dict(episode_id=eid,scene=a['scene'],left=a['total']/a['weight'] if a['weight'] else None,
                                 right=b['total']/b['weight'] if b['weight'] else None))
                for name,n in a['counts'].items():counts['left_'+name]+=n
                for name,n in b['counts'].items():counts['right_'+name]+=n
            result=paired_bootstrap(rows);meta['model']=f'{right} - {left}'
            out.append(dict(meta,**result,counts=dict(counts)))
        return out
    def export_episodes(self,path):
        fields=sorted({k for key in self.groups for k,_ in key})+['episode_id','scene','value','aggregation_weight','counts_json']
        with Path(path).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for key,records in sorted(self.groups.items(),key=lambda p:str(p[0])):
                for r in records.values():writer.writerow(dict(dict(key),episode_id=r['episode_id'],scene=r['scene'],
                    value=r['total']/r['weight'] if r['weight'] else None,aggregation_weight=r['weight'],counts_json=json.dumps(r['counts'],sort_keys=True)))


class CsvSink:
    def __init__(self,path):self.file=Path(path).open('x',newline='');self.writer=None
    def add(self,row):
        if self.writer is None:self.writer=csv.DictWriter(self.file,fieldnames=list(row));self.writer.writeheader()
        self.writer.writerow(row)
    def close(self):self.file.close()


def attention_rows(record,strict=True):
    """Collapse ONLY query/head axes; preserve counts and three query roles."""
    roles=np.asarray(record['query_roles']);categories=dict(record['categories'])
    if strict and (len(roles)!=33 or list(roles).count('anchor')!=1 or list(roles).count('visual')!=32):
        raise ValueError('Expected anchor+32 visual queries')
    if not set(roles)<= {'anchor','visual'}:raise ValueError('Unknown query role')
    if 'bank_lag_8_15' in categories and 'bank_lag_16_32' in categories:
        a,b=categories['bank_lag_8_15'],categories['bank_lag_16_32']
        categories['bank_lag_8_32']={'mass':(np.asarray(a['mass'])+np.asarray(b['mass'])).tolist(),
            'visible_tokens':(np.asarray(a['visible_tokens'])+np.asarray(b['visible_tokens'])).tolist()}
    result=[]
    for category,values in categories.items():
        mass=np.asarray(values['mass'],dtype=np.float64);counts=np.asarray(values['visible_tokens'],dtype=np.int64)
        if mass.ndim!=2 or mass.shape[0]!=len(roles) or counts.shape!=(len(roles),) or not np.isfinite(mass).all() or np.any(counts<0):
            raise ValueError('Invalid query/head mass/count shape')
        for role in ('anchor','visual','combined'):
            selected=np.ones(len(roles),bool) if role=='combined' else roles==role
            m=mass[selected];c=counts[selected];valid=c>0
            ratios=m[valid]/c[valid,None]
            result.append(dict(role=role,category=category,mass=float(m.mean()) if m.size else None,
                mass_per_token=float(ratios.mean()) if ratios.size else None,
                n_queries=len(c),n_query_heads=m.size,n_defined_query_heads=ratios.size,
                n_visible_tokens=int(c.sum()),n_token_head_visits=int(c.sum()*mass.shape[1]),
                n_undefined_queries=int((~valid).sum())))
    return result


def validate_captures(captures,panel_path):
    panel=json.loads(Path(panel_path).read_text());ids=[str(r['episode_id']) for r in panel['episodes']]
    if set(captures)!=set(MODELS) or len(ids)!=240 or len(set(ids))!=240:raise ValueError('All5 completed frozen240 captures required')
    journals={};identities=None;common_manifest=None;verified_files=0
    for model,root in captures.items():
        root=Path(root);complete=json.loads((root/'COMPLETE.json').read_text());prov=json.loads((root/'provenance.json').read_text())
        if complete['episodes']!=240 or complete['journal_sha256']!=sha256(root/'episodes.jsonl') or prov['model']!=model or prov['panel_sha256']!=sha256(panel_path):
            raise ValueError(f'Incomplete or mismatched capture: {model}')
        if complete.get('parameter_versions_unchanged') is not True or complete['weights_sha256']!=prov['weights_sha256']:
            raise ValueError('Capture weight identity or parameter invariance missing')
        if common_manifest is None:common_manifest=prov['manifest_sha256']
        elif common_manifest!=prov['manifest_sha256']:raise ValueError('Cross-model input manifests differ')
        rows=[json.loads(line) for line in (root/'episodes.jsonl').read_text().splitlines() if line.strip()]
        if len(rows)!=240 or [str(r['episode_id']) for r in rows]!=ids:raise ValueError('Capture episode identity/order differs')
        actual=[]
        for row,expected in zip(rows,panel['episodes']):
            if row.get('parameter_versions_unchanged') is not True:raise ValueError('Episode parameter invariance missing')
            if set(row['layers'])!=set((8,16,22,24,26) if model=='TRIPLE' else (22,24,26)):
                raise ValueError('Requested recall layer coverage differs')
            if row['frames']!=expected['gt_action_count'] or Path(row['scene_id']).stem!=expected['scene']:
                raise ValueError('Frozen panel scene/length mismatch')
            if json.loads((root/row['folder']/'COMPLETE.json').read_text())!=row:raise ValueError('Episode completion record differs from journal')
            required={'attention.json.gz','attention_gate.json','selection_capture.pt','selection.json','eviction_bridge.json'}
            required|={f'recall_layer_{l}.csv' for l in ((8,16,22,24,26) if model=='TRIPLE' else (22,24,26))}
            if model in ('G3_M5','TRIPLE'):required.add('probe_drift.json')
            if not required<=row['files_sha256'].keys():raise ValueError('Missing required diagnostic artifacts')
            # Verify ALL capture files before any aggregation, including unused tensors.
            for name,expected_hash in row['files_sha256'].items():
                if sha256(root/row['folder']/name)!=expected_hash:raise ValueError(f'Changed capture file: {model}/{row["folder"]}/{name}')
                verified_files+=1
            actual.append((row['episode_uid'],row['scene_id'],row['frames']))
        if identities is None:identities=actual
        elif identities!=actual:raise ValueError('Cross-model episode UIDs/scenes/lengths differ')
        journals[model]=rows
    return panel,journals,verified_files


def recall_episode(c,model,episode,folder):
    from scripts.diag.recall import sampled_pairs,lag_bucket,lag_group
    for layer in episode['layers']:
        with (folder/f'recall_layer_{layer}.csv').open() as f:rows=list(csv.DictReader(f))
        expected={(t,s,r) for t,s in sampled_pairs(episode['frames']) for r in REGION_TOKENS}
        actual=[(int(r['anchor']),int(r['source']),r['region']) for r in rows]
        if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('Native recall pair/region coverage differs')
        for row in rows:
            lag=int(row['anchor'])-int(row['source']);region=row['region']
            if int(row['lag'])!=lag or row['lag_bucket']!=lag_bucket(lag):raise ValueError('Recall lag metadata differs')
            strata=[lag_bucket(lag)]+([lag_group(lag)] if lag_group(lag) else [])
            for label in strata:
                for metric in ('epsilon','source_epsilon','forgetting'):
                    c.add(dict(diagnostic='D-a',model=model,layer=layer,lag=label,region=region,metric=metric,scope='pair_mean'),
                          episode,float(row[metric]),counts={'pairs':1,'token_visits':REGION_TOKENS[region]})


def attention_episode(c,model,episode,folder,sink):
    if not json.loads((folder/'attention_gate.json').read_text())['passed']:raise ValueError('D-c gate failed in capture')
    with gzip.open(folder/'attention.json.gz','rt') as f:layers=json.load(f)
    if set(map(int,layers))!=set(range(3,32,4)):raise ValueError('Attention layers incomplete')
    for layer,steps in layers.items():
        if [r['step'] for r in steps]!=list(range(episode['frames'])):raise ValueError('Attention step coverage differs')
        for step in steps:
            if not step['gate']['passed']:raise ValueError('Captured D-c query block failed')
            for row in attention_rows(step):
                sink.add(dict(model=model,episode_id=episode['episode_id'],scene_id=episode['scene_id'],layer=int(layer),step=step['step'],**row))
                counts={k[2:]:v for k,v in row.items() if k.startswith('n_')}
                for metric,weight in [('mass',row['n_query_heads']),('mass_per_token',row['n_defined_query_heads'])]:
                    dims=dict(diagnostic='D-c',model=model,layer=int(layer),role=row['role'],category=row['category'],metric=metric,scope='query_head_mean')
                    c.add(dims,episode,row[metric],weight=weight,counts=counts)
                    if int(layer)<=23 and row['category']=='bank_lag_8_32':
                        c.add(dict(dims,layer='3-23'),episode,row[metric],weight=weight,counts=counts)


def selection_episode(c,model,episode,folder,sink,bridge_sink):
    dd=json.loads((folder/'selection.json').read_text())
    for layer,record in dd.items():
        means=('candidate_cv','normalized_cutoff_margin','selected_gini','distinct_source_groups','fraction_from_steps_0_2','mean_age_groups','turnover','border_share','max_source_share')
        if len(record['steps'])!=episode['frames']:raise ValueError('Selection step coverage differs')
        for step in record['steps']:
            sink.add(dict(model=model,episode_id=episode['episode_id'],scene_id=episode['scene_id'],attention_layer=int(layer),scorer_layer=int(layer)-1,
                step=step['step'],**{k:step[k] for k in means},candidate_count=step['candidate_count'],selected_count=step['selected_count'],
                spatial_histogram_json=json.dumps(step['spatial_histogram']),undefined_reasons_json=json.dumps(step['undefined_reasons'])))
            for metric in means:
                c.add(dict(diagnostic='D-d',model=model,layer=int(layer)-1,metric=metric,scope='admission_mean' if metric in ('candidate_cv','normalized_cutoff_margin') else 'step_mean'),
                    episode,step[metric],counts={'steps':1,'candidate_token_visits':step['candidate_count'],'selected_token_visits':step['selected_count'],
                                               'defined_steps':int(step[metric] is not None),'undefined_steps':int(step[metric] is None)})
        final_summary=dict(record['summary'])
        quantiles=final_summary.pop('margin_quantiles')
        final_summary.update({f'cutoff_margin_q{q}':None if quantiles is None else quantiles[i] for i,q in enumerate((0,10,50))})
        for metric,value in final_summary.items():
            if metric=='digest':continue
            c.add(dict(diagnostic='D-d',model=model,layer=int(layer)-1,metric=metric,scope='final_plan_summary'),episode,value,
                  counts={'final_tokens':record['summary']['final_rows']})
    bridge=json.loads((folder/'eviction_bridge.json').read_text())
    for layer,metric in bridge.items():
        bridge_sink.add(dict(model=model,episode_id=episode['episode_id'],scene_id=episode['scene_id'],scorer_layer=int(layer),**metric))
        c.add(dict(diagnostic='D-d',model=model,layer=int(layer),metric='score_eviction_epsilon_spearman',scope='episode_token_correlation'),
            episode,metric['value'],counts={'tokens':metric['n_tokens'],'undefined_tokens':metric['n_undefined']})


def drift_episode(c,model,episode,folder,tertile,sink):
    report=json.loads((folder/'probe_drift.json').read_text())
    if not report['dummy_invariance_passed'] or report['episode_uid']!=episode['episode_uid'] or report['frames']!=episode['frames']:
        raise ValueError('D-f failed dummy invariance or identity')
    drift=report['drift']
    for layer,record in drift['per_layer'].items():
        lookup={r['step']:r for r in record['per_step']}
        if len(lookup)!=record['nonempty_steps'] or record['empty_steps']+len(lookup)!=episode['frames']:raise ValueError('D-f step denominator differs')
        for t in range(episode['frames']):
            r=lookup.get(t);value=None if r is None else r['selected_overlap']
            sink.add(dict(model=model,episode_id=episode['episode_id'],scene_id=episode['scene_id'],layer=int(layer),scorer_layer=int(layer)-1,
                length_tertile=tertile,frames=episode['frames'],step=t,selected_overlap=value,selected_tokens=0 if r is None else r['selected'],
                intersection=None if r is None else r['intersection'],empty_warmup=r is None,probe_layers_json=json.dumps(report['probe_layers'])))
            for cohort in ('all',f'tertile_{tertile}'):
                for name in (model,'G3_M5_and_TRIPLE_descriptive'):
                    base=dict(diagnostic='D-f',model=name,layer=int(layer),length_cohort=cohort,metric='selected_overlap')
                    counts={'nonempty_steps':int(r is not None),'empty_steps':int(r is None),'selected_token_visits':0 if r is None else r['selected']}
                    c.add(dict(base,scope='episode_mean'),episode,value,counts=counts)
                    c.add(dict(base,scope='per_step',step=t),episode,value,counts=counts)
    for metric in ('rms','max_abs','argmax_flips'):
        c.add(dict(diagnostic='D-f',model=model,layer='logits',metric=metric,scope='episode_metric_mean'),episode,drift[metric],
              counts={'decisions':drift['decisions'],'logit_elements':drift['elements'],'argmax_flips':drift['argmax_flips']})
    c.add(dict(diagnostic='D-f',model=model,layer='logits',metric='argmax_flip_rate',scope='episode_metric_mean'),episode,
          drift['argmax_flips']/drift['decisions'],counts={'decisions':drift['decisions']})


def cross_selection(c,captures,journals,sink):
    import torch
    from scripts.diag.selection import compare_models
    for left,right in [('M5_E1','G3_M5'),('M5_E1','TRIPLE')]:
        for a,b in zip(journals[left],journals[right]):
            # Hashes were verified before loading these trusted local dataclasses.
            first=torch.load(captures[left]/a['folder']/'selection_capture.pt',map_location='cpu',weights_only=False)
            second=torch.load(captures[right]/b['folder']/'selection_capture.pt',map_location='cpu',weights_only=False)
            la,lb=first['layout'],second['layout']
            if la.to_dict()!=lb.to_dict() or not np.array_equal(la.visual_matrix().numpy(),lb.visual_matrix().numpy()):raise ValueError('Cross-model token layouts differ')
            da,db=first['diagnostics'],second['diagnostics']
            if da['plans'].keys()!=db['plans'].keys():raise ValueError('Cross-model selection layers differ')
            for layer in da['plans']:
                for row in compare_models(da['plans'][layer],db['plans'][layer],la.visual_matrix(),da['visual_scores'][layer],db['visual_scores'][layer]):
                    corr=row['write_score_spearman'];empty=row['left_bank_size']==row['right_bank_size']==0
                    # Empty set identity=1 is retained raw but excluded from selection-behavior means.
                    sink.add(dict(left=left,right=right,episode_id=a['episode_id'],scene_id=a['scene_id'],layer=layer,step=row['step'],
                        bank_jaccard=row['bank_jaccard'],left_bank_size=row['left_bank_size'],right_bank_size=row['right_bank_size'],
                        empty_banks=empty,score_spearman=corr['value'],n_tokens=corr['n_tokens'],undefined_reason=corr['undefined_reason']))
                    for metric,value in [('bank_jaccard',None if empty else row['bank_jaccard']),('write_score_spearman',corr['value'])]:
                        c.add(dict(diagnostic='D-d-cross',model=f'{left} vs {right}',layer=layer-1,metric=metric,scope='episode_mean'),a,value,
                              counts={'steps':1,'empty_bank_steps':int(empty),'score_tokens':corr['n_tokens'],'selected_token_visits_left':row['left_bank_size'],'selected_token_visits_right':row['right_bank_size']})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',action='append',required=True,help='MODEL=ROOT, all five models')
    p.add_argument('--panel',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    captures={}
    for text in args.capture:
        model,path=text.split('=',1)
        if model in captures:raise ValueError('Duplicate capture model')
        captures[model]=Path(path)
    panel,journals,verified=validate_captures(captures,args.panel)
    args.output.mkdir(parents=True,exist_ok=False)
    meta={str(r['episode_id']):r for r in panel['episodes']};collector=Collector()
    sinks={name:CsvSink(args.output/(name+'.csv')) for name in ('attention_steps','selection_steps','eviction_bridge_episodes','probe_drift_steps','cross_selection_steps')}
    try:
        for model in MODELS:
            for episode in journals[model]:
                folder=captures[model]/episode['folder']
                recall_episode(collector,model,episode,folder)
                attention_episode(collector,model,episode,folder,sinks['attention_steps'])
                selection_episode(collector,model,episode,folder,sinks['selection_steps'],sinks['eviction_bridge_episodes'])
                if model in ('G3_M5','TRIPLE'):
                    drift_episode(collector,model,episode,folder,meta[str(episode['episode_id'])]['length_tertile'],sinks['probe_drift_steps'])
        cross_selection(collector,captures,journals,sinks['cross_selection_steps'])
    finally:
        for sink in sinks.values():sink.close()
    collector.export_episodes(args.output/'episode_metrics.csv')
    contrasts=[]
    for left,right in [('M5_E1','G3_M5'),('M5_E1','TRIPLE'),('G1','G3')]:
        contrasts.extend(collector.contrasts(left,right,diagnostics={'D-a','D-c','D-d'}))
    summary=dict(status='partial_diagnostic_summary',hypothesis_decisions=None,
        pending_for_final_decisions=['D-a Metric2','D-b','D-e'],summaries=collector.summaries(),paired_contrasts=contrasts,
        bootstrap=dict(unit='episode',within_scene=True,resamples=2000,seed=20261019),
        definitions=__doc__,primary_R1_filter=dict(diagnostic='D-a',model='G3_M5 - M5_E1',layer=24,region='Global',metric='forgetting',lag=['B','C']),
        primary_Dc_filter=dict(diagnostic='D-c',model='G3_M5 - M5_E1',layer='3-23',category='bank_lag_8_32',metric='mass_per_token'),
        query_roles='anchor alone, visual32 alone, combined33; all retained separately',
        count_units='token_visits repeat source tokens across distinct pairs/admissions/queries; queries and query-heads separately counted',
        verified_input_files=verified,panel_sha256=sha256(args.panel),
        capture_roots={k:str(v.resolve()) for k,v in captures.items()},
        capture_complete_sha256={k:sha256(v/'COMPLETE.json') for k,v in captures.items()},
        code_sha256=sha256(__file__),output_csv_sha256={p.name:sha256(p) for p in args.output.glob('*.csv')})
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (args.output/'COMPLETE.json').write_text(json.dumps(dict(summary_sha256=sha256(args.output/'summary.json'),models=list(MODELS),episodes_per_model=240),indent=2)+'\n')


if __name__=='__main__':main()
